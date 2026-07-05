# WorldWise Architecture — MWAE-based World Scene Graph Generation

WorldWise is the proposed method: the top tier of the nested ladder (see
[BASELINES.md](BASELINES.md)). It replaces the baselines' passive LKS buffer
with a fully differentiable **Masked World Auto-Encoder (MWAE)** — occlusion is
treated as *masking*, and occluded objects are recovered by attention over
their visible appearances — and adds ego-motion encoding plus a tail-aware
training objective. It runs at all four backbones (resnet50, dinov2b, dinov2l,
dinov3l); the plugin tiers run at dinov3l.

Model: [lib/supervised/worldwise/worldwise.py](../lib/supervised/worldwise/worldwise.py) ·
Loss: [loss.py](../lib/supervised/worldwise/loss.py) →
[amwae_loss.py](../lib/supervised/worldwise/amwae_loss.py)

## Forward pipeline (single batched pass, B = T frames)

```
corners ──► 1. GlobalStructuralEncoder ─────────────┐
poses ────► 2. ObjectSpatialEncoder (per-object)    │
        └─► 3. CameraTemporalEncoder (ego-motion)   ├─► 5. ScaffoldTokenizer
corners ──► 4. ObjectMotionEncoder (vel/accel)      │      (mask + fuse)
ROI feats ──────────────────────────────────────────┘          │
                                                               ▼
                                            6. AssociativeRetriever (per-object
                                               cross-attn over visible frames)
                                                               ▼
                             7. + VisibilityEmbedding ─► 8. InterObjectTransformer
                                                               ▼
                                        (8b. EnergyDiffusion — plugin I-8)
                                                               ▼
                      9. NodePredictor        10. reconstruction_proj (MWAE)
                                                               ▼
                  11. RelationshipPredictor.batched_form_and_attend
                                                               ▼
                  12. TemporalEdgeAttention  ─► (12b. PrototypeMemory — I-6)
                                                               ▼
                  13. batched_predict → attention / spatial / contacting
```

## The MWAE core (WorldWise-exclusive)

- **ScaffoldTokenizer** ([scaffold_tokenizer.py](../lib/supervised/worldwise/scaffold_tokenizer.py)) —
  top-down tokenization from the world scaffold. Visible objects get their
  projected ROI features; unseen objects get a learnable **[MASK]** embedding;
  geometry/camera/motion/ego features are fused in either case. During
  training, a fraction `p_mask_visible` (default 0.3) of *visible* objects is
  artificially masked, creating self-supervised recovery targets.
  Reconstruction targets come from an **EMA copy of the visual projector**
  (`use_ema_recon_target`, default on — a stable data2vec-style target rather
  than the live projector's moving output).
- **AssociativeRetriever** ([associative_retriever.py](../lib/supervised/worldwise/associative_retriever.py)) —
  per-object bidirectional cross-attention: each slot's masked tokens query
  that slot's *visible* tokens across all T frames (K/V restricted by
  visibility), with view-aware Q/K biases from camera features and iterative
  K/V refinement across layers. This is the differentiable replacement for the
  baselines' zero-order-hold copy.
- **Visibility embedding** — a 2-way embedding added after retrieval so
  downstream layers know which tokens are real observations vs recovered.
- **Ego-motion (`CameraTemporalEncoder`)** — relative camera poses (6D rotation
  + translation) per step, self-attended over the full sequence; broadcast into
  every object token.

## Loss — `WorldWiseLoss` = AMWAE triple objective + tail-aware adjustment

`AMWAELoss` (in `amwae_loss.py`):
1. **Scene-graph loss** — same visible/unseen bucketing as the baselines'
   LKSLoss (λ_vlm-weighted, smoothed unseen bucket).
2. **Reconstruction loss** — MSE between retrieved tokens (projected back to
   visual space) and the EMA-target features, on *artificially masked* tokens
   only.
3. **Simulated-unseen loss** — clean-GT supervision on the artificially masked
   pairs.
4. (I-8 only) **Attractor stability loss** — MSE between the energy refiner's
   final and penultimate states (`lambda_stability`).

`WorldWiseLoss` adds **tail-aware logit adjustment** (Menon et al., 2021):
per-class log-prior offsets added to the relation logits *at train time only*
(`use_logit_adjustment`, `logit_adjustment_tau`; priors from
`tools/compute_predicate_priors.py`). Inference logits are untouched, so rare
predicates are favoured at test time — this targets mR@K directly.

## Ablation flags (MWAE component study)

`use_object_spatial_encoder`, `use_camera_temporal`,
`use_object_motion_encoder`, `use_temporal_edge_attn` — all default **on**;
full WorldWise = all enabled.

## WorldWise⁺ plugins (all default OFF except I-4)

With every plugin flag off, the architecture is the post-fix **I-0** reference.

| Flag | Plugin | Mechanism | Target metric |
|---|---|---|---|
| `use_pair_geometry` | **I-1** | Explicit relative-3D-geometry vector (unit direction, log center distance, both log-volumes, ratio, log min corner-gap) → MLP → appended to pair tokens | spatial R/mR |
| `use_soft_text_embedding` | **I-2** | Text features = expected CLIP embedding under softmax(node logits / τ) — differentiable; GT override still wins in predcls | contacting mR |
| `use_geometric_attn_bias` | **I-3** | `BiasedSpatialGNN`: pairwise geometry → per-head additive attention-logit bias (Graphormer-style), replacing the pooled spatial PE | spatial R/mR |
| `use_ema_recon_target` | **I-4** | EMA reconstruction target (bug fix; default ON — the `noema` tier is the A/B control) | occluded-pair recall |
| `use_confidence_weighted_vlm` | **I-5** | Per-pair VLM-confidence weights on the unseen bucket (in `amwae_loss.py`). **Inert** until the dataset provides a `vlm_confidence` (T, K) tensor | unseen-pair mR |
| `use_predicate_prototypes` | **I-6** | [prototype_memory.py](../lib/supervised/worldwise/prototype_memory.py): EMA class prototypes of pair tokens (updated from GT during training), read back via gated cross-attention before the contacting head (TEMPURA-style) | contacting mR |
| `use_cross_object_retrieval` | **I-7** | Retriever attends across ALL objects' visible tokens (slot-identity embeddings on Q/K; padding slots pruned so cost scales with real object count) | masked-pair recall |
| `use_energy_refinement` | **I-8** | [energy_diffusion.py](../lib/supervised/worldwise/energy_diffusion.py): weight-tied recurrent transformer iterated to convergence over object tokens (spatial PE re-injected each step) + stability loss | with-constraint R |

Cumulative Stage-A tiers: `plus1` = {I-1}, `plus2` = {I-1,2}, `plus3` =
{I-1,2,3}; Stage-B research plugins (`conf`, `proto`, `xobj`, `energy`) run
one-at-a-time on the Stage-A winner. Gate: ≥ +0.3 on the plugin's target
metric at dinov3l without degrading the other task
(see `docs/DECISION_LOG.md`).

## What separates WorldWise from W-DSGDetr++ (the strongest baseline)

1. MWAE mask-and-retrieve memory (differentiable) vs zero-order-hold copy.
2. Ego-motion encoding (CameraTemporalEncoder).
3. Reconstruction + simulated-unseen objectives.
4. Tail-aware logit adjustment.
5. The plugin surface above.
