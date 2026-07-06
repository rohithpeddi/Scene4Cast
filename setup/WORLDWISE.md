# WorldWise Architecture — MWAE-based World Scene Graph Generation (v2e)

WorldWise is the proposed method: the top tier of the method ladder (see
[BASELINES.md](BASELINES.md)). It replaces the baselines' passive LKS buffer
with a fully differentiable **Masked World Auto-Encoder (MWAE)** — occlusion
is treated as *masking*, and occluded objects are recovered by attention over
their visible appearances — plus ego-motion encoding, explicit pair geometry,
and a tuned tail-aware objective.

**The main configuration is the round-2 winner "v2e"** (experiment stem
`worldwise_v2e`): MWAE core + pair geometry + τ=0.5 logit adjustment +
**no noisy VLM supervision** (λ_vlm=0) + 0.3 artificial masking. On predcls it
beats every baseline on R@20 (0.68 vs 0.67), mR@20 (0.50 vs 0.38), hR@20
(0.57 vs 0.49) *and* occluded-pair recall — see `docs/DECISION_LOG.md`.

Model: [lib/supervised/worldwise/worldwise.py](../lib/supervised/worldwise/worldwise.py) ·
Loss: [loss.py](../lib/supervised/worldwise/loss.py) →
[amwae_loss.py](../lib/supervised/worldwise/amwae_loss.py) ·
Config source of truth: `tools/gen_grid_configs.py` (WORLDWISE_EXTRA)

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
                      9. NodePredictor        10. reconstruction_proj (MWAE)
                                                               ▼
                 11. RelationshipPredictor (pair tokens ⊕ PAIR GEOMETRY)
                                                               ▼
                 12. TemporalEdgeAttention ─► 13. att / spatial / contacting
```

## The MWAE core

- **ScaffoldTokenizer** ([scaffold_tokenizer.py](../lib/supervised/worldwise/scaffold_tokenizer.py)) —
  top-down tokenization: every object gets a token at every frame. Visible
  objects carry their projected ROI features; unseen objects a learnable
  **[MASK]** embedding; geometry/camera/motion/ego features are fused in
  either case. During training, `p_mask_visible = 0.3` of *visible* objects
  are artificially masked — this is a load-bearing choice: the masking acts
  as a tail-class augmentation as well as occlusion training (reducing it to
  0.1 cost ~5 mR in round 2). Reconstruction targets come from an **EMA copy
  of the visual projector** (`use_ema_recon_target: true`).
- **AssociativeRetriever** ([associative_retriever.py](../lib/supervised/worldwise/associative_retriever.py)) —
  per-object bidirectional cross-attention: each slot's masked tokens query
  that slot's *visible* tokens across all T frames, with view-aware Q/K
  biases from camera features. The differentiable replacement for the
  baselines' zero-order-hold copy.
- **Visibility embedding** — flags observed vs recovered tokens downstream.
- **Ego-motion (`CameraTemporalEncoder`)** — relative camera poses (6D rot +
  translation) per step, self-attended over the sequence, broadcast into
  every object token.
- **Pair geometry (I-1)** — an explicit relative-3D-geometry vector (unit
  direction, log center distance, both log-volumes, ratio, log min
  corner-gap) appended to every relation token (`use_pair_geometry: true`).

## Loss — the v2e recipe

`WorldWiseLoss` = `AMWAELoss` + tail-aware logit adjustment, with two
decisive settings found in round 2:

1. **Scene-graph loss on visible pairs only** (`lambda_vlm: 0.0`). The noisy
   VLM pseudo-labels on unseen pairs are **not used** — round 2 showed they
   were actively mis-teaching (removing them: +4 R, +9 mR, and **+9.5
   occluded-pair recall**). Unseen pairs receive *no direct edge
   supervision*; the recovery pathway is trained entirely by:
2. **Simulated-unseen + reconstruction objectives** — clean-GT supervision on
   the artificially masked (visible) pairs, plus MSE reconstruction of their
   EMA-target features. Simulated occlusion with clean labels generalizes to
   real occlusion better than direct-but-noisy supervision.
3. **Tail-aware logit adjustment at τ=0.5** (Menon et al., 2021): per-class
   log-prior offsets added to relation logits at train time only (priors from
   `tools/compute_predicate_priors.py`, mode-specific files). τ traces a
   clean R↔mR Pareto front: 0.5 is the balanced point; **0.75 is the
   published mR-max operating point** (`v2f`); 1.0 is unstable (round-1
   collapse on occluded pairs at resnet50).

## Component-wise ablation set (Table C)

Every ablation changes exactly one thing relative to the main config; all run
@ dinov3l, both modes (`--stages abl`). Three keep historical tier names so
already-trained cells are reused:

| Tier | Changes | Ablates |
|---|---|---|
| `v2a` | λ_vlm 0 → 0.2 | value of REMOVING noisy VLM supervision |
| `v2f` | τ 0.5 → 0.75 | the R↔mR operating point |
| `abl_notau` | logit adjustment off | the tail-aware objective |
| `abl_nomask` | p_mask 0.3 → 0 (recon/sim vanish) | the MWAE self-supervision |
| `abl_noema` | EMA target → live projector | reconstruction-target stability |
| `v2g` | pair geometry off | I-1's contribution |
| `abl_nospatial` | ObjectSpatialEncoder off | camera-frame features |
| `abl_noego` | CameraTemporalEncoder off | ego-motion |
| `abl_nomotion` | ObjectMotionEncoder off | object dynamics |
| `abl_notempedge` | TemporalEdgeAttention off | cross-frame edge reasoning |

## Retired plugins (round-2 gate, docs/DECISION_LOG.md)

Historical context — these were evaluated and failed; their flags remain in
the code (all `false` in configs) and their modules in the tree:
soft text embedding (I-2, unobservable in predcls / hurt sgdet), geometric
attention bias (I-3, −4 R), confidence-weighted VLM (I-5, no data),
predicate prototypes (I-6, catastrophic collapse), cross-object retrieval
(I-7, hurt its own occpair target), energy refinement (I-8, −5 R). The
campaign's conclusion: **the wins came from the loss side, not from adding
architecture.**

## What separates WorldWise (v2e) from W-DSGDetr++ (strongest baseline)

1. MWAE mask-and-retrieve memory (differentiable) vs zero-order-hold copy.
2. Ego-motion encoding.
3. Simulated-unseen + reconstruction objectives *instead of* noisy VLM
   supervision on unseen pairs (baselines keep λ_vlm=0.2).
4. Tail-aware logit adjustment (τ=0.5).
5. Explicit pair geometry in relation tokens.
