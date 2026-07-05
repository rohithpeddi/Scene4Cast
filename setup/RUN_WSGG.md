# Running WSGG Training & Evaluation (Baselines + WorldWise)

End-to-end recipe for the supervised WSGG campaign. Architecture references:
[BASELINES.md](BASELINES.md), [WORLDWISE.md](WORLDWISE.md). A deeper
feature-extraction walkthrough lives in `docs/METHODS_README.md` (note:
`docs/` is git-ignored — that file exists on machines where it was authored,
not in fresh clones).

**Campaign design (July 2026):** baselines run only at **resnet50** (method
comparison at the common backbone); WorldWise runs at **all four backbones**
(scaling story); WorldWise⁺ plugin tiers run at **dinov3l**. 2 tasks
(predcls, sgdet), 1 seed (0), identical budget for every cell → **32 runs**.

## 0. Prerequisites

- ActionGenome4D at `{data_path}` (default `/data/rohith/ag`) with `frames/`,
  `annotations/`, `world_annotations/`, `world4d_rel_annotations/{train,test}/`,
  and `features/clip_features/clip_text_embeddings.npy`.
- Trained detector checkpoints per backbone (see [RUN_MON3D.md](RUN_MON3D.md)).
- Environment: torch + CUDA.
- **wandb** (`use_wandb: true` in every config): all cells log to one shared
  project `wandb_project` (default `worldsgg-v2`), each run named by
  `experiment_name` and grouped by `method_name`, so the ladder and plugin
  tiers overlay on one dashboard. **Before a multi-process launch, authenticate
  once** (`wandb login` or `export WANDB_API_KEY=…`) — an unauthenticated
  `wandb.init` can block on a prompt and hang every concurrent slot. To skip
  the network entirely: `export WANDB_MODE=offline`, or set `use_wandb: false`
  (metrics still land in `results/*_metrics.jsonl`, the source of truth).

## 1. One-time feature extraction (per backbone × mode)

Converts frames + detector checkpoints into per-video PKLs
(`{data_path}/features/roi_features/<mode>/<backbone>/{train,test}/`) holding
ROI features, boxes, 3D corners, labels, pair indices, union features.

```bash
# predcls (GT boxes) / sgdet (detector boxes) — one config per backbone
python datasets/preprocess/features/extract_roi_features_predcls.py \
    --config configs/features/predcls/ex_roi_feat_resnet50_predcls_rohith.yaml
python datasets/preprocess/features/extract_roi_features_sgdet.py \
    --config configs/features/sgdet/ex_roi_feat_resnet50_sgdet_rohith.yaml
# ... repeat with the dinov2b / dinov2l / dinov3l feature configs
```

Needed combinations for this campaign: **resnet50 × both modes** (all five
methods) and **dinov2b/dinov2l/dinov3l × both modes** (WorldWise only).

## 2. One-time setup on the training server

```bash
# a. Smoke-test every model + plugin flag (forward/backward, no data needed)
python tests/test_wsgg_forward.py

# b. Param-count freeze artifact (asserts the ladder is monotone)
python tools/dump_param_counts.py --out docs/param_counts.md

# c. Regenerate configs (16 base + 16 tier = 32, deletes stale ones)
python tools/gen_grid_configs.py --tiers

# d. Predicate priors (used by WorldWise's tail-aware logit adjustment)
python tools/compute_predicate_priors.py --data_path /data/rohith/ag \
    --feature_model dinov2b --mode predcls   # and --mode sgdet
```

## 3. Training

### Single run

```bash
python train_wsgg_methods.py --config configs/methods/predcls/worldwise_predcls_dinov3l.yaml
# any YAML key can be overridden on the CLI, e.g. --lr 5e-5 --nepoch 10
```

Checkpoints: `{save_path}/{experiment_name}/checkpoint_N/checkpoint_state.pth`
(+ `best_model.pth` at the best wc/R@20). Resume with `--ckpt checkpoint_N`.

### Full campaign on 3 GPUs (recommended)

```bash
python tools/run_grid_multigpu.py --gpus 0 1 2 --compute-priors
```

One worker per GPU pulls from a shared, priority-ordered queue:

**Post round-2, the default schedule is trimmed to what still matters**
(completed cells are skip-detected, so the default command re-runs nothing):

| Stage | Cells | Status |
|---|---|---|
| `table_a` | 5 methods × resnet50 × 2 modes (10) | ACTIVE — the ladder table |
| `scaling` | worldwise × dinov2b/2l/3l × 2 modes (6) | ACTIVE — Table B + tier I-0 ref |
| `v2` | v2a…v2f × {resnet50, dinov3l} × 2 modes (24) | ACTIVE — final-ladder candidates (v2e/v2f are the live ones) |
| `stage_a` | plus1/2/3 + noema (8) | RETIRED — verdicts logged (plus1 KEEP, rest DROP) |
| `tune` | τ sweep + plus3pe (8) | RETIRED — answered by v2 differencing |
| `subtract` | notau/lowmask/… (12) | RETIRED — answered; nomotion/noego optional |
| `stage_b` | conf/proto/xobj/energy (8) | RETIRED — all failed the gate (proto catastrophic) |

Retired stages still run if named explicitly (`--stages subtract`) — the
launcher prints why they were retired.

Final tables once a v2 winner is chosen (winner also needs dinov2b/2l runs via
`python tools/run_grid.py --methods worldwise --tiers v2e --backbones dinov2b dinov2l`):

```bash
python tools/aggregate_results.py --mode predcls --tiers --v2 v2e   # ladder row + v2 scaling
python tools/report_gate_metrics.py --mode predcls                  # occpair check on λ_vlm=0
```

- Per-run logs: `logs/grid/<experiment>.log`; live status:
  `results/grid_run_status.csv`.
- **Resume-safe**: completed runs (final-epoch row present in the metrics
  jsonl) are skipped — just re-run the same command after interruptions.
- Subsets: `--stages table_a scaling stage_a`, `--modes predcls`, `--dry-run`.
- Sequential single-GPU alternative: `tools/run_grid.py` (same cells,
  `--tiers plus1 ...` for tier configs).

## 4. Evaluation

Every training run already evaluates after each epoch (with-constraint +
no-constraint + occlusion-stratified). Standalone testing of a checkpoint:

```bash
python test_wsgg_methods.py \
    --config configs/methods/predcls/worldwise_predcls_dinov3l.yaml \
    --ckpt checkpoint_19
```

## 5. Results & aggregation

Each run appends to `results/<experiment_name>_metrics.jsonl`
(experiment names carry the `_v2` suffix — `_v1` results predate the
July 2026 fixes and are not comparable):

- a one-time **header row** (git commit, full config incl. plugin flags,
  param counts);
- one row per epoch: `wc|nc / R|mR|hR @ {10,20,50,100}`, the per-predicate
  recall vector at K=20 (`wc/per_predicate_R@20`), occlusion-stratified
  `vispair/*` and `occpair/*` metrics, all loss sub-terms (`loss/*`),
  `epoch_time_s`, `peak_vram_gb`.

Render the three tables (best epoch by wc/R@20 per cell):

```bash
python tools/aggregate_results.py --mode predcls --tiers   # Tables A, B, C + CSV
python tools/aggregate_results.py --mode sgdet  --tiers
```

- **Table A** — methods @ resnet50 · **Table B** — WorldWise across backbones ·
  **Table C** — WorldWise⁺ plugin ladder @ dinov3l.

## 6. Decision gate

A plugin survives if it moves its **target metric** by ≥ +0.3 @ dinov3l
without degrading the other task (targets per plugin in
[WORLDWISE.md](WORLDWISE.md)). Record every verdict in
`docs/DECISION_LOG.md` — hypothesis,
run IDs, deltas, keep/drop, follow-up.

## Troubleshooting

- `worldwise_conf_*` (I-5) warns "no vlm_confidence tensor" — expected: the
  plugin is inert until features are regenerated with per-label VLM
  confidences; it falls back to the global λ_vlm.
- All-plugin-flags-off WorldWise must match the I-0 reference —
  `tests/test_wsgg_forward.py` asserts no plugin modules exist in that case.
- Three concurrent runs share CPU comfortably (`batch_size=1`,
  `num_workers=0`) but each holds its feature set in RAM — watch memory at
  campaign start.
