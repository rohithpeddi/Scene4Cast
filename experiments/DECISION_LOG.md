# WorldWise⁺ Decision Log

Append-only record of gate decisions for the plugin round. One entry per
decision: hypothesis → runs → observed deltas → verdict → follow-up. This file
plus the per-predicate/occlusion tables in `results/*_metrics.jsonl` are the
inputs to the next architecture-refinement round.

Gate rule (Stage A/B): a plugin survives if it moves its **target metric** by
≥ +0.3 at the hero backbone (dinov3l) without degrading the other task.
Target metrics — I-1/I-3: wc spatial R@K & mR@K · I-2/I-5/I-6: contacting
mR@K · I-7: occpair (masked-pair) R@K · I-8: wc R@K.

Plugin flags all live in the WorldWise config; `worldwise` with every plugin
flag off (and `use_ema_recon_target: true`) is the I-0 reference.

---

## 2026-07-04 — Round setup (code, no runs yet)

**Change set:** baselines fixed & frozen (velocity gating in W-DSGDetr++,
dead camera params removed from W-STTran, predcls GT text pathway for ALL
methods); WorldWise fixes (EMA reconstruction target, union features now fed
in training, predcls GT text); plugins I-1..I-8 implemented behind `use_*`
flags; tier configs + logging infrastructure added.

**Restructure:** the git-untracked `lib/supervised/worldsgg/` shared package
was dissolved — `worldsgg_base.py` → `lib/supervised/components.py`, LKS
buffer → `lib/supervised/baselines/lks_buffer/`, MWAE pieces flattened into
`lib/supervised/worldwise/` (`amwae_loss.py`). Legacy standalone models
(AMWAE, AMWAE++, LKSGNN, gl_stgn) and dead `config.py`/`funcs.py` removed —
zero active importers, verified by grep. No import references
`lib.supervised.worldsgg` anymore, so the server's old untracked copy at that
path is dead code (safe to delete there; it no longer conflicts with pulls).

**Consequence:** every `_v1` result is invalidated — the fixes change model
behavior for baselines and WorldWise alike. All experiment names bumped to
`_v2`; the full grid must be regenerated under the frozen code before any
comparison is made. Tags `baselines-v1-frozen` / `worldwise-v1-fixed` to be
applied after the server smoke tests pass.

**Known inert plugin:** I-5 (`use_confidence_weighted_vlm`) — the dataset has
no per-label VLM confidence fields; the loss falls back to global λ_vlm with a
one-time warning. Activating it requires regenerating features with
confidences and passing a `vlm_confidence` (T, K_max) tensor in the batch.

**Next:** server smoke tests (`python tests/test_wsgg_forward.py`,
`python tools/dump_param_counts.py`) → tag → regenerate configs
(`python tools/gen_grid_configs.py --tiers`) → Stage A runs.

---

<!-- Template for gate entries:

## YYYY-MM-DD — <tier/plugin> gate

**Hypothesis:**
**Runs:** experiment names + epochs selected (wc/R@20-best)
**Deltas vs parent tier:** (target metric first; both tasks)
**Occlusion split / per-predicate notes:**
**Verdict:** KEEP / DROP
**Follow-up:**
-->
