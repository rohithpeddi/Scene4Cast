"""
Generate the WorldSGG hierarchy-experiment config grid.

Grid structure (July 2026 restructure):
  baselines  x resnet50 only    — method-comparison happens at the common backbone
  worldwise  x all 4 backbones  — backbone-scaling story is WorldWise-only

  = 4 baselines x 1 backbone x 2 tasks (8)
  + worldwise x 4 backbones x 2 tasks (8)   → 16 base configs (1 fixed seed each)

  methods   : w_sttran, w_sttran_pp, w_dsgdetr, w_dsgdetr_pp, worldwise
  backbones : resnet50, dinov2b, dinov2l, dinov3l   (feature_model dir name)
  tasks     : predcls, sgdet

The five methods form a strict nested capability ladder (see the model files);
the ladder is realised in code, so configs differ only by method_name / mode /
feature_model / experiment_name (+ WorldWise-only MWAE & tail-loss keys).

Experiment version is _v2: the Phase-1/2 fixes (velocity gating, predcls GT
text pathway, EMA reconstruction target, WorldWise union features) change
model behavior, so _v1 results are not comparable and every cell must be
regenerated under the frozen code.

Writes to configs/methods/<mode>/<method>_<mode>_<backbone>.yaml

WorldWise⁺ plugin tiers (--tiers): cumulative Stage-A ladder + Stage-B
research plugins at the hero backbone, written as
configs/methods/<mode>/worldwise_<tier>_<mode>_<backbone>.yaml

Run (stdlib only, no torch needed):
    python tools/gen_grid_configs.py           # 40-cell base grid
    python tools/gen_grid_configs.py --tiers   # + WorldWise⁺ tier configs
"""

import argparse
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

METHODS = ["w_sttran", "w_sttran_pp", "w_dsgdetr", "w_dsgdetr_pp", "worldwise"]
BACKBONES = ["resnet50", "dinov2b", "dinov2l", "dinov3l"]
MODES = ["predcls", "sgdet"]
HERO_BACKBONE = "dinov3l"        # WorldWise⁺ tiers run here
METHODS_BACKBONE = "resnet50"    # baselines run here only (method comparison)
VERSION = "v2"

SEED = 0


def backbones_for(method):
    """Baselines run only at the common backbone; WorldWise scales across all."""
    return BACKBONES if method == "worldwise" else [METHODS_BACKBONE]

# Keys shared by every method (ladder is realised in code, not via flags).
COMMON = [
    ("data_path", "/data/rohith/ag"),
    ("save_path", "/data/rohith/ag/checkpoints"),
    ("results_path", "results"),
    ("ckpt", None),
    ("world_sg_dir", ""),
    ("seed", SEED),
    # ---- Shared architecture ----
    ("d_model", 256),
    ("d_struct", 256),
    ("d_visual", 256),
    ("d_detector_roi", 1024),
    ("d_camera", 128),
    ("d_motion", 64),
    ("n_heads", 4),
    ("d_feedforward", 512),
    ("max_objects", 64),
    ("dropout", 0.1),
    # ---- Relationship predictor ----
    ("d_rel", 256),
    ("d_text", 128),
    ("d_union_roi", 1024),
    ("n_rel_layers", 2),
    ("n_rel_heads", 4),
    ("n_temporal_edge_layers", 1),
    ("n_temporal_obj_layers", 2),
    ("clip_embeddings_path", "features/clip_features/clip_text_embeddings.npy"),
    # ---- Training (identical budget across the whole grid) ----
    ("nepoch", 20),
    ("lr", 0.0001),
    ("weight_decay", 0.0001),
    ("warmup_fraction", 0.1),
    ("grad_clip", 5.0),
    ("optimizer", "adamw"),
    ("bce_loss", True),
    ("use_wandb", True),
    ("wandb_project", "worldsgg-v2"),
    ("use_amp", True),
    ("log_every", 100),
    ("include_invisible", True),
    ("datasize", "large"),
    ("skip_test", False),
    ("n_gnn_layers", 3),
    # ---- Loss ----
    ("lambda_vlm", 0.2),
    ("label_smoothing_vlm", 0.2),
]

# WorldWise-only keys: MWAE core + tail-aware logit adjustment.
WORLDWISE_EXTRA = [
    ("n_cross_attn_layers", 2),
    ("n_self_attn_layers", 3),
    ("d_memory", 256),
    ("lambda_reconstruction", 0.5),
    ("lambda_recon_dominance", 0.1),
    ("p_simulate_unseen", 0.3),
    ("p_mask_visible", 0.3),
    ("use_object_spatial_encoder", True),
    ("use_camera_temporal", True),
    ("use_object_motion_encoder", True),
    ("use_temporal_edge_attn", True),
    # Tail-aware loss (WorldWise-exclusive)
    ("use_logit_adjustment", True),
    ("logit_adjustment_tau", 1.0),
    # predicate_priors_path is mode-specific — filled in by render()
    ("predicate_priors_path", "features/predicate_priors_{mode}.json"),
    # ---- WorldWise⁺ plugin flags (I-0 = everything off; I-4 fix = on) ----
    ("use_ema_recon_target", True),
    ("use_pair_geometry", False),
    ("use_soft_text_embedding", False),
    ("use_geometric_attn_bias", False),
    ("attn_bias_keep_pe", False),
    ("use_confidence_weighted_vlm", False),
    ("use_predicate_prototypes", False),
    ("use_cross_object_retrieval", False),
    ("use_energy_refinement", False),
    ("lambda_stability", 0.0),
]

# WorldWise⁺ tiers — Stage A is a strict cumulative ladder (mirrors the
# original nested-method construction); Stage B are one-at-a-time research
# plugins layered on whatever Stage A survivors define WorldWise-v2.
WW_PLUS_TIERS = {
    # Stage A — cumulative
    "plus1": {"use_pair_geometry": True},
    "plus2": {"use_pair_geometry": True, "use_soft_text_embedding": True},
    "plus3": {"use_pair_geometry": True, "use_soft_text_embedding": True,
              "use_geometric_attn_bias": True},
    # I-3 retry (round-1 P6): keep the spatial PE AND add the attention bias
    # (the plus3 R-drop is suspected to be the lost PE, not the bias). Built
    # on plus1 only — I-2 is a provisional drop from round 1.
    "plus3pe": {"use_pair_geometry": True, "use_geometric_attn_bias": True,
                "attn_bias_keep_pe": True},
    # A/B control for the Phase-2 EMA-target fix (predcls only is enough)
    "noema": {"use_ema_recon_target": False},
    # τ sweep (round-1 P1): the τ=1.0 default trades ~13 R@20 points for the
    # mR gain — find the knee of the R/mR curve.
    "tau025": {"logit_adjustment_tau": 0.25},
    "tau05":  {"logit_adjustment_tau": 0.5},
    "tau075": {"logit_adjustment_tau": 0.75},
    # Subtraction tiers (round-1.5): WorldWise carries training pressures the
    # baselines don't — test whether REMOVING them recovers R@K. Each removes
    # exactly one thing from I-0.
    "notau":   {"use_logit_adjustment": False},        # no tail rebalancing at all
    "lowmask": {"p_mask_visible": 0.1},                # gentler artificial masking
    "nomask":  {"p_mask_visible": 0.0,                 # no artificial masking →
                "p_simulate_unseen": 0.0},             #   recon/sim losses vanish
    "novlm":   {"lambda_vlm": 0.0},                    # drop noisy unseen-pair labels
    "nomotion": {"use_object_motion_encoder": False},  # component subtraction
    "noego":    {"use_camera_temporal": False},        # component subtraction
    # Stage B — research plugins (run one at a time on the Stage-A winner)
    "conf":   {"use_confidence_weighted_vlm": True},
    "proto":  {"use_predicate_prototypes": True},
    "xobj":   {"use_cross_object_retrieval": True},
    "energy": {"use_energy_refinement": True, "lambda_stability": 0.1},
    # ---- WorldWise-v2 recomposition candidates (round 2) ----
    # Each = I-0 + I-1 (the only R-positive plugin) + tuned training pressures.
    # They bracket the expected optimum so ONE campaign settles the final
    # composition: v2a = mild fix (τ only) · v2b = τ + gentler masking ·
    # v2c = R-leaning (low τ) · v2d = v2b + drop noisy VLM labels.
    "v2a": {"use_pair_geometry": True, "logit_adjustment_tau": 0.5},
    "v2b": {"use_pair_geometry": True, "logit_adjustment_tau": 0.5,
            "p_mask_visible": 0.1},
    "v2c": {"use_pair_geometry": True, "logit_adjustment_tau": 0.25,
            "p_mask_visible": 0.1},
    "v2d": {"use_pair_geometry": True, "logit_adjustment_tau": 0.5,
            "p_mask_visible": 0.1, "lambda_vlm": 0.0},
}

# v2 candidates compete for the FINAL ladder, so unlike diagnostic tiers they
# are generated at every backbone: the ladder table lives at resnet50 and the
# winner also fills Table B (all backbones).
V2_CANDIDATES = ["v2a", "v2b", "v2c", "v2d"]


def fmt(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return "''" if v == "" else v
    return repr(v) if isinstance(v, float) else str(v)


def render(method, mode, backbone, tier=None):
    stem = f"{method}_{tier}" if tier else method
    exp = f"{stem}_{mode}_{backbone}_{VERSION}"
    title = f"{method}  |  mode={mode}  |  backbone={backbone}"
    if tier:
        title += f"  |  tier={tier} ({', '.join(sorted(WW_PLUS_TIERS[tier]))})"
    lines = [
        "# ============================================================================",
        f"# {title}",
        "# Auto-generated by tools/gen_grid_configs.py - hierarchy experiment grid.",
        "# Usage:",
        f"#   python train_wsgg_methods.py --config configs/methods/{mode}/{stem}_{mode}_{backbone}.yaml",
        "# ============================================================================",
        "",
    ]
    items = list(COMMON)
    if method == "worldwise":
        items = items + WORLDWISE_EXTRA
        # Mode-specific priors file (predcls/sgdet label distributions differ;
        # a single shared path let one mode's priors overwrite the other's)
        items = [
            (k, v.format(mode=mode) if k == "predicate_priors_path" else v)
            for k, v in items
        ]
        if tier:
            overrides = WW_PLUS_TIERS[tier]
            items = [(k, overrides.get(k, v)) for k, v in items]
            # Tier keys not present in the base list (e.g. lambda_stability
            # is already listed; this catches any future additions)
            listed = {k for k, _ in items}
            items += [(k, v) for k, v in overrides.items() if k not in listed]
    items = items + [
        ("method_name", method),
        ("mode", mode),
        ("feature_model", backbone),
        ("task_name", "worldsgg"),
        ("experiment_name", exp),
    ]
    for k, v in items:
        lines.append(f"{k}: {fmt(v)}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", action="store_true",
                    help="also generate WorldWise⁺ tier configs at the hero backbone")
    args = ap.parse_args()

    n = 0
    removed = 0
    for mode in MODES:
        out_dir = os.path.join(REPO, "configs", "methods", mode)
        os.makedirs(out_dir, exist_ok=True)
        for method in METHODS:
            for backbone in backbones_for(method):
                fname = f"{method}_{mode}_{backbone}.yaml"
                with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
                    f.write(render(method, mode, backbone))
                n += 1
                print(f"  wrote configs/methods/{mode}/{fname}")
            # Drop stale configs for combos no longer in the design
            # (baselines at non-resnet50 backbones)
            for backbone in BACKBONES:
                if backbone in backbones_for(method):
                    continue
                stale = os.path.join(out_dir, f"{method}_{mode}_{backbone}.yaml")
                if os.path.exists(stale):
                    os.remove(stale)
                    removed += 1
                    print(f"  removed stale configs/methods/{mode}/{method}_{mode}_{backbone}.yaml")

        if args.tiers:
            for tier in WW_PLUS_TIERS:
                # Diagnostic tiers live at the hero backbone; v2 recomposition
                # candidates are generated at every backbone (ladder @ resnet50,
                # scaling for the winner).
                tier_backbones = BACKBONES if tier in V2_CANDIDATES else [HERO_BACKBONE]
                for backbone in tier_backbones:
                    fname = f"worldwise_{tier}_{mode}_{backbone}.yaml"
                    with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
                        f.write(render("worldwise", mode, backbone, tier=tier))
                    n += 1
                    print(f"  wrote configs/methods/{mode}/{fname}")
    print(f"\nGenerated {n} configs ({removed} stale removed).")


if __name__ == "__main__":
    main()
