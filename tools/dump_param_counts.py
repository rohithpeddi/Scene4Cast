"""
Dump per-method parameter counts for the WSGG ladder (freeze artifact).

Instantiates every method with the shared grid architecture hyperparameters
and writes a markdown table. The ladder must be monotone:
W-STTran < W-STTran++ < W-DSGDetr < W-DSGDetr++ < WorldWise.

Run on the training server (needs torch, not data):

    python tools/dump_param_counts.py [--out docs/experiments/param_counts.md]
"""

import argparse
import os
import sys
from types import SimpleNamespace

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def shared_config(**overrides):
    """Architecture keys matching tools/gen_grid_configs.py COMMON/WORLDWISE_EXTRA."""
    cfg = dict(
        data_path="", clip_embeddings_path="",
        d_model=256, d_struct=256, d_visual=256, d_detector_roi=1024,
        d_camera=128, d_motion=64,
        n_heads=4, d_feedforward=512, dropout=0.1,
        d_rel=256, d_text=128, d_union_roi=1024,
        n_rel_layers=2, n_rel_heads=4,
        n_temporal_edge_layers=1, n_temporal_obj_layers=2,
        n_gnn_layers=3, n_cross_attn_layers=2, n_self_attn_layers=3,
        max_objects=64,
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def count(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="optional markdown output path")
    args = ap.parse_args()

    from lib.supervised.baselines.w_sttran.w_sttran import WSTTran
    from lib.supervised.baselines.w_sttran.w_sttran_pp import WSTTranPP
    from lib.supervised.baselines.w_dsgdetr.w_dsgdetr import WDSGDetr
    from lib.supervised.baselines.w_dsgdetr.w_dsgdetr_pp import WDSGDetrPP
    from lib.supervised.worldwise.worldwise import WorldWise

    rows = []
    ladder = [
        ("w_sttran", WSTTran, {}),
        ("w_sttran_pp", WSTTranPP, {}),
        ("w_dsgdetr", WDSGDetr, {}),
        ("w_dsgdetr_pp", WDSGDetrPP, {}),
        ("worldwise (I-0)", WorldWise, {}),
        ("worldwise +I-1", WorldWise, {"use_pair_geometry": True}),
        ("worldwise +I-1..3", WorldWise, {
            "use_pair_geometry": True,
            "use_soft_text_embedding": True,
            "use_geometric_attn_bias": True,
        }),
        ("worldwise all plugins", WorldWise, {
            "use_pair_geometry": True,
            "use_soft_text_embedding": True,
            "use_geometric_attn_bias": True,
            "use_predicate_prototypes": True,
            "use_cross_object_retrieval": True,
            "use_energy_refinement": True,
        }),
    ]
    for name, cls, flags in ladder:
        model = cls(shared_config(**flags))
        total, trainable = count(model)
        rows.append((name, total, trainable))
        print(f"{name:28s} total={total:>12,}  trainable={trainable:>12,}")

    # Monotonicity check over the 5-method ladder
    core = [r[1] for r in rows[:5]]
    assert all(a < b for a, b in zip(core, core[1:])), (
        f"Ladder param counts not monotone: {core}"
    )
    print("\nLadder monotonicity: OK")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("# WSGG ladder parameter counts (freeze artifact)\n\n")
            f.write("| Method | Total params | Trainable |\n|---|---:|---:|\n")
            for name, total, trainable in rows:
                f.write(f"| {name} | {total:,} | {trainable:,} |\n")
        print(f"Written → {args.out}")


if __name__ == "__main__":
    main()
