"""
Run the WorldSGG hierarchy-experiment grid (40 cells, 1 seed each).

Sequentially trains every (method, backbone, mode) cell by invoking
train_wsgg_methods.py with the generated config. Continues on failure and
prints a pass/fail summary at the end.

Examples:
    # whole grid (computes predicate priors first, then 40 runs)
    python tools/run_grid.py --compute-priors

    # only the DINOv3-L column for predcls (the hero method-comparison cells)
    python tools/run_grid.py --modes predcls --backbones dinov3l

    # just WorldWise across all backbones (the backbone-scaling cells)
    python tools/run_grid.py --methods worldwise

    # see what would run without launching anything
    python tools/run_grid.py --dry-run

    # Stage A of the WorldWise⁺ plugin round (hero backbone, both tasks):
    python tools/run_grid.py --methods worldwise --backbones dinov3l --tiers plus1 plus2 plus3

    # Stage B research plugins, one at a time:
    python tools/run_grid.py --methods worldwise --backbones dinov3l --tiers proto --modes predcls
"""

import argparse
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

METHODS = ["w_sttran", "w_sttran_pp", "w_dsgdetr", "w_dsgdetr_pp", "worldwise"]
BACKBONES = ["resnet50", "dinov2b", "dinov2l", "dinov3l"]
MODES = ["predcls", "sgdet"]
TIERS = ["plus1", "plus2", "plus3", "noema", "conf", "proto", "xobj", "energy"]
METHODS_BACKBONE = "resnet50"  # baselines exist only here; worldwise everywhere


def _valid(method, backbone):
    """Baselines run only at resnet50; worldwise (incl. tiers) at any backbone."""
    return method.startswith("worldwise") or backbone == METHODS_BACKBONE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    ap.add_argument("--backbones", nargs="+", default=BACKBONES, choices=BACKBONES)
    ap.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    ap.add_argument("--tiers", nargs="+", default=[], choices=TIERS,
                    help="WorldWise⁺ tier configs to run (worldwise_<tier>_...)")
    ap.add_argument("--data_path", default="/data/rohith/ag")
    ap.add_argument("--compute-priors", action="store_true",
                    help="run tools/compute_predicate_priors.py before training")
    ap.add_argument("--priors-feature-model", default="dinov2b")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.tiers:
        # Tier configs exist for the hero backbone only; config stem is
        # worldwise_<tier>_<mode>_<backbone>.yaml with method_name=worldwise.
        # Appended to --methods so "--methods worldwise --tiers plus1" runs
        # the I-0 reference AND the tier cell.
        args.methods = list(args.methods) + [f"worldwise_{t}" for t in args.tiers]

    py = sys.executable

    if args.compute_priors and not args.dry_run:
        for mode in args.modes:
            print(f"\n=== computing predicate priors ({mode}) ===")
            subprocess.run([
                py, os.path.join("tools", "compute_predicate_priors.py"),
                "--data_path", args.data_path,
                "--feature_model", args.priors_feature_model,
                "--mode", mode,
            ], cwd=REPO, check=False)

    cells = [(mode, m, b)
             for mode in args.modes
             for m in args.methods
             for b in args.backbones
             if _valid(m, b)]

    print(f"\n{len(cells)} cells to run:")
    for mode, m, b in cells:
        print(f"  {mode:7s} | {m:13s} | {b}")
    if args.dry_run:
        return

    results = []
    for i, (mode, m, b) in enumerate(cells, 1):
        cfg = os.path.join("configs", "methods", mode, f"{m}_{mode}_{b}.yaml")
        if not os.path.exists(os.path.join(REPO, cfg)):
            print(f"[{i}/{len(cells)}] MISSING CONFIG {cfg} — skipping")
            results.append((mode, m, b, "missing"))
            continue
        print(f"\n{'=' * 70}\n[{i}/{len(cells)}] {mode} | {m} | {b}\n{'=' * 70}")
        t0 = time.time()
        rc = subprocess.run(
            [py, "train_wsgg_methods.py", "--config", cfg], cwd=REPO,
        ).returncode
        dt = time.time() - t0
        status = "ok" if rc == 0 else f"FAILED(rc={rc})"
        results.append((mode, m, b, status))
        print(f"[{i}/{len(cells)}] {status} in {dt / 60:.1f} min")

    print(f"\n{'=' * 70}\nGRID SUMMARY\n{'=' * 70}")
    for mode, m, b, status in results:
        print(f"  {status:14s} | {mode:7s} | {m:13s} | {b}")
    n_ok = sum(1 for *_, s in results if s == "ok")
    print(f"\n{n_ok}/{len(results)} succeeded.")


if __name__ == "__main__":
    main()
