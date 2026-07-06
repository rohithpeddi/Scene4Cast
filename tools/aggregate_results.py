"""
Aggregate WorldSGG results into the final campaign tables.

WorldWise's MAIN configuration is v2e (experiment stem `worldwise_v2e` — kept
so historical v2e cells are reused). Reads results/<experiment>_metrics.jsonl
and emits:

  Table A — 4 baselines + WorldWise @ resnet50            [the method ladder]
  Table B — WorldWise across all four backbones            [backbone scaling]
  Table C — component-wise ablations of WorldWise @ dinov3l  [--tiers]

For each cell we select one epoch (default: best With-Constraint R@20,
matching checkpoint selection) and read all metrics from it. Column maxima
are **bold**. Also writes a wide CSV (results/grid_summary_<mode>.csv).

    python tools/aggregate_results.py --mode predcls --tiers
    python tools/aggregate_results.py --mode sgdet  --tiers
"""

import argparse
import csv
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASELINES = ["w_sttran", "w_sttran_pp", "w_dsgdetr", "w_dsgdetr_pp"]
BACKBONES = ["resnet50", "dinov2b", "dinov2l", "dinov3l"]
METHODS_BACKBONE = "resnet50"   # the only backbone all methods share
HERO_BACKBONE = "dinov3l"       # ablations run here
MAIN_STEM = "worldwise_v2e"     # main WorldWise experiment stem (v2e recipe)
MAIN_LABEL = "worldwise (v2e)"

# Ablation rows: tier stem -> table label (order = table order).
# Must match ABLATION_TIERS in tools/gen_grid_configs.py.
ABLATIONS = [
    ("v2a",            "+ noisy VLM supervision (λ_vlm=0.2)"),
    ("v2f",            "τ=0.75 (mR-max operating point)"),
    ("abl_notau",      "− logit adjustment"),
    ("abl_nomask",     "− artificial masking (recon/sim off)"),
    ("abl_noema",      "− EMA recon target"),
    ("v2g",            "− pair geometry (I-1)"),
    ("abl_nospatial",  "− ObjectSpatialEncoder"),
    ("abl_noego",      "− ego-motion encoder"),
    ("abl_nomotion",   "− object motion encoder"),
    ("abl_notempedge", "− temporal edge attention"),
]

KS = [10, 20, 50]
METRIC_COLS = (
    [f"wc/R@{k}" for k in KS] + [f"wc/mR@{k}" for k in KS] + ["wc/hR@20"]
    + [f"nc/R@{k}" for k in KS] + [f"nc/mR@{k}" for k in KS] + ["nc/hR@20"]
)


def load_exp(results_dir, stem, select, sel_metric):
    """Load one experiment's best/last metric row; stem excludes _metrics.jsonl."""
    path = os.path.join(results_dir, f"{stem}_metrics.jsonl")
    if not os.path.exists(path):
        return None
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # Skip run-identity header rows (written once per run)
            if row.get("type") == "header" or "epoch" not in row:
                continue
            rows.append(row)
    if not rows:
        return None
    if select == "last":
        return rows[-1]
    return max(rows, key=lambda r: r.get(sel_metric, 0.0))


def render_table(title, row_labels, row_keys, cells, constraint):
    """cells: dict[row_key] -> metric row (or None). constraint: 'wc'|'nc'."""
    cols = (
        [f"{constraint}/R@{k}" for k in KS]
        + [f"{constraint}/mR@{k}" for k in KS]
        + [f"{constraint}/hR@20"]  # balanced headline: harmonic mean of R/mR
    )
    headers = ["R@10", "R@20", "R@50", "mR@10", "mR@20", "mR@50", "hR@20"]

    # column maxima for bolding
    maxima = {}
    for c in cols:
        vals = [cells[k][c] for k in row_keys if cells.get(k) and c in cells[k]]
        maxima[c] = max(vals) if vals else None

    def cell(rk, c):
        r = cells.get(rk)
        if not r or c not in r:
            return "—"
        v = r[c]
        s = f"{v:.2f}"
        if maxima[c] is not None and abs(v - maxima[c]) < 1e-9:
            s = f"**{s}**"
        return s

    lines = [
        f"\n### {title} — {'With Constraint' if constraint == 'wc' else 'No Constraint'}",
        "",
        "| Method | " + " | ".join(headers) + " |",
        "|" + "---|" * (len(headers) + 1),
    ]
    for rk, label in zip(row_keys, row_labels):
        lines.append(f"| {label} | " + " | ".join(cell(rk, c) for c in cols) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default=os.path.join(REPO, "results"))
    ap.add_argument("--mode", default="predcls", choices=["predcls", "sgdet"])
    ap.add_argument("--select", default="best", choices=["best", "last"])
    ap.add_argument("--sel-metric", default="wc/R@20")
    ap.add_argument("--version", default="v2",
                    help="experiment-name version suffix")
    ap.add_argument("--tiers", action="store_true",
                    help="also render Table C (component-wise ablations)")
    args = ap.parse_args()

    def load(stem_prefix, backbone):
        return load_exp(
            args.results_dir,
            f"{stem_prefix}_{args.mode}_{backbone}_{args.version}",
            args.select, args.sel_metric,
        )

    # ---- Table A: the method ladder at the common backbone ----
    a_keys = BASELINES + ["main"]
    a_labels = BASELINES + [MAIN_LABEL]
    a_cells = {m: load(m, METHODS_BACKBONE) for m in BASELINES}
    a_cells["main"] = load(MAIN_STEM, METHODS_BACKBONE)

    print(f"\n# WorldSGG — mode={args.mode}, select={args.select} ({args.sel_metric})")
    print(render_table(f"Table A · Method ladder @ {METHODS_BACKBONE}",
                       a_labels, a_keys, a_cells, "wc"))
    print(render_table(f"Table A · Method ladder @ {METHODS_BACKBONE}",
                       a_labels, a_keys, a_cells, "nc"))

    # ---- Table B: WorldWise backbone scaling ----
    b_cells = {b: load(MAIN_STEM, b) for b in BACKBONES}
    print(render_table(f"Table B · {MAIN_LABEL} backbone scaling",
                       BACKBONES, BACKBONES, b_cells, "wc"))
    print(render_table(f"Table B · {MAIN_LABEL} backbone scaling",
                       BACKBONES, BACKBONES, b_cells, "nc"))

    # ---- Table C: component-wise ablations at the hero backbone ----
    if args.tiers:
        c_keys = ["main"] + [t for t, _ in ABLATIONS]
        c_labels = [f"{MAIN_LABEL} — full"] + [label for _, label in ABLATIONS]
        c_cells = {"main": load(MAIN_STEM, HERO_BACKBONE)}
        for t, _label in ABLATIONS:
            c_cells[t] = load(f"worldwise_{t}", HERO_BACKBONE)
        print(render_table(f"Table C · Component ablations @ {HERO_BACKBONE}",
                           c_labels, c_keys, c_cells, "wc"))
        print(render_table(f"Table C · Component ablations @ {HERO_BACKBONE}",
                           c_labels, c_keys, c_cells, "nc"))

    # ---- CSV dump (every cell in the design) ----
    csv_path = os.path.join(args.results_dir, f"grid_summary_{args.mode}.csv")
    os.makedirs(args.results_dir, exist_ok=True)
    design = [(m, METHODS_BACKBONE) for m in BASELINES]
    design += [(MAIN_STEM, b) for b in BACKBONES]
    design += [(f"worldwise_{t}", HERO_BACKBONE) for t, _ in ABLATIONS]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "backbone", "mode", "epoch"] + METRIC_COLS)
        for stem, b in design:
            r = load(stem, b)
            if r is None:
                w.writerow([stem, b, args.mode, "—"] + ["—"] * len(METRIC_COLS))
            else:
                w.writerow([stem, b, args.mode, r.get("epoch", "?")]
                           + [r.get(c, "") for c in METRIC_COLS])
    print(f"\nCSV → {csv_path}")


if __name__ == "__main__":
    main()
