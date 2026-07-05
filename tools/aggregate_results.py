"""
Aggregate WorldSGG grid results into the two hierarchy tables.

Reads results/<experiment>_metrics.jsonl (one row per epoch, written by
train_wsgg_base.py) for every grid cell and emits:

  Table A — method comparison @ the common backbone (resnet50) [rows = ladder]
  Table B — backbone scaling for WorldWise                     [rows = backbones]
  Table C — WorldWise⁺ plugin ladder @ the hero backbone       [--tiers]

Grid structure (July 2026): baselines exist only at resnet50; WorldWise at
all four backbones; tiers at the hero backbone (dinov3l).

For each cell we select one epoch (default: the epoch with the best
With-Constraint R@20, matching the checkpoint-selection metric) and read all
metrics from it. The column maximum in each table is **bold**.

Also writes a wide CSV (results/grid_summary_<mode>.csv) with every cell.

    python tools/aggregate_results.py --mode predcls --hero-backbone dinov3l --tiers
"""

import argparse
import csv
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

METHODS = ["w_sttran", "w_sttran_pp", "w_dsgdetr", "w_dsgdetr_pp", "worldwise"]
BACKBONES = ["resnet50", "dinov2b", "dinov2l", "dinov3l"]
METHODS_BACKBONE = "resnet50"  # the only backbone all 5 methods share
# Stage-A cumulative ladder + A/B + tuning follow-ups + Stage-B research
# plugins (must match WW_PLUS_TIERS in tools/gen_grid_configs.py)
TIERS = ["plus1", "plus2", "plus3", "plus3pe", "noema",
         "tau025", "tau05", "tau075",
         "notau", "lowmask", "nomask", "novlm", "nomotion", "noego",
         "v2a", "v2b", "v2c", "v2d", "v2e", "v2f",
         "conf", "proto", "xobj", "energy"]


def valid_combo(method, backbone):
    return method == "worldwise" or backbone == METHODS_BACKBONE


KS = [10, 20, 50]
METRIC_COLS = (
    [f"wc/R@{k}" for k in KS] + [f"wc/mR@{k}" for k in KS] + ["wc/hR@20"]
    + [f"nc/R@{k}" for k in KS] + [f"nc/mR@{k}" for k in KS] + ["nc/hR@20"]
)


def load_exp(results_dir, stem, select, sel_metric):
    """Load one experiment's metric rows; stem = '<name>_<mode>_<backbone>_<ver>'."""
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


def load_cell(results_dir, method, mode, backbone, select, sel_metric, version):
    return load_exp(
        results_dir, f"{method}_{mode}_{backbone}_{version}", select, sel_metric,
    )


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
        "| Method/Backbone | " + " | ".join(headers) + " |",
        "|" + "---|" * (len(headers) + 1),
    ]
    for rk, label in zip(row_keys, row_labels):
        lines.append(f"| {label} | " + " | ".join(cell(rk, c) for c in cols) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default=os.path.join(REPO, "results"))
    ap.add_argument("--mode", default="predcls", choices=["predcls", "sgdet"])
    ap.add_argument("--hero-backbone", default="dinov3l", choices=BACKBONES)
    ap.add_argument("--select", default="best", choices=["best", "last"])
    ap.add_argument("--sel-metric", default="wc/R@20")
    ap.add_argument("--version", default="v2",
                    help="experiment-name version suffix (v1 = pre-fix legacy)")
    ap.add_argument("--tiers", action="store_true",
                    help="also render Table C (WorldWise⁺ plugin ladder)")
    ap.add_argument("--v2", default=None, choices=["v2a", "v2b", "v2c", "v2d", "v2e", "v2f"],
                    help="append this recomposition candidate to Table A as "
                         "worldwise-v2 (reads worldwise_<cand>_<mode>_resnet50) "
                         "and use it for Table B across backbones")
    args = ap.parse_args()

    # Load every cell in the design
    grid = {}
    for m in METHODS:
        for b in BACKBONES:
            grid[(m, b)] = load_cell(
                args.results_dir, m, args.mode, b, args.select, args.sel_metric,
                args.version,
            ) if valid_combo(m, b) else None

    # ---- CSV dump ----
    csv_path = os.path.join(args.results_dir, f"grid_summary_{args.mode}.csv")
    os.makedirs(args.results_dir, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "backbone", "mode", "epoch"] + METRIC_COLS)
        for m in METHODS:
            for b in BACKBONES:
                if not valid_combo(m, b):
                    continue  # not part of the design (baseline @ non-resnet50)
                r = grid[(m, b)]
                if r is None:
                    w.writerow([m, b, args.mode, "—"] + ["—"] * len(METRIC_COLS))
                else:
                    w.writerow([m, b, args.mode, r.get("epoch", "?")]
                               + [r.get(c, "") for c in METRIC_COLS])

    # ---- Table A: method comparison at the common backbone (resnet50) ----
    a_keys, a_labels = list(METHODS), list(METHODS)
    a_cells = {m: grid[(m, METHODS_BACKBONE)] for m in METHODS}
    if args.v2:
        # The final ladder row: recomposed WorldWise at the ladder backbone
        a_keys.append("v2")
        a_labels.append(f"worldwise-v2 ({args.v2})")
        a_cells["v2"] = load_exp(
            args.results_dir,
            f"worldwise_{args.v2}_{args.mode}_{METHODS_BACKBONE}_{args.version}",
            args.select, args.sel_metric,
        )
    print(f"\n# WorldSGG hierarchy — mode={args.mode}, select={args.select} ({args.sel_metric})")
    print(render_table(f"Table A · Methods @ {METHODS_BACKBONE}", a_labels, a_keys, a_cells, "wc"))
    print(render_table(f"Table A · Methods @ {METHODS_BACKBONE}", a_labels, a_keys, a_cells, "nc"))

    # ---- Table B: WorldWise backbone scaling (v2 candidate when selected) ----
    if args.v2:
        b_cells = {
            b: load_exp(
                args.results_dir,
                f"worldwise_{args.v2}_{args.mode}_{b}_{args.version}",
                args.select, args.sel_metric,
            )
            for b in BACKBONES
        }
        b_title = f"Table B · WorldWise-v2 ({args.v2}) backbone scaling"
    else:
        b_cells = {b: grid[("worldwise", b)] for b in BACKBONES}
        b_title = "Table B · WorldWise backbone scaling"
    print(render_table(b_title, BACKBONES, BACKBONES, b_cells, "wc"))
    print(render_table(b_title, BACKBONES, BACKBONES, b_cells, "nc"))

    # ---- Table C: WorldWise⁺ plugin ladder at the hero backbone ----
    if args.tiers:
        hero = args.hero_backbone
        c_keys = ["base"] + TIERS
        c_labels = [f"worldwise (I-0)"] + [f"worldwise+{t}" for t in TIERS]
        c_cells = {"base": grid[("worldwise", hero)]}
        for t in TIERS:
            c_cells[t] = load_exp(
                args.results_dir,
                f"worldwise_{t}_{args.mode}_{hero}_{args.version}",
                args.select, args.sel_metric,
            )
        print(render_table(f"Table C · WorldWise⁺ plugin ladder @ {hero}",
                           c_labels, c_keys, c_cells, "wc"))
        print(render_table(f"Table C · WorldWise⁺ plugin ladder @ {hero}",
                           c_labels, c_keys, c_cells, "nc"))

    print(f"\nCSV → {csv_path}")


if __name__ == "__main__":
    main()
