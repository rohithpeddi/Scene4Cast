"""
Generate paper-ready LaTeX result tables (booktabs) from results/*.jsonl.

Matches the target layout: a 4-way column split
    Recall (R@K)                 |  Mean Recall (mR@K)
    With Constraint | No Constr.  |  With Constraint | No Constr.
    R@10 R@20 R@50  | R@10 ...     |  mR@10 ...       | ...
with the per-column best value highlighted (green cell + bold). Values are
scaled x100. One row per experiment, read from its best-wc/R@20 epoch (the
checkpoint-selection metric), matching tools/aggregate_results.py.

Emits, per mode:
  <out>/<mode>_main.tex       — baselines + WorldWise-per-backbone (the main table)
  <out>/<mode>_ablation.tex   — full WorldWise + component ablations @ hero
plus <out>/preview.tex, a standalone compilable document \input-ing them.

Stdlib only. Usage:
    python tools/gen_paper_tables.py                       # both modes → results_tables/
    python tools/gen_paper_tables.py --mode predcls
    python tools/gen_paper_tables.py --out some/other/dir
"""

import argparse
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MAIN_STEM = "worldwise_v2e"          # the WorldWise v2e recipe
METHODS_BACKBONE = "resnet50"        # baselines live here
HERO = "dinov3l"                     # ablations run here

# Baselines (frozen controls) — (experiment stem, LaTeX display name)
BASELINES = [
    ("w_sttran",     r"W-STTran"),
    ("w_sttran_pp",  r"W-STTran\textsuperscript{++}"),
    ("w_dsgdetr",    r"W-DSGDetr"),
    ("w_dsgdetr_pp", r"W-DSGDetr\textsuperscript{++}"),
]

# WorldWise backbones shown as method rows in the main table
WW_BACKBONES = ["dinov2b", "dinov2l", "dinov3l"]
BACKBONE_DISPLAY = {
    "resnet50": "ResNet50", "dinov2b": "DINOv2-B",
    "dinov2l": "DINOv2-L", "dinov3l": "DINOv3-L",
}

# Ablation rows: (tier stem, LaTeX display name). Must match ABLATION_TIERS
# in tools/gen_grid_configs.py.
ABLATIONS = [
    ("v2a",            r"$+$ VLM supervision ($\lambda_{\text{vlm}}{=}0.2$)"),
    ("v2f",            r"$\tau{=}0.75$ (mR-max)"),
    ("abl_notau",      r"$-$ logit adjustment"),
    ("abl_nomask",     r"$-$ artificial masking"),
    ("abl_noema",      r"$-$ EMA recon.\ target"),
    ("v2g",            r"$-$ pair geometry"),
    ("abl_nospatial",  r"$-$ object spatial enc."),
    ("abl_noego",      r"$-$ ego-motion enc."),
    ("abl_nomotion",   r"$-$ object motion enc."),
    ("abl_notempedge", r"$-$ temporal edge attn."),
]

KS = [10, 20, 50]
# Column order matches the target figure: R (wc, nc) then mR (wc, nc)
COLS = (
    [f"wc/R@{k}" for k in KS] + [f"nc/R@{k}" for k in KS]
    + [f"wc/mR@{k}" for k in KS] + [f"nc/mR@{k}" for k in KS]
)


def load_row(results_dir, stem, mode, backbone, version, select, sel_metric):
    path = os.path.join(results_dir, f"{stem}_{mode}_{backbone}_{version}_metrics.jsonl")
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("type") == "header" or "epoch" not in r:
                continue
            rows.append(r)
    if not rows:
        return None
    return rows[-1] if select == "last" else max(rows, key=lambda r: r.get(sel_metric, 0.0))


def _val(row, col, scale):
    if row is None or col not in row or row[col] is None:
        return None
    return row[col] * scale


def _render(caption, label, method_rows, groups, scale, decimals):
    """method_rows: list of (display_name, row_dict_or_None).
       groups: list of ints — sizes of visually-separated row blocks."""
    # numeric matrix + per-column maxima (higher is better for R and mR)
    values = [[_val(row, c, scale) for c in COLS] for _, row in method_rows]
    col_max = []
    for j in range(len(COLS)):
        present = [values[i][j] for i in range(len(values)) if values[i][j] is not None]
        col_max.append(max(present) if present else None)

    def cell(v, j):
        if v is None:
            return "--"
        s = f"{v:.{decimals}f}"
        if col_max[j] is not None and abs(v - col_max[j]) < 1e-9:
            return r"\cellcolor{bestcol}\textbf{" + s + "}"
        return s

    L = []
    L.append(r"\begin{table}[t]")
    L.append(r"\centering")
    L.append(r"\caption{" + caption + r"}")
    L.append(r"\label{" + label + r"}")
    L.append(r"\setlength{\tabcolsep}{4pt}")
    L.append(r"\resizebox{\linewidth}{!}{%")
    L.append(r"\begin{tabular}{l cccccc cccccc}")
    L.append(r"\toprule")
    L.append(r"& \multicolumn{6}{c}{\textbf{Recall (R@K)}} "
             r"& \multicolumn{6}{c}{\textbf{Mean Recall (mR@K)}} \\")
    L.append(r"\cmidrule(lr){2-7}\cmidrule(lr){8-13}")
    L.append(r"& \multicolumn{3}{c}{\textit{With Constraint}} "
             r"& \multicolumn{3}{c}{\textit{No Constraint}} "
             r"& \multicolumn{3}{c}{\textit{With Constraint}} "
             r"& \multicolumn{3}{c}{\textit{No Constraint}} \\")
    L.append(r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}")
    hdr = " & ".join([r"\textbf{Method}"]
                     + [f"R@{k}" for k in KS] + [f"R@{k}" for k in KS]
                     + [f"mR@{k}" for k in KS] + [f"mR@{k}" for k in KS])
    L.append(hdr + r" \\")
    L.append(r"\midrule")

    i = 0
    for gi, gsize in enumerate(groups):
        if gi > 0:
            L.append(r"\addlinespace")
        for _ in range(gsize):
            name, _row = method_rows[i]
            cells = " & ".join(cell(values[i][j], j) for j in range(len(COLS)))
            L.append(f"{name} & {cells} " + r"\\")
            i += 1
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}}")
    L.append(r"\end{table}")
    return "\n".join(L) + "\n"


def main_table(results_dir, mode, version, select, sel_metric, scale, decimals):
    rows = []
    for stem, name in BASELINES:
        rows.append((name, load_row(results_dir, stem, mode, METHODS_BACKBONE,
                                    version, select, sel_metric)))
    for bb in WW_BACKBONES:
        name = r"WorldWise\textsubscript{" + BACKBONE_DISPLAY[bb] + "}"
        rows.append((name, load_row(results_dir, MAIN_STEM, mode, bb,
                                    version, select, sel_metric)))
    groups = [2, 2, len(WW_BACKBONES)]   # STTran pair | DSGDetr pair | WorldWise
    disp = "PredCls" if mode == "predcls" else "SGDet"
    return _render(f"{disp} results on ActionGenome4D.",
                   f"tab:{mode}_main", rows, groups, scale, decimals)


def ablation_table(results_dir, mode, version, select, sel_metric, scale, decimals):
    rows = [(r"WorldWise (full)",
             load_row(results_dir, MAIN_STEM, mode, HERO, version, select, sel_metric))]
    for stem, name in ABLATIONS:
        rows.append((name, load_row(results_dir, f"worldwise_{stem}", mode, HERO,
                                    version, select, sel_metric)))
    groups = [1, len(ABLATIONS)]         # full config | ablations
    disp = "PredCls" if mode == "predcls" else "SGDet"
    return _render(f"Component-wise ablation of WorldWise ({disp}, DINOv3-L).",
                   f"tab:{mode}_ablation", rows, groups, scale, decimals)


PREAMBLE_NOTE = (
    "% Requires in your preamble:\n"
    "%   \\usepackage{booktabs}\n"
    "%   \\usepackage{amsmath}     % for \\text in ablation labels\n"
    "%   \\usepackage[table]{xcolor}\n"
    "%   \\usepackage{graphicx}   % for \\resizebox\n"
    "%   \\definecolor{bestcol}{RGB}{213,232,212}  % light green highlight\n"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default=os.path.join(REPO, "results"))
    ap.add_argument("--mode", default="all", choices=["predcls", "sgdet", "all"])
    ap.add_argument("--out", default=os.path.join(REPO, "results_tables"))
    ap.add_argument("--version", default="v2")
    ap.add_argument("--select", default="best", choices=["best", "last"])
    ap.add_argument("--sel-metric", default="wc/R@20")
    ap.add_argument("--scale", type=float, default=100.0)
    ap.add_argument("--decimals", type=int, default=2)
    args = ap.parse_args()

    modes = ["predcls", "sgdet"] if args.mode == "all" else [args.mode]
    os.makedirs(args.out, exist_ok=True)
    written = []
    for mode in modes:
        for kind, fn in (("main", main_table), ("ablation", ablation_table)):
            tex = fn(args.results_dir, mode, args.version, args.select,
                     args.sel_metric, args.scale, args.decimals)
            path = os.path.join(args.out, f"{mode}_{kind}.tex")
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(PREAMBLE_NOTE + tex)
            written.append(path)
            print(f"  wrote {os.path.relpath(path, REPO)}")

    # standalone preview document
    preview = os.path.join(args.out, "preview.tex")
    with open(preview, "w", encoding="utf-8", newline="\n") as f:
        f.write(r"""\documentclass[10pt]{article}
\usepackage[margin=0.5in,landscape]{geometry}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage[table]{xcolor}
\usepackage{graphicx}
\definecolor{bestcol}{RGB}{213,232,212}
\begin{document}
""")
        for mode in modes:
            for kind in ("main", "ablation"):
                f.write("\\input{" + f"{mode}_{kind}.tex" + "}\n\n")
        f.write(r"\end{document}" + "\n")
    print(f"  wrote {os.path.relpath(preview, REPO)}  "
          f"(compile: cd {os.path.relpath(args.out, REPO)} && pdflatex preview.tex)")


if __name__ == "__main__":
    main()
