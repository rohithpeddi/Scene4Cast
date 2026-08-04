"""Generate LaTeX tables for the per-visibility-bucket rebuttal analysis.

Reads results/bucketed_breakdown.json and emits three .tex files matching the
WSGG paper's table conventions (booktabs, tableheader/rowalt/bestcell colors,
With/No-Constraint super-panels, WorldWise\\textsubscript{...} naming).

Usage:
    python tools/gen_rebuttal_tables.py <breakdown.json> <out_dir>
"""
import json
import os
import sys

JSON = sys.argv[1] if len(sys.argv) > 1 else "results/bucketed_breakdown.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "."

ORDER = ["w_sttran", "w_sttran_pp", "w_dsgdetr", "w_dsgdetr_pp", "worldwise"]
NAME = {
    "w_sttran": "W-STTran",
    "w_sttran_pp": r"W-STTran\textsuperscript{++}",
    "w_dsgdetr": "W-DsgDetr",
    "w_dsgdetr_pp": r"W-DsgDetr\textsuperscript{++}",
    "worldwise": r"WorldWise\textsubscript{DINOv3-L}",
}
ALT = {"w_sttran_pp", "w_dsgdetr_pp", "worldwise"}  # rows that get rowalt shading

data = json.load(open(JSON))
BY = {(r["meta"]["mode"], r["meta"]["method"]): r for r in data}


def gk(d, k):
    """key lookup tolerant to int/str (JSON coerces int keys to str)."""
    if d is None:
        return None
    return d.get(k, d.get(str(k)))


def val(mode, method, constraint, bucket, metric, k, nontrivial=False):
    r = BY.get((mode, method))
    if r is None:
        return None
    c = r["constraints"][constraint]
    if nontrivial:
        node = c.get("ou_nontrivial", {}).get("drop_trivial", {})
    else:
        node = c["buckets"].get(bucket, {})
    return gk(node.get(metric, {}), k)


def cell(x, best=False):
    if x is None:
        return "--"
    s = f"{100 * x:.1f}"
    return rf"\cellcolor{{bestcell}}\textbf{{{s}}}" if best else s


def col_best_idx(vals):
    """index of max (ignoring None); higher=better for all recall metrics."""
    bi, bv = None, None
    for i, v in enumerate(vals):
        if v is None:
            continue
        if bv is None or v > bv:
            bv, bi = v, i
    return bi


HEADER = r"""\definecolor{tableheader}{HTML}{E8EAF6}
\definecolor{rowalt}{HTML}{F5F5F5}
\definecolor{bestcell}{HTML}{C8E6C9}
"""


def recall_table(mode, label, caption, star=True):
    """Per-bucket recall table: R@50 and mR@50 super-panels, each split into
    With/No Constraint, each with OO / OU / OU-nt columns."""
    env = "table*" if star else "table"
    # column data: list of (constraint, bucket, metric, nontrivial)
    specs = [
        ("wc", "OO", "R", False), ("wc", "OU", "R", False), ("wc", None, "R", True),
        ("nc", "OO", "R", False), ("nc", "OU", "R", False), ("nc", None, "R", True),
        ("wc", "OO", "mR", False), ("wc", "OU", "mR", False), ("wc", None, "mR", True),
        ("nc", "OO", "mR", False), ("nc", "OU", "mR", False), ("nc", None, "mR", True),
    ]
    # precompute best index per column
    best = []
    for (con, buc, met, nt) in specs:
        vals = [val(mode, m, con, buc, met, 50, nt) for m in ORDER]
        best.append(col_best_idx(vals))

    lines = []
    lines.append(HEADER)
    lines.append(rf"\begin{{{env}}}[!t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.0}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\footnotesize")
    lines.append(r"\begin{tabular}{l ccc ccc c ccc ccc}")
    lines.append(r"\toprule")
    lines.append(r"\rowcolor{tableheader}")
    lines.append(r"& \multicolumn{6}{c}{\textbf{Recall (R@50)}} & & "
                 r"\multicolumn{6}{c}{\textbf{Mean Recall (mR@50)}} \\")
    lines.append(r"\cmidrule(lr){2-7} \cmidrule(lr){9-14}")
    lines.append(r"\rowcolor{tableheader}")
    lines.append(r"& \multicolumn{3}{c}{\textit{With Constraint}} & "
                 r"\multicolumn{3}{c}{\textit{No Constraint}} & & "
                 r"\multicolumn{3}{c}{\textit{With Constraint}} & "
                 r"\multicolumn{3}{c}{\textit{No Constraint}} \\")
    lines.append(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} "
                 r"\cmidrule(lr){9-11} \cmidrule(lr){12-14}")
    lines.append(r"\rowcolor{tableheader}")
    bh = r"\scriptsize OO & \scriptsize OU & \scriptsize OU\textsubscript{nt}"
    lines.append(rf"\textbf{{Method}} & {bh} & {bh} & & {bh} & {bh} \\")
    lines.append(r"\midrule")

    for ri, m in enumerate(ORDER):
        cells = []
        for ci, (con, buc, met, nt) in enumerate(specs):
            v = val(mode, m, con, buc, met, 50, nt)
            cells.append(cell(v, best=(best[ci] == ri)))
        row = (f"{NAME[m]} & " + " & ".join(cells[:6]) + " & & "
               + " & ".join(cells[6:]) + r" \\")
        if m in ALT:
            lines.append(r"\rowcolor{rowalt}")
        lines.append(row)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(rf"\end{{{env}}}")
    return "\n".join(lines) + "\n"


def unobserved_lowk_table(mode, label, caption):
    """Focused table: the unobserved bucket across K (R@10/20/50), both
    constraints -- shows the low-K discrimination that R@50 saturation hides."""
    ks = [10, 20, 50]
    specs = [("wc", k) for k in ks] + [("nc", k) for k in ks]
    best = []
    for (con, k) in specs:
        vals = [val(mode, m, con, "OU", "R", k) for m in ORDER]
        best.append(col_best_idx(vals))
    lines = [HEADER, r"\begin{table}[!t]", r"\centering",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{1.0}",
             r"\resizebox{\columnwidth}{!}{%", r"\footnotesize",
             r"\begin{tabular}{l ccc c ccc}", r"\toprule", r"\rowcolor{tableheader}",
             r"& \multicolumn{3}{c}{\textit{With Constraint}} & & "
             r"\multicolumn{3}{c}{\textit{No Constraint}} \\",
             r"\cmidrule(lr){2-4} \cmidrule(lr){6-8}", r"\rowcolor{tableheader}",
             r"\textbf{Method} & \scriptsize R@10 & \scriptsize R@20 & \scriptsize R@50 "
             r"& & \scriptsize R@10 & \scriptsize R@20 & \scriptsize R@50 \\",
             r"\midrule"]
    for ri, m in enumerate(ORDER):
        c = [cell(val(mode, m, con, "OU", "R", k), best[ci] == ri)
             for ci, (con, k) in enumerate(specs)]
        row = f"{NAME[m]} & " + " & ".join(c[:3]) + " & & " + " & ".join(c[3:]) + r" \\"
        if m in ALT:
            lines.append(r"\rowcolor{rowalt}")
        lines.append(row)
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def composition_table():
    """GT composition by bucket + OU predicate mix (PredCls, shared GT)."""
    s = BY[("predcls", "worldwise")]["constraints"]["nc"]
    sizes = s["bucket_sizes"]
    tot = sum(sizes.values())
    comp = s["ou_composition"]
    ntg = s["ou_nontrivial"]["drop_trivial"]["n_gt"]
    triv = sum(v[1] for k, v in comp.items()
               if k in ("not_looking_at", "not_contacting"))
    # top real (non-trivial) predicates
    real = sorted(((k, v) for k, v in comp.items()
                   if k not in ("not_looking_at", "not_contacting") and v[0] > 0),
                  key=lambda kv: -kv[1][0])[:6]
    disp = {"in_front_of": "in front of", "on_the_side_of": "on the side of",
            "not_looking_at": "not looking at", "not_contacting": "not contacting"}

    lines = [HEADER, r"\begin{table}[!t]", r"\centering",
             r"\caption{\textbf{Composition of the evaluation set by object "
             r"visibility (\AGFourD{} test, PredCls).} Every relation is a "
             r"person--object pair anchored to the always-observed person, so "
             r"only two buckets are non-empty. The unobserved-object (OU) bucket "
             r"is \emph{not} dominated by the trivial \texttt{not\_looking\_at}/"
             r"\texttt{not\_contacting} negatives: the majority of its positives "
             r"are genuine relations.}",
             r"\label{tab:bucket_composition}",
             r"\setlength{\tabcolsep}{5pt}", r"\renewcommand{\arraystretch}{1.05}",
             r"\footnotesize", r"\begin{tabular}{lrr}", r"\toprule",
             r"\rowcolor{tableheader}",
             r"\textbf{Visibility bucket} & \textbf{\# GT triplets} & \textbf{Share} \\",
             r"\midrule",
             rf"Observed--observed (OO) & {sizes['OO']:,} & {100*sizes['OO']/tot:.1f}\% \\",
             r"\rowcolor{rowalt}",
             rf"Observed--\textbf{{unobserved}} (OU) & {sizes['OU']:,} & "
             rf"{100*sizes['OU']/tot:.1f}\% \\",
             r"\midrule",
             r"\multicolumn{3}{l}{\textit{Within the OU (unobserved-object) bucket:}} \\",
             rf"\quad trivial (\texttt{{not\_looking\_at}}$+$\texttt{{not\_contacting}}) "
             rf"& & {triv:.1f}\% \\",
             r"\rowcolor{rowalt}",
             rf"\quad \textbf{{non-trivial (real) positives}} & {ntg:,} & "
             rf"{100-triv:.1f}\% \\",
             r"\midrule",
             r"\multicolumn{3}{l}{\textit{Top non-trivial OU predicates "
             r"(\% of OU positives):}} \\"]
    for k, (c, pct) in real:
        lines.append(rf"\quad {disp.get(k, k.replace('_',' '))} & {c:,} & {pct:.1f}\% \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def write(name, content):
    path = os.path.join(OUT, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("wrote", path)


PRED_CAP = (r"\textbf{Recall by object-visibility bucket (PredCls, \AGFourD{} "
            r"test).} OO$=$observed pair; OU$=$unobserved object; "
            r"OU\textsubscript{nt}$=$OU excluding the trivial "
            r"\texttt{not\_looking\_at}/\texttt{not\_contacting} negatives. "
            r"Best per column highlighted. \emph{No Constraint saturates} "
            r"($\ge$90) and is uninformative; the discriminative signal is in "
            r"the With-Constraint and OU\textsubscript{nt} columns, where "
            r"WorldWise leads by a large margin on unobserved relations.")
SG_CAP = (r"\textbf{Recall by object-visibility bucket (SGDet, \AGFourD{} "
          r"test).} Columns as in Table~\ref{tab:predcls_bucket_recall}. "
          r"SGDet denominators are per-method (objects are detector-predicted; "
          r"WorldWise uses a DINOv3-L detector vs.\ ResNet-50 baselines), as in "
          r"any SGDet R@K. WorldWise leads on both recall and mean recall for "
          r"non-trivial unobserved positives.")
LOWK_CAP = (r"\textbf{Recall on unobserved-object relations (OU) across $K$ "
            r"(PredCls).} No-Constraint R@50 saturates for every method, masking "
            r"differences; at the stricter R@10/R@20 WorldWise ranks unobserved "
            r"relations far higher than baselines.")

write("bucket_composition.tex", composition_table())
write("predcls_bucket_recall.tex",
      recall_table("predcls", "tab:predcls_bucket_recall", PRED_CAP, star=True))
write("sgdet_bucket_recall.tex",
      recall_table("sgdet", "tab:sgdet_bucket_recall", SG_CAP, star=True))
write("predcls_unobserved_lowk.tex",
      unobserved_lowk_table("predcls", "tab:predcls_unobserved_lowk", LOWK_CAP))
print("done")
