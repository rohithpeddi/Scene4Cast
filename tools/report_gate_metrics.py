"""
Per-head report for the WorldWise component-ablation table.

The keep/drop gate criteria are per-head (spatial R/mR for I-1/I-3,
contacting mR for I-2/I-6, masked-pair recall for I-7) — the aggregate
tables only show overall R/mR, which cannot decide a gate. This tool reads
the per-predicate recall vectors (`wc/per_predicate_R@20`) and the
occlusion-stratified metrics (`vispair/*`, `occpair/*`) that every run logs
per epoch in results/*_metrics.jsonl, and renders:

  1. Methods @ resnet50   — per-head mR@20 + occlusion split
  2. Tier ladder @ hero   — per-head mR@20 with Δ vs each tier's parent,
                            annotated with the tier's gate-target head
  3. Per-predicate movers — biggest per-class deltas of a tier vs its parent

Per-head values are means over that head's per-predicate recalls (i.e. the
head's mean-recall). True per-head *weighted* recall is not derivable from
the logged vectors (no per-class GT counts) — mR is the gate metric anyway.

Stdlib only. Usage:
    python tools/report_gate_metrics.py --mode predcls
    python tools/report_gate_metrics.py --mode sgdet --tier-vs plus1=base
"""

import argparse
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASELINE_METHODS = ["w_sttran", "w_sttran_pp", "w_dsgdetr", "w_dsgdetr_pp"]
METHODS_BACKBONE = "resnet50"
HERO = "dinov3l"

# Head sizes, in the order the predicate vector is logged
# (attention 3, spatial 6, contacting 17 — see wsgg_base._init_evaluators)
HEADS = [("attention", 3), ("spatial", 6), ("contacting", 17)]

# Main WorldWise experiment stem (the v2e recipe) — ablations diff against it
MAIN_STEM = "worldwise_v2e"

# ablation tier -> (parent, what it removes/changes)
TIER_INFO = {
    "v2a":            ("base", "+ noisy VLM supervision back (lambda_vlm=0.2)"),
    "v2f":            ("base", "tau=0.75 (mR-max operating point)"),
    "abl_notau":      ("base", "- logit adjustment"),
    "abl_nomask":     ("base", "- artificial masking (recon/sim off)"),
    "abl_noema":      ("base", "- EMA recon target"),
    "v2g":            ("base", "- pair geometry (I-1)"),
    "abl_nospatial":  ("base", "- ObjectSpatialEncoder"),
    "abl_noego":      ("base", "- ego-motion encoder"),
    "abl_nomotion":   ("base", "- object motion encoder"),
    "abl_notempedge": ("base", "- temporal edge attention"),
}


def load_best_row(results_dir, stem, select, sel_metric):
    path = os.path.join(results_dir, f"{stem}_metrics.jsonl")
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("type") == "header" or "epoch" not in row:
                continue
            rows.append(row)
    if not rows:
        return None
    return rows[-1] if select == "last" else max(rows, key=lambda r: r.get(sel_metric, 0.0))


def head_means(row):
    """Split the per-predicate vector into per-head mean recalls."""
    vec = row.get("wc/per_predicate_R@20")
    if not vec:
        return None
    values = list(vec.values())  # insertion order = att ++ spa ++ con
    out, i = {}, 0
    for name, size in HEADS:
        chunk = values[i:i + size]
        out[name] = sum(chunk) / max(len(chunk), 1)
        i += size
    return out


def fmt(v, delta=False):
    if v is None:
        return "—"
    s = f"{v:+.3f}" if delta else f"{v:.3f}"
    return s


def occl(row, key):
    return row.get(key) if row else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default=os.path.join(REPO, "results"))
    ap.add_argument("--mode", default="predcls", choices=["predcls", "sgdet"])
    ap.add_argument("--hero-backbone", default=HERO)
    ap.add_argument("--version", default="v2")
    ap.add_argument("--select", default="best", choices=["best", "last"])
    ap.add_argument("--sel-metric", default="wc/R@20")
    args = ap.parse_args()

    def load(stem_prefix, backbone):
        stem = f"{stem_prefix}_{args.mode}_{backbone}_{args.version}"
        return load_best_row(args.results_dir, stem, args.select, args.sel_metric)

    print(f"\n# WorldWise+ gate report — mode={args.mode}, "
          f"select={args.select} ({args.sel_metric}), K=20, with-constraint\n")

    # ---------- 1. Methods @ common backbone ----------
    print(f"## Per-head mean recall — methods @ {METHODS_BACKBONE}\n")
    print("| Method | R@20 | mR@20 | att mR | spa mR | con mR | vis-pair R | occ-pair R |")
    print("|---|---|---|---|---|---|---|---|")
    for m in BASELINE_METHODS + [MAIN_STEM]:
        row = load(m, METHODS_BACKBONE)
        if row is None:
            print(f"| {m} | — | — | — | — | — | — | — |")
            continue
        hm = head_means(row) or {}
        print(f"| {m} | {fmt(row.get('wc/R@20'))} | {fmt(row.get('wc/mR@20'))} | "
              f"{fmt(hm.get('attention'))} | {fmt(hm.get('spatial'))} | "
              f"{fmt(hm.get('contacting'))} | "
              f"{fmt(occl(row, 'vispair/R@20'))} | {fmt(occl(row, 'occpair/R@20'))} |")

    # ---------- 2. Tier ladder @ hero ----------
    hero = args.hero_backbone
    rows = {"base": load(MAIN_STEM, hero)}
    for t in TIER_INFO:
        rows[t] = load(f"worldwise_{t}", hero)

    print(f"\n## Tier gate table @ {hero} (delta vs parent in parentheses)\n")
    print("| Tier | R@20 | mR@20 | att mR | spa mR | con mR | occ-pair R | gate target |")
    print("|---|---|---|---|---|---|---|---|")

    def tier_line(label, t):
        row = rows.get(t)
        parent_key, target = ("—", "I-0 reference") if t == "base" else TIER_INFO[t]
        parent = rows.get(parent_key) if t != "base" else None
        if row is None:
            print(f"| {label} | — | — | — | — | — | — | {target} |")
            return
        hm = head_means(row) or {}
        phm = head_means(parent) if parent else None

        def cell(val, pval):
            if val is None:
                return "—"
            s = f"{val:.3f}"
            if pval is not None:
                s += f" ({val - pval:+.3f})"
            return s

        p = parent or {}
        print(f"| {label} | "
              f"{cell(row.get('wc/R@20'), p.get('wc/R@20') if parent else None)} | "
              f"{cell(row.get('wc/mR@20'), p.get('wc/mR@20') if parent else None)} | "
              f"{cell(hm.get('attention'), phm.get('attention') if phm else None)} | "
              f"{cell(hm.get('spatial'), phm.get('spatial') if phm else None)} | "
              f"{cell(hm.get('contacting'), phm.get('contacting') if phm else None)} | "
              f"{cell(occl(row, 'occpair/R@20'), occl(parent, 'occpair/R@20') if parent else None)} | "
              f"{target} |")

    tier_line("worldwise (v2e, full)", "base")
    for t in TIER_INFO:
        tier_line(f"+{t}", t)

    # ---------- 3. Per-predicate movers ----------
    print("\n## Biggest per-predicate movers vs parent (wc R@20)\n")
    for t, (parent_key, _target) in TIER_INFO.items():
        row, parent = rows.get(t), rows.get(parent_key)
        if row is None or parent is None:
            continue
        vec = row.get("wc/per_predicate_R@20") or {}
        pvec = parent.get("wc/per_predicate_R@20") or {}
        deltas = sorted(
            ((name, vec[name] - pvec.get(name, 0.0)) for name in vec),
            key=lambda kv: -abs(kv[1]),
        )[:6]
        movers = ", ".join(f"{n} {d:+.3f}" for n, d in deltas if abs(d) >= 0.005)
        print(f"- **+{t}** (vs {parent_key}): {movers or 'no movers ≥ 0.005'}")

    print("\nGate rule: a plugin survives if its TARGET metric improves >= +0.003 "
          "without degrading the other task. Log verdicts in "
          "docs/DECISION_LOG.md.")


if __name__ == "__main__":
    main()
