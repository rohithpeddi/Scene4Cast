"""
Per-visibility-bucket recall breakdown across methods (offline, no model).
==========================================================================

Reads the dumps produced by tools/dump_predictions.py and computes, per method
and per constraint (With / No), recall split by the visibility of the pair's
endpoints:

  OO  observed-observed        (person + detected object)
  OU  observed-unobserved      (person + off-screen / RAG object)   <-- the claim
  UO/UU  person-unobserved cases (near-empty; reported for transparency)

For the OU bucket it additionally reports recall on **non-trivial positives**
(dropping not_looking_at / not_contacting, and a stricter variant also dropping
unsure) -- the metric the reviewer asks for -- plus the GT predicate composition
of the bucket, which directly quantifies "overwhelmingly not_looking_at /
not_contacting".

Usage:
    python tools/bucketed_breakdown.py --dumps results/bucket_dumps \
        --out results/bucketed_breakdown.json
"""

import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from dataloader.world_ag_dataset import (                 # noqa: E402
    ATTENTION_RELATIONSHIPS, SPATIAL_RELATIONSHIPS, CONTACTING_RELATIONSHIPS,
    OBJECT_CLASSES,
)
from lib.supervised.evaluation_recall import BasicSceneGraphEvaluator  # noqa: E402
from lib.supervised.evaluation_recall_bucketed import (   # noqa: E402
    BucketAccumulator, evaluate_wsgg_video_bucketed,
)

PRED_NAMES = list(ATTENTION_RELATIONSHIPS) + list(SPATIAL_RELATIONSHIPS) \
    + list(CONTACTING_RELATIONSHIPS)
KS = (10, 20, 50, 100)


def make_evaluator(mode, constraint):
    return BasicSceneGraphEvaluator(
        mode=mode,
        AG_object_classes=OBJECT_CLASSES,
        AG_all_predicates=PRED_NAMES,
        AG_attention_predicates=list(ATTENTION_RELATIONSHIPS),
        AG_spatial_predicates=list(SPATIAL_RELATIONSHIPS),
        AG_contacting_predicates=list(CONTACTING_RELATIONSHIPS),
        iou_threshold=0.5, save_file=os.devnull, constraint=constraint,
    )


def process_dump(path):
    with open(path, "rb") as f:
        blob = pickle.load(f)
    meta, records = blob["meta"], blob["records"]
    mode = meta["mode"]

    out = {"meta": meta, "constraints": {}}
    for cname, cval in (("wc", "with"), ("nc", "no")):
        ev = make_evaluator(mode, cval)
        acc = BucketAccumulator(
            n_att=len(ATTENTION_RELATIONSHIPS),
            n_spa=len(SPATIAL_RELATIONSHIPS),
            n_con=len(CONTACTING_RELATIONSHIPS), ks=KS,
        )
        for rec in records:
            evaluate_wsgg_video_bucketed(rec, acc, ev, mode=mode)

        res = {"bucket_sizes": acc.bucket_sizes(), "buckets": {}}
        for b in ("OO", "OU", "UO", "UU"):
            if b not in acc.total:
                continue
            entry = {"R": {}, "mR": {}}
            for k in KS:
                r, tot = acc.recall(b, k)
                mr, npred = acc.mean_recall(b, k)
                entry["R"][k] = None if np.isnan(r) else round(r, 6)
                entry["mR"][k] = None if np.isnan(mr) else round(mr, 6)
            entry["n_gt"] = int(acc.total[b].sum())
            res["buckets"][b] = entry

        # non-trivial positives in the unobserved-object bucket (OU)
        if "OU" in acc.total:
            nt = {"drop_trivial": {"R": {}, "mR": {}, "n_gt": None},
                  "drop_trivial_and_unsure": {"R": {}, "mR": {}, "n_gt": None}}
            for k in KS:
                r1, t1 = acc.recall("OU", k, drop_trivial=True)
                m1, _ = acc.mean_recall("OU", k, drop_trivial=True)
                r2, t2 = acc.recall("OU", k, drop_trivial=True, drop_unsure=True)
                m2, _ = acc.mean_recall("OU", k, drop_trivial=True, drop_unsure=True)
                nt["drop_trivial"]["R"][k] = None if np.isnan(r1) else round(r1, 6)
                nt["drop_trivial"]["mR"][k] = None if np.isnan(m1) else round(m1, 6)
                nt["drop_trivial_and_unsure"]["R"][k] = None if np.isnan(r2) else round(r2, 6)
                nt["drop_trivial_and_unsure"]["mR"][k] = None if np.isnan(m2) else round(m2, 6)
                nt["drop_trivial"]["n_gt"] = t1
                nt["drop_trivial_and_unsure"]["n_gt"] = t2
            res["ou_nontrivial"] = nt
            res["ou_composition"] = acc.composition("OU", PRED_NAMES)
            res["oo_composition"] = acc.composition("OO", PRED_NAMES)

        out["constraints"][cname] = res
    return out


def _fmt(x):
    return "  n/a" if x is None else f"{100 * x:5.1f}"


def print_report(all_results):
    # group by mode
    for mode in ("predcls", "sgdet"):
        rows = [r for r in all_results if r["meta"]["mode"] == mode]
        if not rows:
            continue
        rows.sort(key=lambda r: r["meta"]["method"])
        print("\n" + "=" * 78)
        print(f"MODE = {mode.upper()}   (No-Constraint = the reviewer's Table-2 target)")
        print("=" * 78)

        # bucket composition is method-invariant (same GT) -- print once
        sample = rows[0]["constraints"]["nc"]
        sizes = sample["bucket_sizes"]
        tot = sum(sizes.values())
        print("\n[GT relationship composition by visibility bucket]")
        for b in ("OO", "OU", "UO", "UU"):
            if b in sizes:
                print(f"  {b}: {sizes[b]:>8,d}  ({100*sizes[b]/tot:4.1f}% of all GT triplets)")
        if "ou_composition" in sample:
            print("\n[OU (unobserved-object) bucket -- GT predicate composition]")
            comp = sorted(sample["ou_composition"].items(),
                          key=lambda kv: -kv[1][0])
            shown = 0
            trivial_pct = 0.0
            for name, (c, pct) in comp:
                if name in ("not_looking_at", "not_contacting"):
                    trivial_pct += pct
                if c > 0 and shown < 10:
                    tag = "  <-- TRIVIAL" if name in ("not_looking_at", "not_contacting") else ""
                    print(f"    {name:<22} {c:>7,d}  ({pct:4.1f}%){tag}")
                    shown += 1
            print(f"    ...  not_looking_at + not_contacting = {trivial_pct:.1f}% of OU positives")

        # per-method recall table (No-Constraint)
        for cname in ("nc", "wc"):
            label = "No-Constraint" if cname == "nc" else "With-Constraint"
            print(f"\n[{label}]  R@50 / mR@50 (%)  per bucket, and OU non-trivial")
            hdr = (f"  {'method':<16} {'OO R':>6} {'OO mR':>6} "
                   f"{'OU R':>6} {'OU mR':>6} "
                   f"{'OU-nt R':>8} {'OU-nt mR':>9}")
            print(hdr)
            print("  " + "-" * (len(hdr) - 2))
            for r in rows:
                c = r["constraints"][cname]
                b = c["buckets"]
                oo = b.get("OO", {"R": {}, "mR": {}})
                ou = b.get("OU", {"R": {}, "mR": {}})
                nt = c.get("ou_nontrivial", {}).get("drop_trivial", {"R": {}, "mR": {}})
                print(f"  {r['meta']['method']:<16} "
                      f"{_fmt(oo['R'].get(50)):>6} {_fmt(oo['mR'].get(50)):>6} "
                      f"{_fmt(ou['R'].get(50)):>6} {_fmt(ou['mR'].get(50)):>6} "
                      f"{_fmt(nt['R'].get(50)):>8} {_fmt(nt['mR'].get(50)):>9}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", default=os.path.join(REPO, "results", "bucket_dumps"),
                    help="directory of *.pkl dumps, or a single .pkl file")
    ap.add_argument("--out", default=os.path.join(REPO, "results", "bucketed_breakdown.json"))
    args = ap.parse_args()

    if os.path.isdir(args.dumps):
        paths = sorted(glob.glob(os.path.join(args.dumps, "*.pkl")))
    else:
        paths = [args.dumps]
    if not paths:
        print(f"No dumps found in {args.dumps}")
        return

    all_results = []
    for p in paths:
        print(f"[analyze] {os.path.basename(p)}")
        all_results.append(process_dump(p))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nsaved -> {args.out}")

    print_report(all_results)


if __name__ == "__main__":
    main()
