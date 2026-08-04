"""Render the per-bucket breakdown table from bucketed_breakdown.json.

Robust to JSON int->str key coercion (K keys become strings after a
json round-trip). Prints predcls + sgdet, both constraints.
"""
import json
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "results/bucketed_breakdown.json"
METHOD_ORDER = ["w_sttran", "w_sttran_pp", "w_dsgdetr", "w_dsgdetr_pp", "worldwise"]
PRETTY = {"w_sttran": "W-STTran", "w_sttran_pp": "W-STTran++",
          "w_dsgdetr": "W-DSGDetr", "w_dsgdetr_pp": "W-DSGDetr++",
          "worldwise": "WorldWise"}


def g(d, k):
    """Lookup tolerant to int/str key."""
    if d is None:
        return None
    return d.get(k, d.get(str(k), d.get(int(k) if str(k).isdigit() else k)))


def f(x):
    return "  -  " if x is None else f"{100 * x:5.1f}"


def main():
    data = json.load(open(PATH))
    by = {(r["meta"]["mode"], r["meta"]["method"]): r for r in data}

    for mode in ("predcls", "sgdet"):
        rows = [by[(mode, m)] for m in METHOD_ORDER if (mode, m) in by]
        if not rows:
            continue
        print("\n" + "=" * 92)
        print(f"MODE = {mode.upper()}")
        print("=" * 92)

        sample = rows[0]["constraints"]["nc"]
        sizes = sample["bucket_sizes"]
        tot = sum(sizes.values())
        print("\nGT composition by visibility bucket:")
        for b in ("OO", "OU", "UO", "UU"):
            if b in sizes:
                lab = {"OO": "observed-observed", "OU": "observed-UNobserved",
                       "UO": "unobs.person-obs.obj", "UU": "both-unobserved"}[b]
                print(f"   {b} ({lab:<22}): {sizes[b]:>10,d}  ({100*sizes[b]/tot:4.1f}%)")

        comp = sample.get("ou_composition", {})
        triv = sum(v[1] for k, v in comp.items()
                   if k in ("not_looking_at", "not_contacting"))
        real = 100.0 - triv
        print(f"\nOU (unobserved-object) predicate mix:  "
              f"trivial (not_looking_at+not_contacting) = {triv:.1f}%  |  "
              f"REAL positives = {real:.1f}%")

        for cname, clabel in (("nc", "No-Constraint"), ("wc", "With-Constraint")):
            print(f"\n[{clabel}]  (%)   OO=observed pair   OU=unobserved-object   "
                  f"OU-nt=OU excluding not_looking_at/not_contacting")
            print(f"   {'method':<13}"
                  f"{'OO R@20':>8}{'R@50':>7}{'mR@50':>7}   "
                  f"{'OU R@20':>8}{'R@50':>7}{'mR@50':>7}   "
                  f"{'OU-nt R@50':>11}{'mR@50':>7}")
            print("   " + "-" * 84)
            for r in rows:
                c = r["constraints"][cname]
                oo = c["buckets"].get("OO", {})
                ou = c["buckets"].get("OU", {})
                nt = c.get("ou_nontrivial", {}).get("drop_trivial", {})
                ooR, ooM = oo.get("R", {}), oo.get("mR", {})
                ouR, ouM = ou.get("R", {}), ou.get("mR", {})
                ntR, ntM = nt.get("R", {}), nt.get("mR", {})
                print(f"   {PRETTY[r['meta']['method']]:<13}"
                      f"{f(g(ooR,20)):>8}{f(g(ooR,50)):>7}{f(g(ooM,50)):>7}   "
                      f"{f(g(ouR,20)):>8}{f(g(ouR,50)):>7}{f(g(ouM,50)):>7}   "
                      f"{f(g(ntR,50)):>11}{f(g(ntM,50)):>7}")


if __name__ == "__main__":
    main()
