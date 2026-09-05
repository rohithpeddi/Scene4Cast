"""Standard wc/nc R/mR/hR over a dump pkl (all frames), reeval-comparable."""
import json
import pickle
import sys

sys.path.insert(0, "/home/rxp190007/CODE/Scene4Cast")

from dataloader.world_ag_dataset import (
    ATTENTION_RELATIONSHIPS, SPATIAL_RELATIONSHIPS, CONTACTING_RELATIONSHIPS,
    OBJECT_CLASSES,
)
from lib.supervised.evaluation_recall import (
    BasicSceneGraphEvaluator, evaluate_wsgg_video,
)

PRED = list(ATTENTION_RELATIONSHIPS) + list(SPATIAL_RELATIONSHIPS) + list(CONTACTING_RELATIONSHIPS)
KS = [10, 20, 50, 100]


def make_ev(mode, constraint):
    import os
    return BasicSceneGraphEvaluator(
        mode=mode, AG_object_classes=OBJECT_CLASSES, AG_all_predicates=PRED,
        AG_attention_predicates=list(ATTENTION_RELATIONSHIPS),
        AG_spatial_predicates=list(SPATIAL_RELATIONSHIPS),
        AG_contacting_predicates=list(CONTACTING_RELATIONSHIPS),
        iou_threshold=0.5, save_file=os.devnull, constraint=constraint,
    )


def main(path):
    blob = pickle.load(open(path, "rb"))
    mode = blob["meta"]["mode"]
    evs = {"wc": make_ev(mode, "with"), "nc": make_ev(mode, "no")}
    for rec in blob["records"]:
        for ev in evs.values():
            evaluate_wsgg_video(rec, ev, mode=mode, verbose=False)
    out = {"meta": blob["meta"], "schemes": {}}
    for name, ev in evs.items():
        s = ev.fetch_stats_json()
        out["schemes"][name] = {
            "R": {k: round(s["recall"].get(k, 0.0), 4) for k in KS},
            "mR": {k: round(s["mean_recall"].get(k, 0.0), 4) for k in KS},
            "hR": {k: round(s["harmonic_mean_recall"].get(k, 0.0), 4) for k in KS},
        }
    print(json.dumps(out["schemes"], indent=1))
    outp = path.replace(".pkl", "_standard_eval.json")
    json.dump(out, open(outp, "w"), indent=1)
    print("wrote", outp)


if __name__ == "__main__":
    main(sys.argv[1])
