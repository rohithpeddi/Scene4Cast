"""
Per-bucket recall breakdown for WSGG (rebuttal analysis)
========================================================

Motivation
----------
The main claim of the benchmark is *reasoning about unobserved objects*. A
reviewer objects that (i) only ~14% of relationship instances involve an
unobserved object and (ii) those are overwhelmingly the trivial "no
interaction" labels (``not_looking_at`` / ``not_contacting``), so a saturated
No-Constraint mR@50 does not demonstrate real reasoning.

This module answers that directly by computing recall *split by the visibility
of the two entities in each relation*:

  * OO : observed-observed   (person observed  + object observed)
  * OU : observed-unobserved (person observed  + object UNobserved)
  * UU : unobserved-unobserved (both unobserved -- ~empty in this benchmark,
         since every relation is anchored to the always-present person node)

and, within the unobserved bucket, recall restricted to **non-trivial
positives** (dropping ``not_looking_at`` / ``not_contacting`` -- and optionally
``unsure`` -- from the denominator).

Design
------
GT triplets are built in *exactly* the same order as
``evaluation_recall.evaluate_wsgg_video`` so that the ``pred_to_gt`` matching
returned by the standard ``evaluate_from_dict`` (same ranking, same IoU matcher,
same constraint handling) aligns index-for-index with a parallel *tag* array we
build here. We therefore reuse the real metric's matching code verbatim and only
add bookkeeping -- the per-bucket numbers are guaranteed consistent with the
headline R@K / mR@K.

Recall here is **micro-averaged** (pooled hits / pooled counts over the whole
corpus), which is the appropriate choice for a breakdown: the rare non-trivial
unobserved bucket has too few instances per frame for the standard per-image
macro average to be stable. R@K/mR@K on the OO bucket will therefore differ
slightly from the paper's per-image-averaged headline numbers; that is expected
and is stated in the report.
"""

from collections import defaultdict

import numpy as np

from lib.supervised.evaluation_recall import evaluate_from_dict


# Attention occupies global predicate indices [0, n_att), spatial the next
# n_spa, contacting the rest. With the canonical vocab: att=3, spa=6, con=17.
# not_looking_at = attention index 1 -> global 1
# not_contacting = contacting index 8 -> global (n_att + n_spa + 8)
def _trivial_global_indices(n_att, n_spa):
    not_looking_at = 1                      # attention idx 1
    not_contacting = n_att + n_spa + 8      # contacting idx 8
    unsure = 2                              # attention idx 2 (ambiguous)
    return {"not_looking_at": not_looking_at,
            "not_contacting": not_contacting,
            "unsure": unsure}


def bucket_name(vis_person, vis_object):
    """Human-readable bucket label from endpoint visibility.

    Every relation is a person<->object pair; the person node is (almost)
    always observed, so OU is 'unobserved object' and UU/UO are the
    (near-empty) person-unobserved cases -- kept separate for transparency.
    """
    if vis_person and vis_object:
        return "OO"           # observed - observed
    if vis_person and not vis_object:
        return "OU"           # observed - unobserved  (the object is off-screen)
    if (not vis_person) and vis_object:
        return "UO"           # unobserved person - observed object (rare)
    return "UU"               # both unobserved (rare)


class BucketAccumulator:
    """Corpus-level pooled counts for per-bucket recall.

    total[bucket][pred]          -> # GT triplets of `pred` in `bucket`
    hit[bucket][k][pred]         -> # of those recalled within top-k
    Everything is integer; recall is computed at report time.
    """

    KS = (10, 20, 50, 100)

    def __init__(self, n_att=3, n_spa=6, n_con=17, ks=KS):
        self.n_att, self.n_spa, self.n_con = n_att, n_spa, n_con
        self.num_rel = n_att + n_spa + n_con
        self.ks = tuple(ks)
        self.trivial = _trivial_global_indices(n_att, n_spa)
        # counts
        self.total = defaultdict(lambda: np.zeros(self.num_rel, dtype=np.int64))
        self.hit = defaultdict(
            lambda: {k: np.zeros(self.num_rel, dtype=np.int64) for k in self.ks}
        )
        self.n_videos = 0
        self.n_frames = 0

    # -- accumulation -------------------------------------------------------
    def add_frame(self, gt_tags, pred_to_gt):
        """gt_tags: list of (bucket, global_pred_idx) aligned to gt order.
        pred_to_gt: list (per pred) of matched gt indices (from evaluate_recall)."""
        self.n_frames += 1
        for bucket, pidx in gt_tags:
            self.total[bucket][pidx] += 1
        for k in self.ks:
            # union of gt indices hit within the top-k predictions
            hit_gt = set()
            for lst in pred_to_gt[:k]:
                hit_gt.update(int(g) for g in lst)
            for g in hit_gt:
                if g < len(gt_tags):
                    bucket, pidx = gt_tags[g]
                    self.hit[bucket][k][pidx] += 1

    # -- reporting ----------------------------------------------------------
    def _pred_mask(self, drop_trivial=False, drop_unsure=False):
        m = np.ones(self.num_rel, dtype=bool)
        if drop_trivial:
            m[self.trivial["not_looking_at"]] = False
            m[self.trivial["not_contacting"]] = False
        if drop_unsure:
            m[self.trivial["unsure"]] = False
        return m

    def recall(self, bucket, k, drop_trivial=False, drop_unsure=False):
        """Micro R@k over a bucket (optionally excluding trivial predicates)."""
        if bucket not in self.total:
            return float("nan"), 0
        m = self._pred_mask(drop_trivial, drop_unsure)
        tot = int(self.total[bucket][m].sum())
        hit = int(self.hit[bucket][k][m].sum())
        return (hit / tot if tot > 0 else float("nan")), tot

    def mean_recall(self, bucket, k, drop_trivial=False, drop_unsure=False):
        """Macro-over-predicates mR@k (predicates with >0 GT in bucket)."""
        if bucket not in self.total:
            return float("nan"), 0
        m = self._pred_mask(drop_trivial, drop_unsure)
        tot = self.total[bucket]
        hit = self.hit[bucket][k]
        vals = []
        for p in range(self.num_rel):
            if m[p] and tot[p] > 0:
                vals.append(hit[p] / tot[p])
        return (float(np.mean(vals)) if vals else float("nan")), len(vals)

    def composition(self, bucket, predicate_names):
        """GT predicate histogram for a bucket -> {name: (count, pct)}."""
        if bucket not in self.total:
            return {}
        tot = self.total[bucket]
        total = int(tot.sum())
        out = {}
        for i, name in enumerate(predicate_names):
            c = int(tot[i])
            out[name] = (c, (100.0 * c / total if total > 0 else 0.0))
        return out

    def bucket_sizes(self):
        return {b: int(self.total[b].sum()) for b in self.total}


def evaluate_wsgg_video_bucketed(pred_pkl, acc, evaluator, mode="predcls"):
    """Tag one video's GT triplets by visibility bucket and accumulate recall.

    Mirrors ``evaluation_recall.evaluate_wsgg_video`` for GT/pred construction,
    then reuses ``evaluate_from_dict`` (same ranking + IoU matcher + constraint)
    for the actual matching. Requires ``pred_pkl['visibility_mask']`` (N_max,).

    Args:
        pred_pkl: per-frame prediction dict (as produced by dump_predictions),
            must additionally contain 'visibility_mask'.
        acc:       BucketAccumulator to update.
        evaluator: a BasicSceneGraphEvaluator (supplies mode, constraint,
            iou_threshold, num_rel and the predicate vocab). Its result_dict
            is NOT mutated -- we pass a throwaway dict to evaluate_from_dict.
        mode:      'predcls' or 'sgdet'.
    """
    pair_valid = pred_pkl["pair_valid"].astype(bool)
    person_idx = pred_pkl["person_idx"]
    object_idx = pred_pkl["object_idx"]
    object_classes = pred_pkl["object_classes"]
    bboxes_2d = pred_pkl["bboxes_2d"]
    valid_mask = pred_pkl["valid_mask"].astype(bool)
    visibility = pred_pkl["visibility_mask"].astype(bool)

    att_dist = pred_pkl["attention_distribution"]
    spa_dist = pred_pkl["spatial_distribution"]
    con_dist = pred_pkl["contacting_distribution"]

    gt_attention = pred_pkl["gt_attention"]
    gt_spatial = pred_pkl["gt_spatial"]
    gt_contacting = pred_pkl["gt_contacting"]

    if int(pair_valid.sum()) == 0:
        return

    n_att = len(evaluator.AG_attention_predicates)
    n_spa = len(evaluator.AG_spatial_predicates)
    spa_offset = n_att
    con_offset = n_att + n_spa

    gt_boxes = bboxes_2d.astype(np.float32)
    gt_classes = object_classes.astype(np.int64)

    # ---- Build GT relations AND aligned tags (same loop order as the metric) ----
    gt_relations = []
    gt_tags = []  # (bucket, global_pred_idx) per appended relation

    def _bucket(p_idx, o_idx):
        vp = bool(visibility[p_idx]) if p_idx < len(visibility) else True
        vo = bool(visibility[o_idx]) if o_idx < len(visibility) else True
        return bucket_name(vp, vo)

    for k in range(len(pair_valid)):
        if not pair_valid[k]:
            continue
        p_idx = int(person_idx[k])
        o_idx = int(object_idx[k])
        b = _bucket(p_idx, o_idx)

        att_label = int(gt_attention[k])
        gt_relations.append([p_idx, o_idx, att_label])
        gt_tags.append((b, att_label))

        for s in range(gt_spatial.shape[1]):
            if gt_spatial[k, s] > 0.5:
                gt_relations.append([o_idx, p_idx, spa_offset + s])
                gt_tags.append((b, spa_offset + s))

        for c in range(gt_contacting.shape[1]):
            if gt_contacting[k, c] > 0.5:
                gt_relations.append([p_idx, o_idx, con_offset + c])
                gt_tags.append((b, con_offset + c))

    if not gt_relations:
        return

    gt_entry = {
        "gt_classes": gt_classes,
        "gt_relations": np.array(gt_relations, dtype=np.int64),
        "gt_boxes": gt_boxes,
    }

    # ---- Build pred entry (identical to evaluate_wsgg_video) ----
    valid_pidx = person_idx[pair_valid]
    valid_oidx = object_idx[pair_valid]
    pair_idx_att = np.stack([valid_pidx, valid_oidx], axis=1)
    pair_idx_spa = np.stack([valid_oidx, valid_pidx], axis=1)
    pair_idx_con = np.stack([valid_pidx, valid_oidx], axis=1)
    rels_i = np.concatenate([pair_idx_att, pair_idx_spa, pair_idx_con], axis=0)

    valid_att = att_dist[pair_valid]
    valid_spa = spa_dist[pair_valid]
    valid_con = con_dist[pair_valid]
    K_v = int(pair_valid.sum())
    n_a, n_s, n_c = valid_att.shape[1], valid_spa.shape[1], valid_con.shape[1]
    scores_att = np.concatenate([valid_att, np.zeros((K_v, n_s + n_c))], axis=1)
    scores_spa = np.concatenate([np.zeros((K_v, n_a)), valid_spa, np.zeros((K_v, n_c))], axis=1)
    scores_con = np.concatenate([np.zeros((K_v, n_a + n_s)), valid_con], axis=1)
    rel_scores = np.concatenate([scores_att, scores_spa, scores_con], axis=0)

    if mode == "predcls":
        pred_boxes = gt_boxes.copy()
        pred_classes = gt_classes.copy()
        obj_scores = np.ones(len(gt_classes), dtype=np.float32)
    else:
        # IMPORTANT: reproduce evaluate_wsgg_video's *effective* sgdet behavior.
        # There, the `gt_boxes = gt_bboxes_2d` line rebinds a local AFTER
        # gt_entry is already built, so gt_entry keeps the detector boxes
        # (bboxes_2d). gt_bboxes_2d is all-zero in this pipeline regardless, so
        # sgdet is evaluated predcls-style: detector boxes as both GT and pred
        # (localization is enforced upstream via IoU-matched detections). We do
        # NOT override gt_entry["gt_boxes"] here -- doing so (with zero boxes)
        # would drive recall to 0 and diverge from the paper's Table 2.
        pred_boxes = pred_pkl.get("bboxes_2d", gt_boxes).astype(np.float32)
        pred_classes = pred_pkl.get("pred_labels", gt_classes).astype(np.int64)
        obj_scores = pred_pkl.get("pred_scores", np.ones(len(gt_classes))).astype(np.float32)

    pred_entry = {
        "pred_boxes": pred_boxes,
        "pred_classes": pred_classes,
        "pred_rel_inds": rels_i,
        "obj_scores": obj_scores,
        "rel_scores": rel_scores,
    }

    # ---- Reuse the real matcher/ranker; ignore its result_dict ----
    throwaway = {
        evaluator.mode + "_recall": {k: [] for k in acc.ks},
        evaluator.mode + "_mean_recall_collect": {
            k: [[] for _ in range(acc.num_rel)] for k in acc.ks
        },
    }
    pred_to_gt, _, _ = evaluate_from_dict(
        gt_entry, pred_entry, evaluator.mode, throwaway,
        iou_thresh=evaluator.iou_threshold,
        method=evaluator.constraint,
        threshold=evaluator.semi_threshold,
        num_rel=acc.num_rel,
    )

    acc.add_frame(gt_tags, pred_to_gt)
