"""
Forward-pass smoke tests for the WSGG method ladder + WorldWise plugins.

Torch-only — no dataset, no GPU required. Run on the training server:

    python -m pytest tests/test_wsgg_forward.py -v
    # or, without pytest:
    python tests/test_wsgg_forward.py

Checks per model:
  - forward runs on random padded tensors (with and without camera poses)
  - output dict has the expected keys and (T, K, C) shapes
  - outputs are finite
  - WorldWise: each plugin flag individually, and all together
  - WorldWise with all plugin flags off contains no plugin modules
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

T, N, K = 6, 5, 4
NUM_CLASSES, N_ATT, N_SPA, N_CON = 37, 3, 6, 17


def make_config(**overrides):
    cfg = dict(
        data_path="",              # no CLIP file → random-embedding fallback
        clip_embeddings_path="",
        d_model=64, d_struct=64, d_visual=64, d_detector_roi=128,
        d_camera=32, d_motion=16,
        n_heads=4, d_feedforward=128, dropout=0.1,
        d_rel=64, d_text=32, d_union_roi=128,
        n_rel_layers=1, n_rel_heads=4,
        n_temporal_edge_layers=1, n_temporal_obj_layers=1,
        n_gnn_layers=1, n_cross_attn_layers=1, n_self_attn_layers=1,
        max_objects=N,
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def make_batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    valid = torch.ones(T, N, dtype=torch.bool)
    valid[:, -1] = False                      # one padding slot
    valid[0, 2] = False                       # a slot gap (velocity gating)
    vis = valid.clone()
    vis[1:3, 1] = False                       # object 1 occluded in frames 1-2

    person_idx = torch.zeros(T, K, dtype=torch.long)
    object_idx = torch.randint(1, N - 1, (T, K), generator=g)
    pair_valid = torch.ones(T, K, dtype=torch.bool)
    pair_valid[:, -1] = False                 # one padding pair

    pose = torch.eye(4).unsqueeze(0).repeat(T, 1, 1)
    pose[:, :3, 3] = torch.randn(T, 3, generator=g) * 0.1

    return dict(
        visual_features_seq=torch.randn(T, N, 128, generator=g),
        corners_seq=torch.randn(T, N, 8, 3, generator=g),
        valid_mask_seq=valid,
        visibility_mask_seq=vis,
        person_idx_seq=person_idx,
        object_idx_seq=object_idx,
        pair_valid=pair_valid,
        camera_pose_seq=pose,
        union_features_seq=torch.randn(T, K, 128, generator=g),
    )


def check_outputs(out, model_name):
    expected = {
        "attention_distribution": (T, K, N_ATT),
        "spatial_distribution": (T, K, N_SPA),
        "contacting_distribution": (T, K, N_CON),
        "attention_logits": (T, K, N_ATT),
        "spatial_logits": (T, K, N_SPA),
        "contacting_logits": (T, K, N_CON),
        "node_logits": (T, N, NUM_CLASSES),
    }
    for key, shape in expected.items():
        assert key in out, f"{model_name}: missing key {key}"
        assert tuple(out[key].shape) == shape, (
            f"{model_name}: {key} shape {tuple(out[key].shape)} != {shape}"
        )
        assert torch.isfinite(out[key]).all(), f"{model_name}: {key} has NaN/Inf"


def _run(model, batch, **extra):
    return model(**batch, **extra)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def test_baseline_forwards():
    from lib.supervised.baselines.w_sttran.w_sttran import WSTTran
    from lib.supervised.baselines.w_sttran.w_sttran_pp import WSTTranPP
    from lib.supervised.baselines.w_dsgdetr.w_dsgdetr import WDSGDetr
    from lib.supervised.baselines.w_dsgdetr.w_dsgdetr_pp import WDSGDetrPP

    cfg = make_config()
    batch = make_batch()
    gt_labels = torch.randint(0, NUM_CLASSES, (T, N))

    for cls in (WSTTran, WSTTranPP, WDSGDetr, WDSGDetrPP):
        model = cls(cfg, NUM_CLASSES, N_ATT, N_SPA, N_CON).eval()
        with torch.no_grad():
            check_outputs(_run(model, batch), cls.__name__)
            # No camera poses
            b2 = dict(batch, camera_pose_seq=None)
            check_outputs(_run(model, b2), f"{cls.__name__}[no-cam]")
            # predcls GT text pathway
            check_outputs(
                _run(model, batch, node_labels_seq=gt_labels),
                f"{cls.__name__}[gt-labels]",
            )
        print(f"  OK {cls.__name__}")


def test_ladder_param_monotonicity():
    from lib.supervised.baselines.w_sttran.w_sttran import WSTTran
    from lib.supervised.baselines.w_sttran.w_sttran_pp import WSTTranPP
    from lib.supervised.baselines.w_dsgdetr.w_dsgdetr import WDSGDetr
    from lib.supervised.baselines.w_dsgdetr.w_dsgdetr_pp import WDSGDetrPP
    from lib.supervised.worldwise.worldwise import WorldWise

    cfg = make_config()
    counts = []
    for cls in (WSTTran, WSTTranPP, WDSGDetr, WDSGDetrPP, WorldWise):
        m = cls(cfg, NUM_CLASSES, N_ATT, N_SPA, N_CON)
        n_params = sum(p.numel() for p in m.parameters())
        counts.append((cls.__name__, n_params))
    for (name_a, a), (name_b, b) in zip(counts, counts[1:]):
        assert a < b, f"ladder param count not monotone: {name_a}={a} >= {name_b}={b}"
    print("  OK ladder params:", ", ".join(f"{n}={c:,}" for n, c in counts))


# ---------------------------------------------------------------------------
# WorldWise plugins
# ---------------------------------------------------------------------------

PLUGIN_FLAGS = [
    "use_pair_geometry",
    "use_soft_text_embedding",
    "use_geometric_attn_bias",
    "use_predicate_prototypes",
    "use_cross_object_retrieval",
    "use_energy_refinement",
]


def _worldwise(cfg):
    from lib.supervised.worldwise.worldwise import WorldWise
    return WorldWise(cfg, NUM_CLASSES, N_ATT, N_SPA, N_CON)


def test_worldwise_plugins():
    batch = make_batch()
    gt_con = (torch.rand(T, K, N_CON) > 0.7).float()
    gt_labels = torch.randint(0, NUM_CLASSES, (T, N))

    # All plugin flags off — the post-fix I-0 reference
    base = _worldwise(make_config()).eval()
    with torch.no_grad():
        check_outputs(_run(base, batch), "WorldWise[I-0]")
    for mod in ("pair_geo_mlp",):
        assert not hasattr(base.rel_predictor, mod), f"I-0 must not own {mod}"
    assert not hasattr(base, "prototype_memory")
    assert not hasattr(base, "energy_refiner")

    # Each plugin individually, then all together
    for flags in [[f] for f in PLUGIN_FLAGS] + [PLUGIN_FLAGS]:
        cfg = make_config(**{f: True for f in flags})
        model = _worldwise(cfg)
        label = "+".join(f.replace("use_", "") for f in flags)

        # train-mode pass (exercises masking, EMA update, prototype update)
        model.train()
        out = _run(
            model, batch, p_mask_visible=0.5,
            node_labels_seq=None, gt_contacting_seq=gt_con,
        )
        check_outputs(out, f"WorldWise[{label}][train]")
        out["attention_logits"].sum().backward()  # gradients flow

        model.eval()
        with torch.no_grad():
            check_outputs(_run(model, batch), f"WorldWise[{label}][eval]")
            check_outputs(
                _run(model, batch, node_labels_seq=gt_labels),
                f"WorldWise[{label}][gt-labels]",
            )
        print(f"  OK WorldWise[{label}]")

    # I-8 must emit the stability-loss tensors
    cfg = make_config(use_energy_refinement=True)
    model = _worldwise(cfg).eval()
    with torch.no_grad():
        out = _run(model, batch)
    assert "enriched" in out and "h_prev" in out, "I-8 must emit enriched/h_prev"


def test_worldwise_loss():
    from lib.supervised.worldwise.loss import WorldWiseLoss

    batch = make_batch()
    model = _worldwise(make_config(use_energy_refinement=True)).train()
    out = _run(
        model, batch, p_mask_visible=0.5,
        gt_contacting_seq=(torch.rand(T, K, N_CON) > 0.7).float(),
    )

    loss_fn = WorldWiseLoss(
        lambda_vlm=0.2, lambda_recon=0.5, lambda_recon_dominance=0.1,
        p_simulate_unseen=0.3, label_smoothing=0.2, mode="predcls",
        use_logit_adjustment=False,
        use_confidence_weighted_vlm=True,   # exercises the I-5 path
        lambda_stability=0.1,               # exercises the I-8 term
    )
    losses = loss_fn(
        predictions=out,
        gt_attention=torch.randint(0, N_ATT, (T, K)),
        gt_spatial=(torch.rand(T, K, N_SPA) > 0.7).float(),
        gt_contacting=(torch.rand(T, K, N_CON) > 0.7).float(),
        pair_valid=batch["pair_valid"],
        visibility_mask=batch["visibility_mask_seq"],
        person_idx=batch["person_idx_seq"],
        object_idx=batch["object_idx_seq"],
        valid_mask=batch["valid_mask_seq"],
        gt_node_labels=torch.randint(0, NUM_CLASSES, (T, N)),
        vlm_confidence=torch.rand(T, K),    # I-5 weighted path
    )
    assert torch.isfinite(losses["total"]), "loss total is NaN/Inf"
    assert "stability_loss" in losses, "lambda_stability>0 must emit stability_loss"
    losses["total"].backward()
    print(f"  OK WorldWiseLoss total={losses['total'].item():.4f}")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_baseline_forwards()
    test_ladder_param_monotonicity()
    test_worldwise_plugins()
    test_worldwise_loss()
    print("\nAll WSGG forward smoke tests passed.")
