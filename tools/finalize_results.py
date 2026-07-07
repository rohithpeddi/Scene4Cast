"""
Finalize the campaign: archive biased values -> re-evaluate everything with
the corrected all-frame protocol -> regenerate the paper tables.

Background: the training-time evaluation scored ONLY the last frame of each
video, inflating every logged metric ~1-4 pts (docs/DECISION_LOG.md,
2026-07-06). This orchestrator produces the final, honest numbers:

  1. ARCHIVE   — moves existing results_tables/*.tex into
                 results_tables/archive_lastframe_<timestamp>/ (kept, not
                 deleted, for provenance); with --force also removes stale
                 reeval_*.json so every cell is recomputed.
  2. RE-EVAL   — runs tools/reeval_test.py (--frames all, --ckpt best) for
                 every cell of the final design, distributed over the GPUs
                 (shared queue, --per-gpu slots). Cells whose
                 results/reeval_<exp>.json already exists are skipped, so
                 the script is resumable.
  3. TABLES    — regenerates results_tables/{predcls,sgdet}_{main,ablation}
                 .tex + preview.tex from the re-eval JSONs
                 (gen_paper_tables --source reeval).

The design (36 cells, both modes):
  baselines @ resnet50 · WorldWise(v2e) @ 4 backbones · 10 ablations @ dinov3l

Usage (training server):
    python tools/finalize_results.py --gpus 0 1 2
    python tools/finalize_results.py --gpus 0 1 2 --force      # recompute all
    python tools/finalize_results.py --dry-run                 # list the plan
"""

import argparse
import datetime
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASELINES = ["w_sttran", "w_sttran_pp", "w_dsgdetr", "w_dsgdetr_pp"]
BACKBONES = ["resnet50", "dinov2b", "dinov2l", "dinov3l"]
MODES = ["predcls", "sgdet"]
METHODS_BACKBONE = "resnet50"
HERO = "dinov3l"
# must match ABLATION_TIERS in tools/gen_grid_configs.py
ABLATION_TIERS = ["v2a", "v2f", "abl_notau", "abl_nomask", "abl_noema",
                  "v2g", "abl_nospatial", "abl_noego", "abl_nomotion",
                  "abl_notempedge"]


def read_experiment_name(cfg_path):
    with open(cfg_path, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"^experiment_name:\s*(\S+)", line)
            if m:
                return m.group(1)
    return None


def build_cells(modes):
    """[(config_relpath, experiment_name)] for the final design only —
    configs dirs also hold legacy files, so the design is enumerated, not
    globbed."""
    cells, missing = [], []
    for mode in modes:
        cfg_dir = os.path.join("configs", "methods", mode)
        stems = [f"{m}_{mode}_{METHODS_BACKBONE}" for m in BASELINES]
        stems += [f"worldwise_{mode}_{b}" for b in BACKBONES]
        stems += [f"worldwise_{t}_{mode}_{HERO}" for t in ABLATION_TIERS]
        for stem in stems:
            cfg = os.path.join(cfg_dir, f"{stem}.yaml")
            cfg_abs = os.path.join(REPO, cfg)
            if not os.path.exists(cfg_abs):
                missing.append(cfg)
                continue
            exp = read_experiment_name(cfg_abs)
            cells.append((cfg, exp))
    return cells, missing


def reeval_done(exp):
    return exp and os.path.exists(os.path.join(REPO, "results", f"reeval_{exp}.json"))


def archive_old_tables(tables_dir):
    tex = [f for f in os.listdir(tables_dir)] if os.path.isdir(tables_dir) else []
    tex = [f for f in tex if f.endswith(".tex")]
    if not tex:
        return None
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = os.path.join(tables_dir, f"archive_lastframe_{stamp}")
    os.makedirs(dest, exist_ok=True)
    for f in tex:
        shutil.move(os.path.join(tables_dir, f), os.path.join(dest, f))
    return dest


def worker(gpu_id, slot, work_q, status, lock, py):
    tag = f"gpu{gpu_id}.{slot}"
    log_dir = os.path.join(REPO, "logs", "reeval")
    os.makedirs(log_dir, exist_ok=True)
    while True:
        try:
            cfg, exp = work_q.get_nowait()
        except queue.Empty:
            return
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_id))
        log_path = os.path.join(log_dir, f"{exp}.log")
        with lock:
            print(f"[{tag}] REEVAL {exp}")
        t0 = time.time()
        with open(log_path, "a", encoding="utf-8") as log_f:
            rc = subprocess.run(
                [py, os.path.join("tools", "reeval_test.py"),
                 "--config", cfg, "--ckpt", "best", "--frames", "all"],
                cwd=REPO, env=env, stdout=log_f, stderr=subprocess.STDOUT,
            ).returncode
        dt = (time.time() - t0) / 60.0
        ok = rc == 0 and reeval_done(exp)
        with lock:
            status.append((exp, "ok" if ok else f"FAILED(rc={rc})", round(dt, 1)))
            print(f"[{tag}] {'DONE ' if ok else 'FAIL '} {exp} ({dt:.1f} min)"
                  + ("" if ok else f"  -> see {log_path}"))
        work_q.task_done()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--per-gpu", type=int, default=3,
                    help="concurrent re-eval processes per GPU")
    ap.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    ap.add_argument("--force", action="store_true",
                    help="remove existing reeval_*.json for design cells and recompute")
    ap.add_argument("--tables-out", default=os.path.join(REPO, "results_tables"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    py = sys.executable
    cells, missing = build_cells(args.modes)

    # ---- step 1: archive old values ----
    if not args.dry_run:
        dest = archive_old_tables(args.tables_out)
        if dest:
            print(f"[archive] moved old last-frame tables -> {os.path.relpath(dest, REPO)}")
        if args.force:
            n = 0
            for _, exp in cells:
                p = os.path.join(REPO, "results", f"reeval_{exp}.json")
                if os.path.exists(p):
                    os.remove(p)
                    n += 1
            print(f"[archive] removed {n} stale reeval JSONs (--force)")

    # ---- step 2: schedule re-evals ----
    todo = [(cfg, exp) for cfg, exp in cells if not reeval_done(exp)]
    skipped = len(cells) - len(todo)
    print(f"\nDesign: {len(cells)} cells | already re-evaluated: {skipped} | "
          f"to run: {len(todo)} on GPUs {args.gpus} x {args.per_gpu} slots")
    for cfg, exp in todo:
        print(f"  {exp}")
    if missing:
        print("\nMissing configs (run tools/gen_grid_configs.py --tiers):")
        for m in missing:
            print(f"  {m}")
    if args.dry_run:
        print("\n(dry run - nothing launched)")
        return

    status, lock = [], threading.Lock()
    if todo:
        work_q = queue.Queue()
        for item in todo:
            work_q.put(item)
        threads = [
            threading.Thread(target=worker, args=(g, s, work_q, status, lock, py),
                             daemon=True)
            for g in args.gpus for s in range(max(1, args.per_gpu))
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    failed = [e for e, r, _ in status if r != "ok"]

    # ---- step 3: regenerate the paper tables from corrected numbers ----
    print("\n[tables] regenerating from re-eval JSONs (all-frame protocol)")
    rc = subprocess.run(
        [py, os.path.join("tools", "gen_paper_tables.py"),
         "--source", "reeval", "--out", args.tables_out]
        + (["--mode", args.modes[0]] if len(args.modes) == 1 else []),
        cwd=REPO,
    ).returncode

    # ---- summary ----
    print(f"\n{'=' * 68}\nFINALIZE SUMMARY\n{'=' * 68}")
    print(f"  re-evaluated : {sum(1 for _, r, _ in status if r == 'ok')}")
    print(f"  reused       : {skipped}")
    print(f"  failed       : {len(failed)}" + (f"  -> {failed}" if failed else ""))
    print(f"  tables       : {'OK' if rc == 0 else 'FAILED'} -> "
          f"{os.path.relpath(args.tables_out, REPO)}/  (source: all-frame re-eval)")
    if failed:
        print("\n  Failed cells keep their row as '--' in the tables. Check "
              "logs/reeval/<exp>.log; a missing checkpoint (e.g. abl_nospatial) "
              "means that training cell itself needs a rerun first.")
    print("\n  Verify one number against a reeval JSON before finalizing, then "
          "compile: cd results_tables && pdflatex preview.tex")


if __name__ == "__main__":
    main()
