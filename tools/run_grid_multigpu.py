"""
Multi-GPU launcher for the full WorldSGG campaign.

Distributes every training cell over the available GPUs with a shared work
queue: one worker per GPU, each pinning its runs via CUDA_VISIBLE_DEVICES, so
all GPUs stay busy until the queue drains (no static per-GPU split that leaves
a GPU idle when its share finishes early).

Campaign structure (July 2026): baselines run ONLY at resnet50 (the common
backbone for the method-comparison table); WorldWise runs at all 4 backbones
(the scaling story); WorldWise⁺ tiers run at the hero backbone (dinov3l).

Queue order (highest priority first — the decision-critical cells finish first):
  1. table_a  — 5 methods @ resnet50, both modes         (method comparison, 10)
  2. scaling  — worldwise @ dinov2b/dinov2l/dinov3l      (Table B + tier I-0 ref, 6)
  3. stage_a  — worldwise plus1/plus2/plus3 + noema @ dinov3l  (plugin ladder, 8)
  4. stage_b  — conf/proto/xobj/energy @ dinov3l          (research plugins, 8)
     (interpret AFTER the Stage-A gate — they run last so you can stop early)

Each run's stdout/stderr goes to logs/grid/<experiment>.log; a status line is
appended to results/grid_run_status.csv as runs finish. Already-completed runs
(metrics jsonl contains the final epoch) are skipped, so the script is safe to
re-run after interruptions.

Usage (on the training server):
    # everything: base grid + all tiers, priors computed first (32 runs)
    python tools/run_grid_multigpu.py --gpus 0 1 2 --compute-priors

    # only the decision-critical cells (method table + scaling + Stage A)
    python tools/run_grid_multigpu.py --gpus 0 1 2 --stages table_a scaling stage_a

    # see the schedule without launching
    python tools/run_grid_multigpu.py --gpus 0 1 2 --dry-run
"""

import argparse
import csv
import os
import queue
import re
import subprocess
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

METHODS = ["w_sttran", "w_sttran_pp", "w_dsgdetr", "w_dsgdetr_pp", "worldwise"]
BACKBONES = ["resnet50", "dinov2b", "dinov2l", "dinov3l"]
MODES = ["predcls", "sgdet"]
HERO = "dinov3l"                 # WorldWise⁺ tiers run here
METHODS_BACKBONE = "resnet50"    # all 5 methods compared here
SCALING_BACKBONES = ["dinov2b", "dinov2l", "dinov3l"]  # worldwise-only extras
STAGE_A_TIERS = ["plus1", "plus2", "plus3", "noema"]
STAGE_B_TIERS = ["conf", "proto", "xobj", "energy"]
STAGES = ["table_a", "scaling", "stage_a", "stage_b"]


def read_config_fields(cfg_path):
    """Extract experiment_name and nepoch from a generated YAML (regex —
    avoids a yaml dependency so --dry-run works on any machine)."""
    exp, nepoch = None, None
    with open(cfg_path, "r", encoding="utf-8") as f:
        for line in f:
            m = re.match(r"^experiment_name:\s*(\S+)", line)
            if m:
                exp = m.group(1)
            m = re.match(r"^nepoch:\s*(\d+)", line)
            if m:
                nepoch = int(m.group(1))
    return exp, nepoch


def is_completed(exp, nepoch):
    """True if the metrics jsonl already contains the final epoch's row."""
    if exp is None or nepoch is None:
        return False
    path = os.path.join(REPO, "results", f"{exp}_metrics.jsonl")
    if not os.path.exists(path):
        return False
    needle = f'"epoch": {nepoch}'
    with open(path, "r", encoding="utf-8") as f:
        return any(needle in line for line in f)


def build_schedule(args):
    """Ordered list of (stage, stem, mode, cfg_relpath). Stems are the config
    file basenames without .yaml, e.g. 'worldwise_plus1_predcls_dinov3l'."""
    cells = []

    def add(stage, stem, mode):
        cfg = os.path.join("configs", "methods", mode, f"{stem}.yaml")
        cells.append((stage, stem, mode, cfg))

    # 1. Method comparison — all 5 methods at the common backbone
    if "table_a" in args.stages:
        for mode in args.modes:
            for m in METHODS:
                add("table_a", f"{m}_{mode}_{METHODS_BACKBONE}", mode)

    # 2. WorldWise backbone scaling (also produces the tier I-0 reference @ hero)
    if "scaling" in args.stages:
        for mode in args.modes:
            for b in SCALING_BACKBONES:
                add("scaling", f"worldwise_{mode}_{b}", mode)

    # 3. Stage A plugin ladder @ hero backbone
    if "stage_a" in args.stages:
        for mode in args.modes:
            for t in STAGE_A_TIERS:
                add("stage_a", f"worldwise_{t}_{mode}_{HERO}", mode)

    # 4. Stage B research plugins @ hero backbone (gate on Stage A first)
    if "stage_b" in args.stages:
        for mode in args.modes:
            for t in STAGE_B_TIERS:
                add("stage_b", f"worldwise_{t}_{mode}_{HERO}", mode)

    return cells


def worker(gpu_id, work_q, status, lock, py, dry_run):
    while True:
        try:
            item = work_q.get_nowait()
        except queue.Empty:
            return
        stage, stem, mode, cfg, exp = item

        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_id))
        log_dir = os.path.join(REPO, "logs", "grid")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{exp or stem}.log")

        with lock:
            print(f"[gpu{gpu_id}] START {stage:8s} {stem}  → {log_path}")

        t0 = time.time()
        if dry_run:
            rc = 0
        else:
            with open(log_path, "a", encoding="utf-8") as log_f:
                log_f.write(f"\n===== launch {time.strftime('%Y-%m-%d %H:%M:%S')} "
                            f"gpu={gpu_id} cfg={cfg} =====\n")
                log_f.flush()
                rc = subprocess.run(
                    [py, "train_wsgg_methods.py", "--config", cfg],
                    cwd=REPO, env=env, stdout=log_f, stderr=subprocess.STDOUT,
                ).returncode
        dt = (time.time() - t0) / 60.0

        result = "ok" if rc == 0 else f"FAILED(rc={rc})"
        with lock:
            status.append((stage, stem, mode, gpu_id, result, round(dt, 1)))
            print(f"[gpu{gpu_id}] DONE  {stage:8s} {stem}  {result} in {dt:.1f} min")
            _write_status_csv(status)
        work_q.task_done()


def _write_status_csv(status):
    path = os.path.join(REPO, "results", "grid_run_status.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stage", "experiment_stem", "mode", "gpu", "status", "minutes"])
        for row in status:
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2],
                    help="GPU ids to use (one worker per id)")
    ap.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES,
                    help="which campaign stages to schedule")
    ap.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    ap.add_argument("--data_path", default="/data/rohith/ag")
    ap.add_argument("--compute-priors", action="store_true",
                    help="run tools/compute_predicate_priors.py once first")
    ap.add_argument("--priors-feature-model", default="dinov2b")
    ap.add_argument("--no-skip-completed", action="store_true",
                    help="re-run cells even if their final epoch is logged")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    py = sys.executable

    # Priors are shared by every WorldWise cell — compute once, up front.
    if args.compute_priors and not args.dry_run:
        for mode in args.modes:
            print(f"=== computing predicate priors ({mode}) ===")
            subprocess.run([
                py, os.path.join("tools", "compute_predicate_priors.py"),
                "--data_path", args.data_path,
                "--feature_model", args.priors_feature_model,
                "--mode", mode,
            ], cwd=REPO, check=False)

    # Build the prioritized schedule
    schedule = []
    skipped, missing = [], []
    for stage, stem, mode, cfg in build_schedule(args):
        cfg_abs = os.path.join(REPO, cfg)
        if not os.path.exists(cfg_abs):
            missing.append(cfg)
            continue
        exp, nepoch = read_config_fields(cfg_abs)
        if not args.no_skip_completed and is_completed(exp, nepoch):
            skipped.append(stem)
            continue
        schedule.append((stage, stem, mode, cfg, exp))

    print(f"\nSchedule: {len(schedule)} runs on GPUs {args.gpus} "
          f"({len(skipped)} already complete, {len(missing)} configs missing)")
    for stage, stem, mode, cfg, exp in schedule:
        print(f"  {stage:8s} | {stem}")
    if missing:
        print("\nMissing configs (run `python tools/gen_grid_configs.py --tiers`):")
        for cfg in missing:
            print(f"  {cfg}")
    if args.dry_run and not schedule:
        return
    if args.dry_run:
        print("\n(dry run — nothing launched)")
        return
    if not schedule:
        print("Nothing to do.")
        return

    # Shared queue, one worker thread per GPU
    work_q = queue.Queue()
    for item in schedule:
        work_q.put(item)

    status, lock = [], threading.Lock()
    threads = [
        threading.Thread(target=worker, args=(g, work_q, status, lock, py, False),
                         daemon=True)
        for g in args.gpus
    ]
    t_start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Summary
    total_h = (time.time() - t_start) / 3600.0
    print(f"\n{'=' * 72}\nCAMPAIGN SUMMARY  ({total_h:.1f} h wall-clock)\n{'=' * 72}")
    for stage, stem, mode, gpu, result, mins in status:
        print(f"  {result:14s} | gpu{gpu} | {stage:8s} | {stem} ({mins} min)")
    n_ok = sum(1 for *_, r, _m in status if r == "ok")
    print(f"\n{n_ok}/{len(status)} succeeded. Status CSV → results/grid_run_status.csv")
    print("Aggregate: python tools/aggregate_results.py --mode predcls --tiers "
          "&& python tools/aggregate_results.py --mode sgdet --tiers")


if __name__ == "__main__":
    main()
