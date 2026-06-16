#!/usr/bin/env python3
"""
Annotation Pipeline Orchestrator
==================================

Runs the complete corrected annotation pipeline for the test dataset
using configuration from configs/annotation_utd.yaml.

Pipeline stages:
  1. Download manual corrections from Firebase
  2. Generate corrected world-frame 3D bboxes
  3. Generate corrected frame-level 3D bboxes (FINAL and CAMERA)
  4. Generate corrected 4D bboxes (with object permanence filling)
  5. Augment relationships with manual corrections
  6. Merge relationships with 4D bboxes
  7. Generate final unified scene graphs

Usage:
    python run_annotation_pipeline.py --config configs/annotation_utd.yaml [--stages STAGE_NUM]
    python run_annotation_pipeline.py --config configs/annotation_utd.yaml --stage 1
    python run_annotation_pipeline.py --config configs/annotation_utd.yaml --stages 1,2,3
    python run_annotation_pipeline.py --config configs/annotation_utd.yaml --stages 5-7
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional
import yaml


class AnnotationPipeline:
    """Orchestrates the annotation pipeline stages."""

    STAGES = {
        1: {
            "name": "Download Manual Corrections",
            "script": "datasets/preprocess/annotations/download_floor_manual_corrections.py",
            "args_template": [
                "--output-dir {manual_corrections[output_dir]}",
            ],
        },
        2: {
            "name": "Generate Corrected World BBoxes",
            "script": "datasets/preprocess/annotations/corrected_world_bbox_generator.py",
            "args_template": [
                "--ag_root_directory {ag_root_directory}",
                "--phase {phase}",
                "--corrections-only",
            ],
        },
        3: {
            "name": "Generate Corrected Frame BBoxes (FINAL + CAMERA)",
            "script": "datasets/preprocess/annotations/corrected_frame_bbox_generator.py",
            "args_template": [
                "--ag_root_directory {ag_root_directory}",
                "--dynamic_scene_dir_path {dynamic_scene_dir_path}",
                "--phase {phase}",
                "--mode both",
                "--corrections-only",
            ],
        },
        4: {
            "name": "Generate Corrected 4D BBoxes",
            "script": "datasets/preprocess/annotations/corrected_4d_bbox_generator.py",
            "args_template": [
                "--ag_root_directory {ag_root_directory}",
                "--dynamic_scene_dir_path {dynamic_scene_dir_path}",
                "--phase {phase}",
                "--corrections-only",
            ],
        },
        5: {
            "name": "Augment Relationships",
            "script": "datasets/preprocess/annotations/augment_relationships_test.py",
            "args_template": [
                "--ag_root_directory {ag_root_directory}",
                "--output_dir {augment_relationships[output_dir]}",
                "--corrections_dir {manual_corrections[output_dir]}",
                "--phase {phase}",
            ],
        },
        6: {
            "name": "Merge Relationships + 4D BBoxes",
            "script": "datasets/preprocess/annotations/combine_world4d_relationships_test.py",
            "args_template": [
                "--ag_root_directory {ag_root_directory}",
            ],
        },
        7: {
            "name": "Generate Final Scene Graphs",
            "script": "datasets/preprocess/annotations/world_scene_graph_generator.py",
            "args_template": [
                "--ag_root_directory {ag_root_directory}",
                "--corrections-only",
            ],
        },
    }

    def __init__(self, config_path: str, project_root: Optional[str] = None):
        """
        Initialize pipeline with config file.

        Args:
            config_path: Path to annotation_utd.yaml config file
            project_root: Root directory of the project (auto-detect if None)
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)

        # Auto-detect project root
        if project_root is None:
            self.project_root = self._find_project_root()
        else:
            self.project_root = Path(project_root)

        print(f"[Pipeline] Project root: {self.project_root}")
        print(f"[Pipeline] Config: {self.config_path}")

    @staticmethod
    def _find_project_root() -> Path:
        """Find project root by locating configs/ directory."""
        current = Path.cwd()
        while current != current.parent:
            if (current / "configs" / "annotation_utd.yaml").exists():
                return current
            current = current.parent
        # Fallback to current directory
        return Path.cwd()

    def _format_args(self, stage_num: int) -> List[str]:
        """Format command-line arguments for a stage using config values."""
        template = self.STAGES[stage_num]["args_template"]
        args = []

        for arg in template:
            try:
                formatted_arg = arg.format(**self.config)
                args.append(formatted_arg)
            except KeyError as e:
                print(f"[Warning] Could not format argument: {arg}, missing key: {e}")

        return args

    def run_stage(
        self,
        stage_num: int,
        overwrite: bool = False,
        dry_run: bool = False,
    ) -> bool:
        """
        Run a single pipeline stage.

        Args:
            stage_num: Stage number (1-7)
            overwrite: If True, add --overwrite flag
            dry_run: If True, print command without executing

        Returns:
            True if stage succeeded, False otherwise
        """
        if stage_num not in self.STAGES:
            print(f"[Error] Invalid stage number: {stage_num}")
            return False

        stage_info = self.STAGES[stage_num]
        script_path = self.project_root / stage_info["script"]

        if not script_path.exists():
            print(f"[Error] Script not found: {script_path}")
            return False

        # Build command
        cmd = [
            sys.executable,
            str(script_path),
        ]
        cmd.extend(self._format_args(stage_num))

        if overwrite:
            cmd.append("--overwrite")

        print(f"\n{'='*70}")
        print(f"Stage {stage_num}: {stage_info['name']}")
        print(f"{'='*70}")
        print(f"Command: {' '.join(cmd)}")
        print()

        if dry_run:
            print("[Dry Run] Command NOT executed")
            return True

        try:
            result = subprocess.run(cmd, cwd=self.project_root, check=False)
            success = result.returncode == 0
            if success:
                print(f"\n✓ Stage {stage_num} completed successfully")
            else:
                print(f"\n✗ Stage {stage_num} failed with exit code {result.returncode}")
            return success
        except Exception as e:
            print(f"\n✗ Stage {stage_num} failed with exception: {e}")
            return False

    def run_stages(
        self,
        stages: List[int],
        overwrite: bool = False,
        dry_run: bool = False,
        continue_on_error: bool = False,
    ) -> Dict[int, bool]:
        """
        Run multiple pipeline stages in sequence.

        Args:
            stages: List of stage numbers to run
            overwrite: If True, add --overwrite flag to all stages
            dry_run: If True, print commands without executing
            continue_on_error: If True, continue even if a stage fails

        Returns:
            Dict mapping stage number to success status
        """
        results = {}

        print(f"\n{'='*70}")
        print(f"Annotation Pipeline: {len(stages)} stage(s)")
        print(f"{'='*70}")
        print(f"Stages: {stages}")
        print(f"Overwrite: {overwrite}")
        print(f"Dry Run: {dry_run}")
        print(f"Continue on Error: {continue_on_error}")
        print()

        for stage_num in stages:
            success = self.run_stage(stage_num, overwrite=overwrite, dry_run=dry_run)
            results[stage_num] = success

            if not success and not continue_on_error:
                print(f"\n[Error] Stage {stage_num} failed. Stopping pipeline.")
                break

        # Summary
        print(f"\n{'='*70}")
        print("Pipeline Summary")
        print(f"{'='*70}")
        for stage_num in stages:
            status = "✓ PASS" if results[stage_num] else "✗ FAIL"
            print(f"Stage {stage_num}: {status}")

        all_passed = all(results.values())
        print(f"\nOverall: {'✓ All stages passed' if all_passed else '✗ Some stages failed'}")

        return results

    @staticmethod
    def parse_stage_spec(spec: str) -> List[int]:
        """
        Parse stage specification string.

        Examples:
            "1" → [1]
            "1,3,5" → [1, 3, 5]
            "1-3" → [1, 2, 3]
            "1,3-5,7" → [1, 3, 4, 5, 7]
        """
        stages = set()

        for part in spec.split(","):
            part = part.strip()
            if "-" in part:
                start, end = map(int, part.split("-"))
                stages.update(range(start, end + 1))
            else:
                stages.add(int(part))

        return sorted(list(stages))


def main():
    parser = argparse.ArgumentParser(
        description="Run the annotation pipeline using centralized config"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/annotation_utd.yaml",
        help="Path to annotation config file",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="1-7",
        help="Stages to run (e.g., '1', '1-3', '1,3,5', '5-7')",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Add --overwrite flag to all stages",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running stages even if one fails",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Project root directory (auto-detected if not provided)",
    )
    args = parser.parse_args()

    try:
        pipeline = AnnotationPipeline(args.config, args.project_root)
        stages = AnnotationPipeline.parse_stage_spec(args.stages)

        pipeline.run_stages(
            stages,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            continue_on_error=args.continue_on_error,
        )

    except Exception as e:
        print(f"[Error] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
