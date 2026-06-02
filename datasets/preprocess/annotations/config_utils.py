"""Shared config helpers for annotation preprocessing scripts.

Single source of truth for:
  - loading the YAML config (``load_config``),
  - the project-wide default paths (constants below),
  - the CLI > config > default resolution used by every script
    (``first``, ``resolve_common``),
  - the standard ``--config`` argparse flag (``add_config_arg``).

Kept intentionally lightweight (only ``pathlib`` + ``yaml``) so that even the
standalone scripts can import it without pulling in cv2/torch/scipy.
"""

from pathlib import Path
from typing import Any, Dict

import yaml

# --- Project-wide default paths (single source of truth) ---
DEFAULT_AG_ROOT = "/data/rohith/ag"
DEFAULT_DYNAMIC_SCENE_DIR = "/data3/rohith/ag/ag4D/dynamic_scenes/pi3_dynamic"
DEFAULT_CONFIG_PATH = "configs/annotation_utd.yaml"


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML config file; return ``{}`` if it doesn't exist."""
    config_path = Path(config_path)
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def first(*values, default=None):
    """Return the first value that is not ``None``, else ``default``.

    Correct for ``None`` specifically, unlike ``or`` which also swallows
    falsy-but-valid values (``''``, ``0``, ``False``).
    """
    for v in values:
        if v is not None:
            return v
    return default


def add_config_arg(parser, default: str = DEFAULT_CONFIG_PATH) -> None:
    """Add the standard ``--config`` argument to an argparse parser."""
    parser.add_argument(
        "--config", type=str, default=default,
        help=f"Path to config file (default: {default})",
    )


def resolve_common(args, config: Dict[str, Any], *, default_phase=None):
    """Resolve the 3 settings shared by every script (CLI > config > default).

    Returns ``(ag_root_directory, dynamic_scene_dir_path, phase)``.
    Missing argparse attributes are treated as ``None``.
    """
    ag_root = first(
        getattr(args, "ag_root_directory", None),
        config.get("ag_root_directory"),
        default=DEFAULT_AG_ROOT,
    )
    dynamic_scene = first(
        getattr(args, "dynamic_scene_dir_path", None),
        config.get("dynamic_scene_dir_path"),
        default=DEFAULT_DYNAMIC_SCENE_DIR,
    )
    phase = first(
        getattr(args, "phase", None),
        config.get("phase"),
        default=default_phase,
    )
    return ag_root, dynamic_scene, phase
