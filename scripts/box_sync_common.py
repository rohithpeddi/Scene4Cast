#!/usr/bin/env python3
"""
Shared config resolution, split resolver, and Box client utilities for Box sync scripts.

Centralizes the common components needed by Box sync scripts:
  1. loading the active config (--config, or sensible local defaults),
  2. resolving the Box CCG credentials file (box_cred.json),
  3. reading named ``box_sync.<purpose>`` entries ({folder_id, local_root}),
  4. constructing an authenticated Box CCG Client (get_box_client),
  5. Action Genome video split and mode resolution (VideoSplitResolver, STANDARD_SPLITS),
  6. providing the shared, robust BoxSyncService for file/folder transfers with chunking & retries.

Config schema:
    box_credentials_path: "<path>/box_cred.json"
    box_proxy: "<proxy_url>" (optional)
    box_sync:
      annotations:
        folder_id: "380446756239"
        local_root: "/data/rohith/ag"
      dynamic_scenes:
        folder_id: "380446756239"
        local_root: "/data2/rohith/ag/ag4D/dynamic_scenes"
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml
from boxsdk import CCGAuth, Client
from boxsdk.exception import BoxAPIException
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project root setup — make repository modules importable.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Optional integration with configs if present
try:
    from configs import load_base_config, set_config_path  # noqa: E402
    _HAS_CONFIGS_MODULE = True
except (ImportError, ModuleNotFoundError):
    _HAS_CONFIGS_MODULE = False

DEFAULT_CONFIG_CANDIDATES = [
    "configs/annotation_utd.yaml",
    "configs/anticipation/config_rohith.yaml",
]
CHUNK_UPLOAD_THRESHOLD_BYTES = 20 * 1024 * 1024  # 20 MB

# Credential fallbacks tried when neither --box-cred, $BOX_CRED_PATH, nor config is set.
_CRED_CANDIDATES = [
    "/data/rohith/box_cred.json",
    str(Path.home() / "box_cred.json"),
    r"E:\PRIVATE_KEY\box_cred.json",
    "/home/cse/visitor/ayushs.visitor/box_cred.json",
]

# Default fallbacks for Scene4Cast Box sync
DEFAULT_BOX_SYNC = {
    "annotations": {
        "folder_id": "380446756239",
        "local_root": "/data/rohith/ag",
    },
    "dynamic_scenes": {
        "folder_id": "380446756239",
        "local_root": "/data2/rohith/ag/ag4D/dynamic_scenes",
    },
}

# Action Genome Shards
STANDARD_SPLITS = {
    "04": set("01234"),
    "05": set("01234"),      # alias for 0-4
    "59": set("56789"),
    "09": set("56789"),      # alias for 5-9
    "AD": set("ABCD"),
    "EH": set("EFGH"),
    "IL": set("IJKL"),
    "MP": set("MNOP"),
    "QT": set("QRST"),
    "UZ": set("UVWXYZ"),
}


def human_size(nbytes: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(nbytes) < 1024.0:
            return f"{nbytes:3.1f} {unit}"
        nbytes /= 1024.0
    return f"{nbytes:.1f} PB"


def _first_existing(candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c and Path(c).exists():
            return str(c)
    return None


def load_config(config_path: Optional[str] = None) -> dict:
    """Set the active config (or default candidate / defaults) and return it.

    Also applies the config's ``box_proxy`` (if any) so Box is reachable on
    clusters that require an HTTP proxy — see :func:`apply_proxy`.
    """
    cfg: Dict[str, Any] = {}

    target_cfg_path = config_path or _first_existing(DEFAULT_CONFIG_CANDIDATES)

    if target_cfg_path and Path(target_cfg_path).exists():
        try:
            with open(target_cfg_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    cfg = loaded
        except Exception as e:
            print(f"Warning: Failed to load config from {target_cfg_path}: {e}")
    elif _HAS_CONFIGS_MODULE:
        try:
            set_config_path(config_path or DEFAULT_CONFIG_CANDIDATES[0])
            cfg = load_base_config()
        except Exception:
            cfg = {}

    # Merge default box_sync if missing
    if "box_sync" not in cfg or not cfg["box_sync"]:
        cfg["box_sync"] = DEFAULT_BOX_SYNC
    else:
        for k, v in DEFAULT_BOX_SYNC.items():
            if k not in cfg["box_sync"]:
                cfg["box_sync"][k] = v

    if "box_credentials_path" not in cfg or not cfg["box_credentials_path"]:
        cand = _first_existing(_CRED_CANDIDATES)
        if cand:
            cfg["box_credentials_path"] = cand

    apply_proxy(cfg)
    return cfg


def apply_proxy(cfg: dict) -> Optional[str]:
    """Export the config's ``box_proxy`` into the environment, if set."""
    proxy = cfg.get("box_proxy")
    if proxy:
        for var in ("https_proxy", "http_proxy", "HTTPS_PROXY", "HTTP_PROXY"):
            os.environ.setdefault(var, str(proxy))
    return (
        os.environ.get("https_proxy")
        or os.environ.get("HTTPS_PROXY")
        or (str(proxy) if proxy else None)
    )


def resolve_box_cred(cfg: Optional[dict] = None, cli: Optional[str] = None) -> Optional[str]:
    """Resolve the box_cred.json path.

    Priority: ``--box-cred`` > ``$BOX_CRED_PATH`` > config
    ``box_credentials_path`` (if it exists on disk) > the first existing
    known candidate.
    """
    if cli:
        return cli
    env = os.environ.get("BOX_CRED_PATH")
    if env and Path(env).exists():
        return env
    cfg_path = (cfg or {}).get("box_credentials_path")
    if cfg_path and Path(cfg_path).exists():
        return str(cfg_path)
    return str(cfg_path) if cfg_path else _first_existing(_CRED_CANDIDATES)


def get_box_client(cfg: Optional[dict] = None, cli_cred: Optional[str] = None) -> Client:
    """Initialize an authenticated Box Client using CCG credentials."""
    cred_path = resolve_box_cred(cfg, cli_cred)

    client_id = os.environ.get("BOX_CLIENT_ID")
    client_secret = os.environ.get("BOX_CLIENT_SECRET")
    user_id = os.environ.get("BOX_USER_ID")

    if cred_path and Path(cred_path).exists():
        try:
            with open(cred_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                client_id = client_id or data.get("client_id")
                client_secret = client_secret or data.get("client_secret")
                user_id = user_id or data.get("user_id")
        except Exception as e:
            print(f"Warning: Could not read credentials from {cred_path}: {e}")

    # Fallback to defaults if not found
    client_id = client_id or "krr2b0dmxvnqn83ikpe6ufs58jg9t82b"
    client_secret = client_secret or "TTsVwLrnv9EzmKJv67yrCyUM09wJSriK"
    user_id = user_id or "23441227496"

    if not all([client_id, client_secret, user_id]):
        print("ERROR: Missing Box credentials (client_id, client_secret, or user_id).", file=sys.stderr)
        sys.exit(1)

    ccg_auth = CCGAuth(client_id=client_id, client_secret=client_secret, user=user_id)
    return Client(ccg_auth)


def _entry(cfg: dict, purpose: str) -> dict:
    return dict((cfg.get("box_sync", {}) or {}).get(purpose, {}) or {})


def box_sync_folder_id(
    cfg: dict,
    purpose: str,
    cli: Optional[str] = None,
    fallback: Optional[str] = None,
) -> str:
    """Box folder id for *purpose*: ``cli`` > ``box_sync.<purpose>.folder_id`` > fallback."""
    if cli:
        return str(cli)
    fid = _entry(cfg, purpose).get("folder_id")
    if fid:
        return str(fid)
    if fallback:
        return str(fallback)
    return "380446756239"


def box_sync_local_root(
    cfg: dict,
    purpose: str,
    cli: Optional[str] = None,
    fallback: Optional[str] = None,
    key: str = "local_root",
) -> Path:
    """Local root for *purpose*: ``cli`` > ``box_sync.<purpose>.<key>`` > fallback."""
    if cli:
        return Path(cli)
    val = _entry(cfg, purpose).get(key)
    if val:
        return Path(val)
    if fallback:
        return Path(fallback)
    return Path("/data/rohith/ag")


# ---------------------------------------------------------------------------
# Action Genome Video Split & Mode Resolver
# ---------------------------------------------------------------------------

class VideoSplitResolver:
    """Resolves video_id -> mode ('train' or 'test') and shard split (e.g. '04', 'AD')."""

    def __init__(self, ag_data_path: str = "/data/rohith/ag"):
        self.ag_path = Path(ag_data_path)
        self.train_vids: Set[str] = set()
        self.test_vids: Set[str] = set()
        self.bbox_sets: Dict[str, str] = {}
        self._load_metadata()

    def _load_metadata(self):
        train_dir = self.ag_path / "world4d_rel_annotations" / "train"
        test_dir = self.ag_path / "world4d_rel_annotations" / "test"

        if train_dir.exists():
            self.train_vids = {p.name.split(".")[0] for p in train_dir.glob("*.pkl")}
        if test_dir.exists():
            self.test_vids = {p.name.split(".")[0] for p in test_dir.glob("*.pkl")}

        bbox_path = self.ag_path / "annotations" / "object_bbox_and_relationship.pkl"
        if bbox_path.exists():
            try:
                with open(bbox_path, "rb") as f:
                    obj_bbox = pickle.load(f)
                self.bbox_sets = {
                    k.split("/")[0].split(".")[0]: v[0]["metadata"]["set"]
                    for k, v in obj_bbox.items()
                    if v and "metadata" in v[0] and "set" in v[0]["metadata"]
                }
            except Exception as e:
                print(f"Warning: Could not load {bbox_path}: {e}")

    def get_mode(self, video_id: str) -> str:
        vid = video_id.split("_")[0].split(".")[0]
        if vid in self.train_vids:
            return "train"
        if vid in self.test_vids:
            return "test"
        return self.bbox_sets.get(vid, "unknown")

    def get_canonical_split(self, video_id: str) -> str:
        vid = video_id.split("_")[0].split(".")[0]
        if not vid:
            return "OTHER"
        c = vid[0].upper()
        if c in "01234":
            return "04"
        if c in "56789":
            return "59"
        if c in "ABCD":
            return "AD"
        if c in "EFGH":
            return "EH"
        if c in "IJKL":
            return "IL"
        if c in "MNOP":
            return "MP"
        if c in "QRST":
            return "QT"
        if c in "UVWXYZ":
            return "UZ"
        return "OTHER"

    def matches_split_filter(self, video_id: str, split_filter: Optional[Set[str]]) -> bool:
        if not split_filter:
            return True
        vid = video_id.split("_")[0].split(".")[0]
        if not vid:
            return False
        first_char = vid[0].upper()
        canon = self.get_canonical_split(vid)

        for s in split_filter:
            norm_s = s.strip().upper()
            if norm_s in ("ALL", "BOTH", "*"):
                return True
            if norm_s in STANDARD_SPLITS and first_char in STANDARD_SPLITS[norm_s]:
                return True
            if canon == norm_s:
                return True
            if vid.upper().startswith(norm_s):
                return True
        return False


# ---------------------------------------------------------------------------
# Reusable Box Synchronization Service
# ---------------------------------------------------------------------------

class BoxSyncService:
    """Core synchronization service between a local path and Box folder."""

    def __init__(
        self,
        client: Client,
        local_root: str | Path,
        box_root_id: str,
        mode: str = "upload",
        workers: int = 8,
        dry_run: bool = False,
        verbose: bool = False,
    ):
        self.client = client
        self.local_root = Path(local_root).resolve()
        self.box_root_id = str(box_root_id)
        self.mode = mode.lower()
        self.workers = max(1, workers)
        self.dry_run = dry_run
        self.verbose = verbose
        self._folder_cache: Dict[str, Dict[str, str]] = {}

    def _with_retries(self, func, *args, max_attempts: int = 5, **kwargs):
        """Execute with exponential backoff on Box rate limit or connection error."""
        attempt = 0
        while True:
            try:
                return func(*args, **kwargs)
            except BoxAPIException as e:
                attempt += 1
                if attempt >= max_attempts:
                    raise
                retry_after = getattr(e, "retry_after", None)
                sleep_time = float(retry_after) if retry_after else min(60.0, 1.5 ** attempt)
                if self.verbose:
                    print(f"  [Box API {e.status}] retry in {sleep_time:.1f}s")
                time.sleep(sleep_time)
            except Exception as e:
                attempt += 1
                if attempt >= max_attempts:
                    raise
                sleep_time = min(30.0, 1.5 ** attempt)
                if self.verbose:
                    print(f"  [Network Warning {e}] retry in {sleep_time:.1f}s")
                time.sleep(sleep_time)

    def _get_folder_children(self, folder_id: str) -> Dict[str, str]:
        if folder_id in self._folder_cache:
            return self._folder_cache[folder_id]

        children: Dict[str, str] = {}
        try:
            items = self._with_retries(
                lambda: list(self.client.folder(folder_id).get_items(fields=["id", "name", "type"]))
            )
            for it in items:
                if it.type == "folder":
                    children[it.name] = it.id
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not list folder {folder_id}: {e}")

        self._folder_cache[folder_id] = children
        return children

    def find_box_path(self, root_id: str, rel_dir: str) -> Optional[str]:
        """Look up subfolder path under Box root_id without creating it if missing."""
        rel_dir = Path(rel_dir).as_posix()
        if rel_dir in ("", "."):
            return root_id

        current = root_id
        for part in Path(rel_dir).parts:
            children = self._get_folder_children(current)
            if part not in children:
                return None
            current = children[part]
        return current

    def ensure_box_path(self, root_id: str, rel_dir: str) -> str:
        """Ensure rel_dir subfolder path exists under Box root_id and return folder_id."""
        rel_dir = Path(rel_dir).as_posix()
        if rel_dir in ("", "."):
            return root_id

        current = root_id
        for part in Path(rel_dir).parts:
            children = self._get_folder_children(current)
            if part in children:
                current = children[part]
                continue
            if self.dry_run:
                created_id = f"DRYRUN_{part}"
            else:
                try:
                    sub = self._with_retries(self.client.folder(current).create_subfolder, part)
                    created_id = sub.id
                except BoxAPIException as e:
                    if e.status == 409 and e.context_info and "conflicts" in e.context_info:
                        created_id = e.context_info["conflicts"]["id"]
                    else:
                        raise
            children[part] = created_id
            current = created_id
        return current

    def get_local_files(self, target_rel_path: str = "") -> Dict[str, int]:
        full_target = (self.local_root / target_rel_path).resolve()
        if not full_target.exists():
            raise FileNotFoundError(f"Local path does not exist: {full_target}")

        local_files: Dict[str, int] = {}
        if full_target.is_file():
            rel = full_target.relative_to(self.local_root).as_posix()
            local_files[rel] = full_target.stat().st_size
            return local_files

        for root, _, files in os.walk(full_target):
            for f in files:
                abs_f = Path(root) / f
                rel_f = abs_f.relative_to(self.local_root).as_posix()
                try:
                    local_files[rel_f] = abs_f.stat().st_size
                except OSError:
                    pass
        return local_files

    def get_box_files(self, folder_id: str, current_rel_prefix: str = "") -> Dict[str, Tuple[int, str]]:
        box_files: Dict[str, Tuple[int, str]] = {}
        stack = [(folder_id, current_rel_prefix)]

        while stack:
            fid, parent = stack.pop()
            try:
                items = self._with_retries(
                    lambda: list(self.client.folder(fid).get_items(limit=1000, fields=["id", "name", "type", "size"]))
                )
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Failed to fetch items for folder {fid}: {e}")
                continue

            for it in items:
                rel = f"{parent}/{it.name}" if parent else it.name
                if it.type == "file":
                    box_files[rel] = (it.size or 0, it.id)
                elif it.type == "folder":
                    stack.append((it.id, rel))
                    if fid not in self._folder_cache:
                        self._folder_cache[fid] = {}
                    self._folder_cache[fid][it.name] = it.id

        return box_files

    def upload_file(
        self,
        rel_path: str,
        file_size: int,
        target_box_root_id: str,
        local_file_path: Optional[Path] = None,
    ):
        local_path = local_file_path or (self.local_root / rel_path)
        dest_rel_dir = Path(rel_path).parent.as_posix()
        dest_folder_id = self.ensure_box_path(target_box_root_id, dest_rel_dir)

        if self.dry_run:
            return

        box_folder = self.client.folder(dest_folder_id)
        if file_size > CHUNK_UPLOAD_THRESHOLD_BYTES and hasattr(box_folder, "get_chunked_uploader"):
            def _chunked():
                try:
                    uploader = box_folder.get_chunked_uploader(str(local_path))
                    return uploader.start()
                except BoxAPIException as e:
                    if e.status == 409 and e.context_info and "conflicts" in e.context_info:
                        conflict_id = e.context_info["conflicts"]["id"]
                        return self.client.file(conflict_id).get_chunked_uploader(str(local_path)).start()
                    raise
            self._with_retries(_chunked)
        else:
            def _simple():
                try:
                    return box_folder.upload(str(local_path), file_name=local_path.name)
                except BoxAPIException as e:
                    if e.status == 409 and e.context_info and "conflicts" in e.context_info:
                        conflict_id = e.context_info["conflicts"]["id"]
                        return self.client.file(conflict_id).update_contents(str(local_path))
                    raise
            self._with_retries(_simple)

    def download_file(self, rel_path: str, file_id: str, dest_file_path: Optional[Path] = None):
        dest_path = dest_file_path or (self.local_root / rel_path)
        if self.dry_run:
            return

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = dest_path.with_suffix(dest_path.suffix + ".part")

        def _do_download():
            with open(temp_path, "wb") as f:
                self.client.file(file_id).download_to(f)
            os.replace(temp_path, dest_path)

        self._with_retries(_do_download)

    def sync(self, target_rel_path: str = ""):
        """Synchronize between local_root and Box folder."""
        print(f"\n==================================================================")
        print(f" Box Sync: Local [{self.local_root}] <-> Box Folder [{self.box_root_id}]")
        print(f" Target path : '{target_rel_path or '.'}'")
        print(f" Mode        : {self.mode.upper()}")
        print(f" Workers     : {self.workers}")
        print(f" Dry Run     : {self.dry_run}")
        print(f"==================================================================\n")

        local_files: Dict[str, int] = {}
        if self.mode in ("upload", "sync"):
            print(f"Scanning local files for '{target_rel_path or '.'}'...")
            local_files = self.get_local_files(target_rel_path)
            print(f"Found {len(local_files):,} local file(s), total: {human_size(sum(local_files.values()))}")

        # Targeted Box scanning
        box_files: Dict[str, Tuple[int, str]] = {}
        norm_target = Path(target_rel_path).as_posix() if target_rel_path else ""

        if norm_target and norm_target != ".":
            # Check if target is a subfolder or inside a subfolder
            local_target = (self.local_root / norm_target).resolve()
            search_box_dir = norm_target if local_target.is_dir() else Path(norm_target).parent.as_posix()
            target_sub_id = self.find_box_path(self.box_root_id, search_box_dir)

            if target_sub_id:
                print(f"Scanning Box destination subfolder '{search_box_dir}'...")
                sub_files = self.get_box_files(target_sub_id, current_rel_prefix=search_box_dir if search_box_dir != "." else "")
                for k, v in sub_files.items():
                    if k == norm_target or k.startswith(norm_target + "/"):
                        box_files[k] = v
            else:
                print(f"Target path '{search_box_dir}' not found on Box. (0 existing remote files)")
        else:
            print(f"Scanning Box folder {self.box_root_id}...")
            box_files = self.get_box_files(self.box_root_id)

        print(f"Found {len(box_files):,} matching Box file(s), total: {human_size(sum(s for s, _ in box_files.values()))}")

        to_upload: List[Tuple[str, int]] = []
        if self.mode in ("upload", "sync"):
            for rel, size in local_files.items():
                if rel not in box_files or box_files[rel][0] != size:
                    to_upload.append((rel, size))

        to_download: List[Tuple[str, int, str]] = []
        if self.mode in ("download", "sync"):
            for rel, (size, fid) in box_files.items():
                local_path = self.local_root / rel
                if not local_path.exists() or local_path.stat().st_size != size:
                    to_download.append((rel, size, fid))

        print("\n------------------------- PLAN -------------------------")
        if self.mode in ("upload", "sync"):
            print(f"To Upload  : {len(to_upload):,} file(s) ({human_size(sum(s for _, s in to_upload))})")
        if self.mode in ("download", "sync"):
            print(f"To Download: {len(to_download):,} file(s) ({human_size(sum(s for _, s, _ in to_download))})")
        print("--------------------------------------------------------\n")

        if self.dry_run:
            print("[Dry Run] No files transferred. Exiting.")
            return

        if to_upload and self.mode in ("upload", "sync"):
            print(f"Starting upload of {len(to_upload):,} file(s)...")
            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                futures = {
                    ex.submit(self.upload_file, rel, size, self.box_root_id): rel
                    for rel, size in to_upload
                }
                with tqdm(total=len(to_upload), unit="file", desc="Uploading") as pbar:
                    for fut in as_completed(futures):
                        rel = futures[fut]
                        try:
                            fut.result()
                        except Exception as e:
                            print(f"\n[Error uploading {rel}]: {e}")
                        pbar.update(1)

        if to_download and self.mode in ("download", "sync"):
            print(f"Starting download of {len(to_download):,} file(s)...")
            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                futures = {
                    ex.submit(self.download_file, rel, fid): rel
                    for rel, _, fid in to_download
                }
                with tqdm(total=len(to_download), unit="file", desc="Downloading") as pbar:
                    for fut in as_completed(futures):
                        rel = futures[fut]
                        try:
                            fut.result()
                        except Exception as e:
                            print(f"\n[Error downloading {rel}]: {e}")
                        pbar.update(1)

        print("\n✓ Synchronization complete.")

    def upload_file_map(
        self,
        file_entries: List[Tuple[Path, str, int]],
        target_box_root_id: Optional[str] = None,
    ):
        """Upload an explicit list of (local_abs_path, rel_box_path, file_size) tuples."""
        box_root = str(target_box_root_id or self.box_root_id)
        total_bytes = sum(s for _, _, s in file_entries)

        print(f"\nChecking destination on Box folder [{box_root}]...")
        # Inspect existing Box files
        # Check target subdirs from the file entries
        target_subdirs = {Path(rel).parts[0] for _, rel, _ in file_entries if len(Path(rel).parts) > 1}
        box_existing: Dict[str, int] = {}

        if target_subdirs:
            for sdir in target_subdirs:
                s_id = self.find_box_path(box_root, sdir)
                if s_id:
                    sub_files = self.get_box_files(s_id, current_rel_prefix=sdir)
                    for k, (sz, _) in sub_files.items():
                        box_existing[k] = sz
        else:
            sub_files = self.get_box_files(box_root)
            for k, (sz, _) in sub_files.items():
                box_existing[k] = sz

        to_upload = [
            (lp, rel, size)
            for lp, rel, size in file_entries
            if rel not in box_existing or box_existing[rel] != size
        ]

        print("\n------------------------- PLAN -------------------------")
        print(f"Total files in selection : {len(file_entries):,} ({human_size(total_bytes)})")
        print(f"Files already on Box     : {len(file_entries) - len(to_upload):,}")
        print(f"Files to upload          : {len(to_upload):,} ({human_size(sum(s for _, _, s in to_upload))})")
        print("--------------------------------------------------------\n")

        if self.dry_run:
            print("[Dry Run] Planning complete. No files transferred.")
            return

        if not to_upload:
            print("All matched files are already synced to Box. Done.")
            return

        print(f"Starting parallel upload with {self.workers} worker(s)...")
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            futures = {
                ex.submit(self.upload_file, rel, size, box_root, lp): rel
                for lp, rel, size in to_upload
            }
            with tqdm(total=len(to_upload), unit="file", desc="Uploading") as pbar:
                for fut in as_completed(futures):
                    rel = futures[fut]
                    try:
                        fut.result()
                    except Exception as e:
                        print(f"\n[Error uploading {rel}]: {e}")
                    pbar.update(1)

        print("\n✓ Upload complete.")


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Box Synchronization Service")
    parser.add_argument("--purpose", default="annotations", choices=["annotations", "dynamic_scenes"],
                        help="Box sync configuration purpose (default: annotations)")
    parser.add_argument("--source", "-s", default="annotations",
                        help="Local relative subfolder or single file to sync (default: annotations)")
    parser.add_argument("--mode", "-m", choices=["upload", "download", "sync"], default="upload",
                        help="Transfer mode: upload, download, or sync (default: upload)")
    parser.add_argument("--folder-id", help="Box destination folder ID (default: from config/fallback)")
    parser.add_argument("--local-root", help="Local directory root (default: from config/fallback)")
    parser.add_argument("--config", help="Path to YAML config")
    parser.add_argument("--box-cred", help="Path to box_cred.json")
    parser.add_argument("--workers", type=int, default=8, help="Parallel transfer workers")
    parser.add_argument("--dry-run", action="store_true", help="Inspect planned operations without transferring files")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()
    cfg = load_config(args.config)
    fid = box_sync_folder_id(cfg, args.purpose, args.folder_id)
    lroot = box_sync_local_root(cfg, args.purpose, args.local_root)
    client = get_box_client(cfg, args.box_cred)

    service = BoxSyncService(
        client=client,
        local_root=lroot,
        box_root_id=fid,
        mode=args.mode,
        workers=args.workers,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )
    service.sync(target_rel_path=args.source)


if __name__ == "__main__":
    main()