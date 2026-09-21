"""
Model asset manager and downloader for ExoCortex.

Manages downloading, caching, and integrity verification of local SLM assets
(Qwen2.5-0.5B-Instruct ONNX model & tokenizer) stored in ~/.exocortex/models/.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from exocortex.config import get_config

logger = logging.getLogger("exocortex.slm.model_manager")

DEFAULT_MODEL_REPO = "onnx-community/Qwen2.5-0.5B-Instruct"
DEFAULT_MODEL_NAME = "qwen2.5-0.5b-instruct"

MODEL_FILES = {
    "tokenizer": "tokenizer.json",
    "config": "config.json",
    "model_onnx": "onnx/model_int8.onnx",
}


@dataclass
class ModelStatus:
    """Status of local model assets."""
    model_name: str
    model_dir: Path
    is_ready: bool
    total_size_mb: float
    files_present: Dict[str, bool]
    file_paths: Dict[str, Path]
    missing_files: List[str]


def get_model_directory(model_name: str = DEFAULT_MODEL_NAME) -> Path:
    """Return the application directory path for storing model weights."""
    config = get_config()
    base_dir = config.ensure_data_dir()
    model_dir = base_dir / "models" / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def check_model_status(model_name: str = DEFAULT_MODEL_NAME) -> ModelStatus:
    """Inspect local disk to check if required model and tokenizer files are present."""
    model_dir = get_model_directory(model_name)
    files_present: Dict[str, bool] = {}
    file_paths: Dict[str, Path] = {}
    missing: List[str] = []
    total_size = 0

    for key, rel_path in MODEL_FILES.items():
        # Destination local path
        dest_filename = Path(rel_path).name
        dest_path = model_dir / dest_filename
        file_paths[key] = dest_path

        if dest_path.exists() and dest_path.stat().st_size > 1000:
            files_present[key] = True
            total_size += dest_path.stat().st_size
        else:
            files_present[key] = False
            missing.append(rel_path)

    is_ready = len(missing) == 0
    size_mb = total_size / (1024 * 1024)

    return ModelStatus(
        model_name=model_name,
        model_dir=model_dir,
        is_ready=is_ready,
        total_size_mb=round(size_mb, 2),
        files_present=files_present,
        file_paths=file_paths,
        missing_files=missing,
    )


def download_file(
    url: str,
    dest_path: Path,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    chunk_size: int = 4 * 1024 * 1024,
) -> None:
    """Download a file with streaming chunking and atomic temporary file swapping."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(".tmp")

    headers = {"User-Agent": "ExoCortex-Agent/0.1.0"}
    req = urllib.request.Request(url, headers=headers)

    with urllib.request.urlopen(req, timeout=60) as response:
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0

        with open(tmp_path, "wb") as out_file:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if progress_callback:
                    progress_callback(downloaded, total_size)

    # Atomic move
    if dest_path.exists():
        dest_path.unlink()
    tmp_path.rename(dest_path)


def download_model_assets(
    repo_id: str = DEFAULT_MODEL_REPO,
    model_name: str = DEFAULT_MODEL_NAME,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> ModelStatus:
    """
    Download missing model and tokenizer files from Hugging Face repository.
    """
    model_dir = get_model_directory(model_name)
    base_hf_url = f"https://huggingface.co/{repo_id}/resolve/main"

    for key, rel_path in MODEL_FILES.items():
        dest_filename = Path(rel_path).name
        dest_path = model_dir / dest_filename

        # Skip if already present with valid size
        if dest_path.exists() and dest_path.stat().st_size > 1000:
            logger.info("File already present: %s", dest_filename)
            continue

        url = f"{base_hf_url}/{rel_path}"
        logger.info("Downloading %s from %s...", dest_filename, url)

        def cb(dl: int, tot: int) -> None:
            if progress_callback:
                progress_callback(dest_filename, dl, tot)

        download_file(url, dest_path, progress_callback=cb)

    return check_model_status(model_name)
