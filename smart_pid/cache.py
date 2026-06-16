from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from smart_pid.models import PipelineResult


def pdf_hash(pdf_bytes: bytes) -> str:
    return hashlib.sha256(pdf_bytes).hexdigest()[:24]


def result_dir(cache_dir: Path, digest: str) -> Path:
    return cache_dir / digest


def has_cached_result(path: Path) -> bool:
    return (
        (path / "manifest.json").exists()
        and (path / "instruments.parquet").exists()
        and (path / "line_tags.parquet").exists()
    )


def load_result(path: Path) -> PipelineResult:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    instruments = pd.read_parquet(path / "instruments.parquet")
    line_tags = pd.read_parquet(path / "line_tags.parquet")
    page_images = {int(k): path / v for k, v in manifest["page_images"].items()}
    return PipelineResult(
        pdf_hash=manifest["pdf_hash"],
        pdf_name=manifest["pdf_name"],
        cache_path=path,
        instruments=instruments,
        line_tags=line_tags,
        page_images=page_images,
    )


def save_result(result: PipelineResult) -> None:
    result.cache_path.mkdir(parents=True, exist_ok=True)
    result.instruments.to_parquet(result.cache_path / "instruments.parquet", index=False)
    result.line_tags.to_parquet(result.cache_path / "line_tags.parquet", index=False)
    manifest = {
        "pdf_hash": result.pdf_hash,
        "pdf_name": result.pdf_name,
        "page_images": {str(k): str(v.relative_to(result.cache_path)) for k, v in result.page_images.items()},
    }
    (result.cache_path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
