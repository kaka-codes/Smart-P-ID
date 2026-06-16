from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

from smart_pid.cache import has_cached_result, load_result, pdf_hash, result_dir, save_result
from smart_pid.config import AppConfig
from smart_pid.detection import apply_global_nms, run_yolo
from smart_pid.line_mapping import map_lines
from smart_pid.models import PipelineResult
from smart_pid.ocr import run_ocr
from smart_pid.pdf_processing import extract_line_tags, render_pdf
from smart_pid.tiling import generate_tiles


class PipelineError(RuntimeError):
    pass


ProgressCallback = Callable[[str, str, float], None]


def _notify(callback: ProgressCallback | None, stage: str, detail: str, progress: float) -> None:
    if callback:
        callback(stage, detail, max(0.0, min(1.0, progress)))


def process_pdf(
    pdf_bytes: bytes,
    pdf_name: str,
    config: AppConfig,
    force: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> PipelineResult:
    digest = pdf_hash(pdf_bytes)
    cache_path = result_dir(config.cache_dir, digest)
    _notify(progress_callback, "Checking cache", "Looking for an existing processed result.", 0.02)
    if not force and has_cached_result(cache_path):
        _notify(progress_callback, "Loading cached result", "This PDF was already processed.", 1.0)
        return load_result(cache_path)

    try:
        _notify(progress_callback, "Preparing PDF", "Saving upload and creating workspace.", 0.05)
        cache_path.mkdir(parents=True, exist_ok=True)
        pdf_path = cache_path / "source.pdf"
        pdf_path.write_bytes(pdf_bytes)

        page_images = render_pdf(
            pdf_path,
            cache_path / "pages",
            config.zoom,
            lambda done, total: _notify(
                progress_callback,
                "Rendering PDF",
                f"Rendered page {done} of {total} at {config.zoom}x resolution.",
                0.08 + 0.12 * (done / max(total, 1)),
            ),
        )
        tile_metadata = generate_tiles(
            page_images,
            cache_path / "tiles",
            config.grid,
            config.overlap,
            lambda done, total: _notify(
                progress_callback,
                "Generating overlapping tiles",
                f"Prepared tiles for page {done} of {total}.",
                0.20 + 0.10 * (done / max(total, 1)),
            ),
        )
        _notify(
            progress_callback,
            "Detecting instruments",
            f"Running YOLO on {len(tile_metadata)} tiles.",
            0.30,
        )
        raw_detections = run_yolo(
            tile_metadata,
            config,
            lambda done, total, count: _notify(
                progress_callback,
                "Detecting instruments",
                f"YOLO tile {done} of {total}; {count} raw detections so far.",
                0.30 + 0.34 * (done / max(total, 1)),
            ),
        )
        _notify(progress_callback, "Merging detections", "Applying global NMS to remove duplicate boxes.", 0.66)
        detections = apply_global_nms(raw_detections, config)
        _notify(
            progress_callback,
            "Reading instrument tags",
            f"Running OCR on {len(detections)} final instrument boxes.",
            0.70,
        )
        instruments = run_ocr(
            detections,
            page_images,
            config.ocr_gpu,
            lambda done, total: _notify(
                progress_callback,
                "Reading instrument tags",
                f"OCR completed for instrument {done} of {total}.",
                0.70 + 0.14 * (done / max(total, 1)),
            ),
        )
        _notify(progress_callback, "Extracting line numbers", "Reading line-number text directly from the PDF.", 0.86)
        line_tags = extract_line_tags(pdf_path, config.zoom)
        _notify(progress_callback, "Preparing line mapping", "Building nearest line-number candidates for each instrument.", 0.92)
        instruments = map_lines(instruments, line_tags, page_images, config)

        if instruments.empty:
            instruments = pd.DataFrame(
                columns=[
                    "instrument_id",
                    "page",
                    "tag_number",
                    "instrument_type",
                    "line_number",
                    "confidence",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                ]
            )

        result = PipelineResult(
            pdf_hash=digest,
            pdf_name=pdf_name,
            cache_path=cache_path,
            instruments=instruments,
            line_tags=line_tags,
            page_images=page_images,
        )
        _notify(progress_callback, "Saving result", "Writing cached instrument table and rendered pages.", 0.98)
        save_result(result)
        _notify(progress_callback, "Complete", "Processing finished.", 1.0)
        return result
    except Exception as exc:
        raise PipelineError(f"Pipeline failed: {exc}") from exc
