from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import fitz
import pandas as pd


LINE_PATTERN = re.compile(
    r"(?:\b(?:\d+\"|\d{4})-[A-Z]{1,3}(?:-[A-Z0-9]+){2,}\b|\d+[A-Z]{2}\d+-[\d/]+\"-[A-Z0-9]+-[A-Z])"
)


def render_pdf(
    pdf_path: Path,
    out_dir: Path,
    zoom: int,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[int, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    page_images: dict[int, Path] = {}
    with fitz.open(pdf_path) as doc:
        total_pages = len(doc)
        for page_index, page in enumerate(doc):
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            image_path = out_dir / f"page_{page_index + 1}.png"
            pix.save(str(image_path))
            page_images[page_index + 1] = image_path
            if progress_callback:
                progress_callback(page_index + 1, total_pages)
    return page_images


def pdf_to_image(cx: float, cy: float, page: fitz.Page, zoom: int) -> tuple[int, int]:
    rotation = page.rotation
    if rotation == 270:
        x_rot = cy
        y_rot = page.rect.height - cx
    elif rotation == 90:
        x_rot = page.rect.width - cy
        y_rot = cx
    elif rotation == 180:
        x_rot = page.rect.width - cx
        y_rot = page.rect.height - cy
    else:
        x_rot = cx
        y_rot = cy
    return int(x_rot * zoom), int(y_rot * zoom)


def extract_line_tags(pdf_path: Path, zoom: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc):
            for block in page.get_text("dict").get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = str(span.get("text", "")).strip()
                        if not LINE_PATTERN.search(text):
                            continue
                        x0, y0, x1, y1 = span["bbox"]
                        cx = (x0 + x1) / 2
                        cy = (y0 + y1) / 2
                        img_x, img_y = pdf_to_image(cx, cy, page, zoom)
                        rows.append(
                            {
                                "page": page_index + 1,
                                "text": text,
                                "pdf_cx": cx,
                                "pdf_cy": cy,
                                "img_x": img_x,
                                "img_y": img_y,
                            }
                        )
    return pd.DataFrame(rows)
