from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import pandas as pd


def generate_tiles(
    page_images: dict[int, Path],
    out_dir: Path,
    grid: int,
    overlap: float,
    progress_callback: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata: list[dict[str, object]] = []

    total_pages = len(page_images)
    for page_index, (page, image_path) in enumerate(page_images.items(), start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        tile_w = max(1, int(width / grid))
        tile_h = max(1, int(height / grid))
        step_x = max(1, int(tile_w * (1 - overlap)))
        step_y = max(1, int(tile_h * (1 - overlap)))

        x_positions = list(range(0, width - tile_w + 1, step_x))
        y_positions = list(range(0, height - tile_h + 1, step_y))
        if not x_positions or x_positions[-1] != width - tile_w:
            x_positions.append(width - tile_w)
        if not y_positions or y_positions[-1] != height - tile_h:
            y_positions.append(height - tile_h)

        for row, y1 in enumerate(y_positions):
            for col, x1 in enumerate(x_positions):
                x2 = x1 + tile_w
                y2 = y1 + tile_h
                tile_name = f"page{page}_r{row}_c{col}.png"
                tile_path = out_dir / tile_name
                cv2.imwrite(str(tile_path), image[y1:y2, x1:x2])
                metadata.append(
                    {
                        "page": page,
                        "tile_path": str(tile_path),
                        "row": row,
                        "col": col,
                        "x_offset": x1,
                        "y_offset": y1,
                        "width": tile_w,
                        "height": tile_h,
                    }
                )
        if progress_callback:
            progress_callback(page_index, total_pages)

    return pd.DataFrame(metadata)
