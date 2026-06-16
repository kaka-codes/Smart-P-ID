from __future__ import annotations

import os
from typing import Callable

import cv2
import pandas as pd

from smart_pid.config import AppConfig


def run_yolo(
    tile_metadata: pd.DataFrame,
    config: AppConfig,
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> pd.DataFrame:
    ultralytics_dir = config.cache_dir / "ultralytics"
    ultralytics_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ultralytics_dir))

    from ultralytics import YOLO

    model = YOLO(str(config.yolo_model_path))
    detections: list[dict[str, object]] = []

    total_tiles = len(tile_metadata)
    for tile_index, row in enumerate(tile_metadata.itertuples(index=False), start=1):
        results = model.predict(
            source=str(row.tile_path),
            imgsz=config.yolo_imgsz,
            conf=config.yolo_conf,
            verbose=False,
        )
        names = getattr(results[0], "names", {}) or {}
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0]) if box.cls is not None else -1
            area = float((x2 - x1) * (y2 - y1))
            detections.append(
                {
                    "page": int(row.page),
                    "x1": float(x1 + row.x_offset),
                    "y1": float(y1 + row.y_offset),
                    "x2": float(x2 + row.x_offset),
                    "y2": float(y2 + row.y_offset),
                    "confidence": conf,
                    "class_id": cls,
                    "class_name": str(names.get(cls, "instrument")),
                    "score": conf * area,
                }
            )
        if progress_callback:
            progress_callback(tile_index, total_tiles, len(detections))

    if not detections:
        return pd.DataFrame(columns=["page", "x1", "y1", "x2", "y2", "confidence", "class_id", "class_name"])

    return pd.DataFrame(detections)


def apply_global_nms(detections: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    if detections.empty:
        return detections.copy()

    kept_pages: list[pd.DataFrame] = []
    for _, page_df in detections.groupby("page"):
        boxes = [
            [int(r.x1), int(r.y1), int(r.x2 - r.x1), int(r.y2 - r.y1)]
            for r in page_df.itertuples(index=False)
        ]
        scores = [float(r.score) for r in page_df.itertuples(index=False)]
        indices = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            score_threshold=config.nms_score_threshold,
            nms_threshold=config.nms_iou_threshold,
        )
        if len(indices) == 0:
            continue
        kept = page_df.iloc[indices.flatten()].copy()
        kept = kept[kept["confidence"] >= config.final_conf_threshold]
        kept_pages.append(kept)

    if not kept_pages:
        return detections.iloc[0:0].copy()

    final = pd.concat(kept_pages, ignore_index=True)
    final.insert(0, "instrument_id", range(len(final)))
    return final.drop(columns=["score"], errors="ignore")
