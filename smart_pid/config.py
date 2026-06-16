from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError


class ConfigError(RuntimeError):
    pass


def _secret(name: str, default: Any = None) -> Any:
    try:
        if name in st.secrets:
            return st.secrets[name]
    except StreamlitSecretNotFoundError:
        pass
    return os.getenv(name, default)


@dataclass(frozen=True)
class AppConfig:
    cache_dir: Path
    yolo_model_path: Path
    gemini_api_key: str | None
    gemini_model: str
    zoom: int
    grid: int
    overlap: float
    yolo_imgsz: int
    yolo_conf: float
    nms_score_threshold: float
    nms_iou_threshold: float
    final_conf_threshold: float
    ocr_gpu: bool
    line_candidate_count: int
    gemini_crop_padding: int

    @classmethod
    def from_streamlit(cls) -> "AppConfig":
        model_path = Path(str(_secret("YOLO_MODEL_PATH", "best.pt")))
        if not model_path.exists():
            raise ConfigError(
                "YOLO model not found. Set YOLO_MODEL_PATH in Streamlit secrets or place weights at best.pt."
            )

        return cls(
            cache_dir=Path(str(_secret("PID_CACHE_DIR", ".cache/pid_app"))),
            yolo_model_path=model_path,
            gemini_api_key=_secret("GEMINI_API_KEY"),
            gemini_model=str(_secret("GEMINI_MODEL", "gemini-3.1-flash-lite")),
            zoom=int(_secret("PID_ZOOM", 8)),
            grid=int(_secret("PID_GRID", 5)),
            overlap=float(_secret("PID_OVERLAP", 0.30)),
            yolo_imgsz=int(_secret("YOLO_IMGSZ", 1280)),
            yolo_conf=float(_secret("YOLO_CONF", 0.25)),
            nms_score_threshold=float(_secret("NMS_SCORE_THRESHOLD", 0.7)),
            nms_iou_threshold=float(_secret("NMS_IOU_THRESHOLD", 0.1)),
            final_conf_threshold=float(_secret("FINAL_CONF_THRESHOLD", 0.62)),
            ocr_gpu=str(_secret("OCR_GPU", "false")).lower() == "true",
            line_candidate_count=int(_secret("LINE_CANDIDATE_COUNT", 7)),
            gemini_crop_padding=int(_secret("GEMINI_CROP_PADDING", 1000)),
        )
