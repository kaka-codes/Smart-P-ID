from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Callable

import cv2
import pandas as pd


@lru_cache(maxsize=2)
def _reader(gpu: bool):
    import easyocr

    print("=" * 60)
    print("Initializing EasyOCR")
    print(f"GPU enabled: {gpu}")

    model_dir = Path(".easyocr_models")
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"EasyOCR model directory: {model_dir.resolve()}")
    print("Loading EasyOCR models... This may take a few minutes the first time.")

    try:
        reader = easyocr.Reader(
            ["en"],
            gpu=gpu,
            model_storage_directory=str(model_dir),
            download_enabled=True,
        )

        print("EasyOCR initialized successfully")
        print("=" * 60)

        return reader

    except Exception as e:
        print("EasyOCR initialization FAILED")
        print(f"Error: {e}")
        print("=" * 60)
        raise


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).replace("\n", " ")).strip()


def extract_tag(text: str) -> str:
    normalized = re.sub(r"[^A-Z0-9\s]", " ", str(text).upper())
    tokens = normalized.split()
    for i in range(len(tokens) - 1):
        first = tokens[i]
        second = tokens[i + 1]
        if not re.fullmatch(r"[A-Z]{2,5}", first):
            continue
        if not re.fullmatch(r"[A-Z0-9]+", second):
            continue
        digit_count = sum(ch.isdigit() for ch in second)
        letter_count = sum(ch.isalpha() for ch in second)
        if digit_count > letter_count:
            return f"{first} {second}"
    return ""


def instrument_type(tag: str, class_name: str = "") -> str:
    tag = str(tag).strip().upper()
    if tag:
        return tag.split()[0]
    return str(class_name or "UNKNOWN").upper()


def run_ocr(
    detections: pd.DataFrame,
    page_images: dict[int, Path],
    gpu: bool,
    progress_callback: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    if detections.empty:
        return detections.copy()

    print("Starting OCR pipeline...")
    reader = _reader(gpu)
    print("EasyOCR reader ready.")
    rows: list[dict[str, object]] = []
    loaded_images: dict[int, object] = {}

    total = len(detections)
    for index, det in enumerate(detections.itertuples(index=False), start=1):
        page = int(det.page)
        if page not in loaded_images:
            loaded_images[page] = cv2.imread(str(page_images[page]))
        image = loaded_images[page]
        height, width = image.shape[:2]
        margin = 10
        x1 = max(0, int(det.x1) - margin)
        y1 = max(0, int(det.y1) - margin)
        x2 = min(width, int(det.x2) + margin)
        y2 = min(height, int(det.y2) + margin)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            raw_text = ""
        else:
            crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            raw_text = " ".join(reader.readtext(thresh, detail=0, paragraph=False))

        cleaned = clean_text(raw_text)
        tag = extract_tag(cleaned)
        item = det._asdict()
        item.update(
            {
                "raw_text": raw_text,
                "clean_text": cleaned,
                "tag_number": tag,
                "instrument_type": instrument_type(tag, str(getattr(det, "class_name", ""))),
                "cx": (float(det.x1) + float(det.x2)) / 2,
                "cy": (float(det.y1) + float(det.y2)) / 2,
            }
        )
        rows.append(item)
        if progress_callback:
            progress_callback(index, total)

    return pd.DataFrame(rows)
