from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from smart_pid.config import AppConfig

try:
    from rapidfuzz import fuzz
except ModuleNotFoundError:
    from difflib import SequenceMatcher

    class _FuzzFallback:
        @staticmethod
        def ratio(left: str, right: str) -> float:
            return SequenceMatcher(None, left, right).ratio() * 100

    fuzz = _FuzzFallback()


def nearest_line_candidates(instruments: pd.DataFrame, line_tags: pd.DataFrame, count: int) -> dict[int, list[dict[str, object]]]:
    output: dict[int, list[dict[str, object]]] = {}
    for inst in instruments.itertuples(index=False):
        page_lines = line_tags[line_tags["page"] == inst.page]
        candidates = []
        for tag in page_lines.itertuples(index=False):
            distance = math.hypot(float(inst.cx) - float(tag.img_x), float(inst.cy) - float(tag.img_y))
            candidates.append(
                {
                    "line_number": tag.text,
                    "distance": distance,
                    "x": int(tag.img_x),
                    "y": int(tag.img_y),
                }
            )
        output[int(inst.instrument_id)] = sorted(candidates, key=lambda item: item["distance"])[:count]
    return output


def _enhance_crop(crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
    thick = cv2.dilate(bw, np.ones((3, 3), np.uint8), iterations=2)
    result = 255 - thick
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


def _highlight_context(
    image: np.ndarray,
    instrument: pd.Series,
    nearest_tags: list[dict[str, object]],
    padding: int,
) -> np.ndarray:
    height, width = image.shape[:2]
    xs = [float(instrument["cx"])] + [float(tag["x"]) for tag in nearest_tags]
    ys = [float(instrument["cy"])] + [float(tag["y"]) for tag in nearest_tags]

    crop_x1 = max(0, int(min(xs) - padding))
    crop_y1 = max(0, int(min(ys) - padding))
    crop_x2 = min(width, int(max(xs) + padding))
    crop_y2 = min(height, int(max(ys) + padding))
    crop = _enhance_crop(image[crop_y1:crop_y2, crop_x1:crop_x2].copy())

    local_x1 = int(instrument["x1"] - crop_x1)
    local_y1 = int(instrument["y1"] - crop_y1)
    local_x2 = int(instrument["x2"] - crop_x1)
    local_y2 = int(instrument["y2"] - crop_y1)
    cv2.rectangle(crop, (local_x1, local_y1), (local_x2, local_y2), (0, 0, 255), 3)

    overlay = crop.copy()
    for line in nearest_tags:
        x = int(line["x"] - crop_x1)
        y = int(line["y"] - crop_y1)
        if 0 <= x < crop.shape[1] and 0 <= y < crop.shape[0]:
            cv2.circle(overlay, (x, y), 40, (255, 255, 0), -1)
            cv2.circle(crop, (x, y), 40, (220, 220, 220), 1)
    return cv2.addWeighted(overlay, 0.10, crop, 0.90, 0)


def _parse_json(text: str) -> dict[str, object]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.replace("```json", "").replace("```", "").strip()
    return json.loads(stripped)


def _gemini_line_number(crop: np.ndarray, instrument_tag: str, config: AppConfig) -> dict[str, object]:
    if not config.gemini_api_key:
        return {"instrument": instrument_tag, "line_number": "UNKNOWN", "confidence": 0.0}

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        cv2.imwrite(tmp.name, crop)
        image = Image.open(tmp.name)

    prompt = f"""
You are a senior P&ID engineer.

The red highlighted boxed instrument is: {instrument_tag}

Task:
Determine the incoming process line number associated with the red highlighted boxed instrument.

Nearby line tags are highlighted with light-yellow circles. All process line numbers follow a consistent format, typically dash-separated like:
- [Size or 4-digit code]-[Service]-[Sequence]-[Material] (e.g., 4"-FG-1001-CS)
- [Unit/Service Code]-[Size"]-[Material]-[Insulation] (e.g., 100FG101-4"-CS-A)

Follow the physical line trace through inline components to find a tag matching this format.


Connectivity Rules :-

1. Determine the INCOMING process line connected to the highlighted instrument.

2. Use flow arrows to determine direction and identify the incoming side.

3. Follow physical line connectivity, not proximity. Do NOT choose the nearest visible line number.

4. Treat a process line as continuous through valves, instruments, analyzers, transmitters, indicators, strainers, reducers, tees, branches, and other inline symbols.

5. Inline components do NOT break line continuity.

6. If the instrument is connected to or is inside a vessel or a drum, and is not connected to a line with line number then return that vessel or drum tag.

7. Continue tracing the connected line until a valid line number is found, even if the line number is far from the instrument.

8. A line changes only when it has more than one bent

9. Crossing lines are NOT connected unless a junction, branch, tee, or explicit connection symbol is shown.

10. If multiple line numbers are visible, return only the line number belonging to the same continuous incoming line as the highlighted instrument.

11. Ignore unrelated nearby text and line numbers.

12. If no valid incoming line or equipment connection can be determined, return UNKNOWN.

Return only JSON:
{{"instrument":"{instrument_tag}","line_number":"...","confidence":0.00}}
"""
    try:
        import google.genai as genai

        client = genai.Client(api_key=config.gemini_api_key)
        response = client.models.generate_content(model=config.gemini_model, contents=[prompt, image])
        return _parse_json(response.text)
    except ModuleNotFoundError:
        try:
            import google.generativeai as legacy_genai
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Gemini SDK is not installed in this Python environment. "
                "Run: pip install google-genai"
            ) from exc

        legacy_genai.configure(api_key=config.gemini_api_key)
        model = legacy_genai.GenerativeModel(config.gemini_model)
        response = model.generate_content([prompt, image])
        return _parse_json(response.text)


def _fuzzy_match_line(line_no: str, line_tags: pd.DataFrame, page: int) -> tuple[str, float, int | None, int | None]:
    if not line_no or line_no == "UNKNOWN":
        return "UNKNOWN", 0.0, None, None

    def _norm(s: str) -> str:
        return str(s).lower().replace(" ", "").replace(",", "")

    target = _norm(line_no)
    best_score = 0.0
    best_row = None
    for _, row in line_tags[line_tags["page"] == page].iterrows():
        score = fuzz.ratio(_norm(row["text"]), target)
        if score > best_score:
            best_score = float(score)
            best_row = row
    if best_row is not None and best_score >= 90:
        return str(best_row["text"]), best_score, int(best_row["img_x"]), int(best_row["img_y"])
    return str(line_no), best_score, None, None


def locate_line_number(line_no: str, line_tags: pd.DataFrame, page: int) -> tuple[str, float, int | None, int | None]:
    return _fuzzy_match_line(line_no, line_tags, page)


def map_single_instrument(
    instrument: pd.Series,
    line_tags: pd.DataFrame,
    page_image: Path,
    config: AppConfig,
) -> dict[str, object]:
    page = int(instrument["page"])
    page_lines = line_tags[line_tags["page"] == page]
    candidates = nearest_line_candidates(pd.DataFrame([instrument]), line_tags, config.line_candidate_count)
    nearest_tags = candidates.get(int(instrument["instrument_id"]), [])
    mapped = {"instrument": instrument.get("tag_number", ""), "line_number": "UNKNOWN", "confidence": 0.0}

    if config.gemini_api_key and not page_lines.empty:
        image = cv2.imread(str(page_image))
        crop = _highlight_context(image, instrument, nearest_tags, config.gemini_crop_padding)
        mapped = _gemini_line_number(crop, str(instrument.get("tag_number", "")), config)

    line_number, match_score, line_x, line_y = _fuzzy_match_line(
        str(mapped.get("line_number", "UNKNOWN")),
        line_tags,
        page,
    )
    row = instrument.to_dict()
    row["nearest_line_candidates"] = json.dumps(nearest_tags)
    row["line_number"] = line_number
    row["line_mapping_confidence"] = float(mapped.get("confidence", 0.0) or 0.0)
    row["line_match_score"] = match_score
    row["line_mapping_source"] = "gemini"
    row["line_x"] = line_x
    row["line_y"] = line_y
    return row


def map_lines(instruments: pd.DataFrame, line_tags: pd.DataFrame, page_images: dict[int, Path], config: AppConfig) -> pd.DataFrame:
    if instruments.empty:
        return instruments.copy()

    candidates = nearest_line_candidates(instruments, line_tags, config.line_candidate_count)
    rows = []

    for _, inst in instruments.iterrows():
        nearest_tags = candidates.get(int(inst["instrument_id"]), [])
        nearest_line = nearest_tags[0] if nearest_tags else {}
        row = inst.to_dict()
        line_number = row.get("line_number")
        if pd.isna(line_number) or not str(line_number):
            line_number = None
        line_x = row.get("line_x")
        line_y = row.get("line_y")
        row["nearest_line_candidates"] = json.dumps(nearest_tags)
        row["line_number"] = line_number or nearest_line.get("line_number") or "UNKNOWN"
        row["line_mapping_confidence"] = float(row.get("line_mapping_confidence") or 0.0)
        row["line_match_score"] = float(row.get("line_match_score") or 0.0)
        row["line_mapping_source"] = row.get("line_mapping_source") or "nearest"
        row["line_x"] = nearest_line.get("x") if pd.isna(line_x) else line_x
        row["line_y"] = nearest_line.get("y") if pd.isna(line_y) else line_y
        rows.append(row)

    return pd.DataFrame(rows)
