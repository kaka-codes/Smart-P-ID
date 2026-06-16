# Smart P&ID Instrument Detector

Production-ready Streamlit app for processing P&ID PDFs into a structured instrument register with visual overlays and AI-assisted line-number mapping.

## Features

- PDF upload and cached processing by file hash.
- PDF rendering, tiled YOLO detection, global NMS, OCR, tag extraction, PDF text line extraction, and Gemini line mapping.
- Structured dataframe with tag, type, confidence, bounding box, page, and associated line number.
- Full P&ID drawing viewer with instrument overlays.
- Instrument counts by type, type filtering, multi-selection, and row-hover highlighting with blinking line markers.
- Modular architecture for future AI P&ID querying.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Add configuration in `.streamlit/secrets.toml` locally:

```toml
GEMINI_API_KEY = "your-key"
YOLO_MODEL_PATH = "best.pt"
```

On Streamlit Cloud, set `GEMINI_API_KEY` in app secrets. The YOLO weights are expected at `best.pt`, matching the uploaded model file in this repository.

3. Run:

```bash
streamlit run app.py
```

## Notes

- `YOLO_MODEL_PATH` is required for detection.
- `GEMINI_API_KEY` is required in Streamlit secrets. The app stops at startup if it is missing.
- Processing high-resolution P&IDs can be CPU and memory intensive on Streamlit Cloud. Tune `PID_ZOOM`, `PID_GRID`, and `PID_OVERLAP` in environment variables or Streamlit secrets as needed.
