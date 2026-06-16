from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class PipelineResult:
    pdf_hash: str
    pdf_name: str
    cache_path: Path
    instruments: pd.DataFrame
    line_tags: pd.DataFrame
    page_images: dict[int, Path]
