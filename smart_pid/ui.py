from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path
import zipfile
from xml.sax.saxutils import escape

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from smart_pid.cache import save_result
from smart_pid.config import AppConfig
from smart_pid.line_mapping import locate_line_number, map_single_instrument
from smart_pid.models import PipelineResult


def _image_data_uri(path: Path) -> str:
    data = path.read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _nearest_line(row: pd.Series) -> dict[str, object]:
    try:
        candidates = json.loads(str(row.get("nearest_line_candidates") or "[]"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(candidates, list) or not candidates:
        return {}
    first = candidates[0]
    return first if isinstance(first, dict) else {}


def _with_initial_line_numbers(instruments: pd.DataFrame) -> pd.DataFrame:
    updated = instruments.copy()
    if "line_number" not in updated:
        updated["line_number"] = "UNKNOWN"
    if "line_x" not in updated:
        updated["line_x"] = None
    if "line_y" not in updated:
        updated["line_y"] = None
    if "nearest_line_candidates" not in updated:
        return updated

    for idx, row in updated.iterrows():
        if str(row.get("line_mapping_source") or "").lower() == "gemini":
            continue
        current_line = row.get("line_number")
        line_number = "UNKNOWN" if pd.isna(current_line) or not str(current_line) else str(current_line)
        if line_number.upper() != "UNKNOWN":
            continue
        nearest = _nearest_line(row)
        if not nearest:
            continue
        line_x = row.get("line_x")
        line_y = row.get("line_y")
        updated.at[idx, "line_number"] = nearest.get("line_number") or "UNKNOWN"
        updated.at[idx, "line_x"] = nearest.get("x") if pd.isna(line_x) else line_x
        updated.at[idx, "line_y"] = nearest.get("y") if pd.isna(line_y) else line_y
    return updated


def _viewer_html(image_path: Path, instruments: pd.DataFrame, selected_ids: set[int]) -> str:
    data_uri = _image_data_uri(image_path)
    rows = []
    overlays = []
    for inst in instruments.itertuples(index=False):
        inst_id = int(inst.instrument_id)
        tag = html.escape(str(getattr(inst, "tag_number", "") or "UNTAGGED"))
        typ = html.escape(str(getattr(inst, "instrument_type", "") or "UNKNOWN"))
        line = html.escape(str(getattr(inst, "line_number", "") or "UNKNOWN"))
        overlays.append(
            {
                "id": inst_id,
                "x1": float(inst.x1),
                "y1": float(inst.y1),
                "x2": float(inst.x2),
                "y2": float(inst.y2),
                "tag": tag,
                "line": line,
                "line_x": None if pd.isna(getattr(inst, "line_x", None)) else float(getattr(inst, "line_x")),
                "line_y": None if pd.isna(getattr(inst, "line_y", None)) else float(getattr(inst, "line_y")),
                "selected": inst_id in selected_ids,
            }
        )
        selected = " selected" if inst_id in selected_ids else ""
        rows.append(f"<tr data-id='{inst_id}' class='{selected}'><td>{tag}</td><td>{typ}</td><td>{line}</td></tr>")

    payload = json.dumps(overlays)
    return f"""
<div class="pid-shell">
  <div class="drawing" id="drawing">
    <div class="viewer-tools">
      <button type="button" id="zoom-out" title="Zoom out">-</button>
      <button type="button" id="zoom-reset" title="Reset view">100%</button>
      <button type="button" id="zoom-in" title="Zoom in">+</button>
    </div>
    <canvas id="pid-canvas"></canvas>
    <svg id="overlay"></svg>
    <img id="pid-img" src="{data_uri}" alt="" />
  </div>
  <div class="table-wrap">
    <table>
      <thead><tr><th>Tag</th><th>Type</th><th>Line</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </div>
</div>
<style>
  .pid-shell {{ display:grid; grid-template-columns:minmax(0, 1fr) 420px; gap:16px; height:760px; min-height:520px; font-family:Inter, Arial, sans-serif; }}
  .drawing {{ position:relative; width:100%; height:100%; min-height:520px; overflow:hidden; background:#fff; border:1px solid #d7dde8; border-radius:8px; cursor:grab; touch-action:none; }}
  .drawing.dragging {{ cursor:grabbing; }}
  #pid-canvas {{ position:absolute; inset:0; width:100%; height:100%; user-select:none; pointer-events:none; }}
  #pid-img {{ display:none; }}
  #overlay {{ position:absolute; inset:0; width:100%; height:100%; pointer-events:none; }}
  .viewer-tools {{ position:absolute; z-index:5; top:10px; left:10px; display:flex; gap:6px; padding:6px; background:rgba(255,255,255,0.92); border:1px solid #d7dde8; border-radius:8px; box-shadow:0 8px 24px rgba(15,23,42,0.12); }}
  .viewer-tools button {{ min-width:36px; height:32px; border:1px solid #cbd5e1; background:#ffffff; color:#0f172a; border-radius:6px; font-weight:700; cursor:pointer; }}
  .viewer-tools button:hover {{ background:#f1f5f9; }}
  .table-wrap {{ overflow:auto; border:1px solid #d7dde8; border-radius:8px; background:white; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th {{ position:sticky; top:0; background:#f1f5f9; text-align:left; padding:9px; border-bottom:1px solid #d7dde8; }}
  td {{ padding:8px 9px; border-bottom:1px solid #edf1f7; white-space:nowrap; }}
  tr:hover, tr.active {{ background:#e8f2ff; }}
  tr.selected {{ background:#ecfdf5; }}
  .box {{ fill:rgba(31, 111, 235, 0.08); stroke:#1f6feb; stroke-width:3; vector-effect:non-scaling-stroke; }}
  .box.selected {{ stroke:#0f9f6e; stroke-width:5; }}
  .box.active {{ stroke:#ef4444; stroke-width:6; }}
  .label {{ font-size:24px; fill:#0f172a; font-weight:700; paint-order:stroke; stroke:#fff; stroke-width:5px; }}
  .line-dot {{ fill:rgba(255, 214, 10, 0.72); stroke:#f59e0b; stroke-width:7; paint-order:stroke; }}
  .line-dot.active {{ animation: blink 0.55s steps(2, start) infinite; fill:rgba(255, 49, 49, 0.82); stroke:#ffea00; stroke-width:12; }}
  .connector-line {{ fill: none; stroke: #334155; stroke-width: 4; stroke-dasharray: 10, 5; opacity: 0; pointer-events: none; transition: opacity 0.15s; }}
  .connector-line.active {{ opacity: 1; stroke: #7f1d1d; stroke-width: 6; }}
  @keyframes blink {{ 50% {{ opacity:1; fill:rgba(255, 234, 0, 0.96); stroke:#dc2626; }} }}
  @media (max-width: 900px) {{ .pid-shell {{ grid-template-columns:1fr; height:auto; }} .drawing {{ height:640px; }} .table-wrap {{ max-height:360px; }} }}
</style>
<script>
const instruments = {payload};
const drawing = document.getElementById('drawing');
const canvas = document.getElementById('pid-canvas');
const ctx = canvas.getContext('2d', {{ alpha: false }});
const img = document.getElementById('pid-img');
const svg = document.getElementById('overlay');
let zoom = 1;
let fitZoom = 1;
let panX = 0;
let panY = 0;
let dragging = false;
let lastX = 0;
let lastY = 0;
let rafId = 0;
let overlayGroup = null;

function scheduleRender() {{
  document.getElementById('zoom-reset').textContent = `${{Math.round(zoom * 100)}}%`;
  if (rafId) return;
  rafId = window.requestAnimationFrame(renderViewport);
}}

function clampZoom(value) {{
  return Math.min(24, Math.max(0.03, value));
}}

function clampPan() {{
  const viewportW = drawing.clientWidth;
  const viewportH = drawing.clientHeight;
  const imageW = img.naturalWidth * zoom;
  const imageH = img.naturalHeight * zoom;
  if (imageW <= viewportW) {{
    panX = (viewportW - imageW) / 2;
  }} else {{
    panX = Math.min(0, Math.max(viewportW - imageW, panX));
  }}
  if (imageH <= viewportH) {{
    panY = (viewportH - imageH) / 2;
  }} else {{
    panY = Math.min(0, Math.max(viewportH - imageH, panY));
  }}
}}

function zoomAt(nextZoom, clientX, clientY) {{
  const rect = drawing.getBoundingClientRect();
  const localX = clientX - rect.left;
  const localY = clientY - rect.top;
  const contentX = (localX - panX) / zoom;
  const contentY = (localY - panY) / zoom;
  zoom = clampZoom(nextZoom);
  panX = localX - contentX * zoom;
  panY = localY - contentY * zoom;
  clampPan();
  scheduleRender();
}}

function zoomCenter(factor) {{
  const rect = drawing.getBoundingClientRect();
  zoomAt(zoom * factor, rect.left + rect.width / 2, rect.top + rect.height / 2);
}}

function resetView() {{
  zoom = fitZoom;
  panX = 0;
  panY = 0;
  clampPan();
  scheduleRender();
}}

function buildOverlay() {{
  const w = img.naturalWidth, h = img.naturalHeight;
  svg.setAttribute('viewBox', `0 0 ${{drawing.clientWidth}} ${{drawing.clientHeight}}`);
  svg.innerHTML = '';
  overlayGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
  svg.appendChild(overlayGroup);
  instruments.forEach(d => {{
    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    rect.setAttribute('x', d.x1); rect.setAttribute('y', d.y1);
    rect.setAttribute('width', d.x2 - d.x1); rect.setAttribute('height', d.y2 - d.y1);
    rect.setAttribute('class', 'box' + (d.selected ? ' selected' : ''));
    rect.dataset.id = d.id;
    overlayGroup.appendChild(rect);
    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    text.setAttribute('x', d.x1); text.setAttribute('y', Math.max(26, d.y1 - 8));
    text.setAttribute('class', 'label'); text.textContent = d.tag;
    overlayGroup.appendChild(text);
    if (d.line_x !== null && d.line_y !== null) {{
      const conn = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      conn.setAttribute('x1', (d.x1 + d.x2) / 2);
      conn.setAttribute('y1', (d.y1 + d.y2) / 2);
      conn.setAttribute('x2', d.line_x);
      conn.setAttribute('y2', d.line_y);
      conn.setAttribute('class', 'connector-line');
      conn.dataset.id = d.id;
      overlayGroup.appendChild(conn);

      const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      circle.setAttribute('cx', d.line_x); circle.setAttribute('cy', d.line_y); circle.setAttribute('r', 42);
      circle.setAttribute('class', 'line-dot'); circle.dataset.id = d.id;
      overlayGroup.appendChild(circle);
    }}
  }});
}}

function renderViewport() {{
  rafId = 0;
  const cssW = drawing.clientWidth;
  const cssH = drawing.clientHeight;
  const dpr = window.devicePixelRatio || 1;
  const pixelW = Math.max(1, Math.round(cssW * dpr));
  const pixelH = Math.max(1, Math.round(cssH * dpr));
  if (canvas.width !== pixelW || canvas.height !== pixelH) {{
    canvas.width = pixelW;
    canvas.height = pixelH;
  }}
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(img, panX, panY, img.naturalWidth * zoom, img.naturalHeight * zoom);
  svg.setAttribute('viewBox', `0 0 ${{cssW}} ${{cssH}}`);
  if (overlayGroup) {{
    overlayGroup.setAttribute('transform', `translate(${{panX}} ${{panY}}) scale(${{zoom}})`);
  }}
}}

function draw() {{
  const w = img.naturalWidth, h = img.naturalHeight;
  if (!w || !h) return;
  if (!drawing.clientWidth || !drawing.clientHeight) {{
    window.setTimeout(draw, 50);
    return;
  }}
  fitZoom = Math.min(1, drawing.clientWidth / w, drawing.clientHeight / h);
  if (!drawing.dataset.ready) {{
    zoom = fitZoom;
    drawing.dataset.ready = '1';
  }}
  buildOverlay();
  clampPan();
  scheduleRender();
}}
function setActive(id, on) {{
  document.querySelectorAll(`[data-id="${{id}}"]`).forEach(el => el.classList.toggle('active', on));
}}

drawing.addEventListener('wheel', event => {{
  event.preventDefault();
  const factor = event.deltaY < 0 ? 1.15 : 1 / 1.15;
  zoomAt(zoom * factor, event.clientX, event.clientY);
}}, {{ passive:false }});

drawing.addEventListener('pointerdown', event => {{
  if (event.target.closest('.viewer-tools')) return;
  dragging = true;
  lastX = event.clientX;
  lastY = event.clientY;
  drawing.classList.add('dragging');
  drawing.setPointerCapture(event.pointerId);
}});

drawing.addEventListener('pointermove', event => {{
  if (!dragging) return;
  panX += event.clientX - lastX;
  panY += event.clientY - lastY;
  lastX = event.clientX;
  lastY = event.clientY;
  clampPan();
  scheduleRender();
}});

function endDrag(event) {{
  dragging = false;
  drawing.classList.remove('dragging');
  if (event.pointerId !== undefined && drawing.hasPointerCapture(event.pointerId)) {{
    drawing.releasePointerCapture(event.pointerId);
  }}
}}

drawing.addEventListener('pointerup', endDrag);
drawing.addEventListener('pointercancel', endDrag);
document.getElementById('zoom-out').addEventListener('click', () => zoomCenter(1 / 1.2));
document.getElementById('zoom-in').addEventListener('click', () => zoomCenter(1.2));
document.getElementById('zoom-reset').addEventListener('click', resetView);
document.querySelectorAll('tbody tr').forEach(row => {{
  row.addEventListener('mouseenter', () => setActive(row.dataset.id, true));
  row.addEventListener('mouseleave', () => setActive(row.dataset.id, false));
}});
window.addEventListener('resize', () => {{
  clampPan();
  buildOverlay();
  scheduleRender();
}});
if (img.complete) draw(); else img.onload = draw;
</script>
"""


def _selection_rows(event: object) -> list[int]:
    if event is None:
        return []
    selection = getattr(event, "selection", None)
    if selection is None and isinstance(event, dict):
        selection = event.get("selection")
    if selection is None:
        return []
    rows = getattr(selection, "rows", None)
    if rows is None and isinstance(selection, dict):
        rows = selection.get("rows")
    return list(rows or [])


def _excel_bytes(register: pd.DataFrame) -> bytes:
    def col_name(index: int) -> str:
        name = ""
        while index:
            index, remainder = divmod(index - 1, 26)
            name = chr(65 + remainder) + name
        return name

    def cell_xml(row_num: int, col_num: int, value: object) -> str:
        ref = f"{col_name(col_num)}{row_num}"
        style = ' s="1"' if row_num == 1 else ""
        if pd.isna(value):
            return f'<c r="{ref}"{style}/>'
        if isinstance(value, bool):
            return f'<c r="{ref}"{style} t="b"><v>{int(value)}</v></c>'
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return f'<c r="{ref}"{style}><v>{value}</v></c>'
        text = escape(str(value))
        return f'<c r="{ref}"{style} t="inlineStr"><is><t>{text}</t></is></c>'

    rows = []
    values = [list(register.columns)] + register.fillna("").values.tolist()
    for row_num, row_values in enumerate(values, start=1):
        cells = "".join(cell_xml(row_num, col_num, value) for col_num, value in enumerate(row_values, start=1))
        rows.append(f'<row r="{row_num}">{cells}</row>')

    cols = []
    for col_num, column in enumerate(register.columns, start=1):
        text_lengths = [len(str(column))]
        if column in register:
            text_lengths.extend(len("" if pd.isna(value) else str(value)) for value in register[column])
        width = max(10, min(max(text_lengths) + 3, 60))
        cols.append(f'<col min="{col_num}" max="{col_num}" width="{width}" customWidth="1"/>')

    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<cols>{''.join(cols)}</cols>"
        f"<sheetData>{''.join(rows)}</sheetData>"
        "</worksheet>"
    )
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="Instrument Register" sheetId="1" r:id="rId1"/></sheets>'
        "</workbook>"
    )
    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
        "</Relationships>"
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        "</Relationships>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        "</Types>"
    )
    styles_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="2">'
        '<font><sz val="11"/><color theme="1"/><name val="Calibri"/><family val="2"/></font>'
        '<font><b/><sz val="11"/><color theme="1"/><name val="Calibri"/><family val="2"/></font>'
        '</fonts>'
        '<fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="2">'
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
        '<xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/>'
        '</cellXfs>'
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        '</styleSheet>'
    )

    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("xl/workbook.xml", workbook_xml)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        archive.writestr("xl/styles.xml", styles_xml)
        archive.writestr("xl/worksheets/sheet1.xml", sheet_xml)
    return output.getvalue()


def _resolve_selected_line(result: PipelineResult, config: AppConfig, instrument_id: int) -> bool:
    if not config.gemini_api_key:
        st.warning("Add GEMINI_API_KEY in Streamlit secrets to resolve line numbers with Gemini.")
        return False

    matches = result.instruments.index[result.instruments["instrument_id"] == instrument_id].tolist()
    if not matches:
        return False

    idx = matches[0]
    instrument = result.instruments.loc[idx]
    st.toast(f"Consulting Gemini for {instrument.get('tag_number') or 'instrument'}...")
    with st.spinner("Analyzing P&ID connectivity..."):
        try:
            mapped = map_single_instrument(
                instrument,
                result.line_tags,
                result.page_images[int(instrument["page"])],
                config,
            )
            st.toast("Gemini analysis complete!", icon="✅")
        except Exception as exc:
            st.error(f"Gemini line mapping failed: {exc}")
            return False
    for key, value in mapped.items():
        result.instruments.at[idx, key] = value
    save_result(result)
    return True


def _render_line_resolver_table(result: PipelineResult, config: AppConfig, rows: pd.DataFrame, page: int) -> None:
    st.markdown(
        """
        <style>
        div[data-testid="stVerticalBlock"]:has(.gemini-resolver-table) div[data-testid="stHorizontalBlock"] {
            border-bottom: 1px solid #d7dde8;
            padding: 4px 0;
            margin: 0;
        }
        div[data-testid="stVerticalBlock"]:has(.gemini-resolver-table) div[data-testid="stHorizontalBlock"]:first-of-type {
            border-top: 1px solid #d7dde8;
            background: #f8fafc;
        }
        .gemini-resolver-table {
            height: 0;
        }
        </style>
        <div class="gemini-resolver-table"></div>
        """,
        unsafe_allow_html=True,
    )
    header = st.columns([0.22, 0.12, 0.34, 0.32], gap="small")
    header[0].markdown("**Tag**")
    header[1].markdown("**Type**")
    header[2].markdown("**Line**")

    for row in rows.itertuples(index=False):
        inst_id = int(row.instrument_id)
        tag = str(getattr(row, "tag_number", "") or "UNTAGGED")
        typ = str(getattr(row, "instrument_type", "") or "UNKNOWN")
        line = str(getattr(row, "line_number", "") or "UNKNOWN")
        c1, c2, c3, c4 = st.columns([0.22, 0.12, 0.34, 0.32], gap="small")
        c1.caption(tag)
        c2.caption(typ)
        c3.caption(line)
        if c4.button(
            "Resolve with Gemini",
            key=f"resolve_gemini:{result.pdf_hash}:{page}:{inst_id}",
            use_container_width=True,
        ):
            if _resolve_selected_line(result, config, inst_id):
                st.rerun()


def render_app(result: PipelineResult, config: AppConfig) -> None:
    instruments = _with_initial_line_numbers(result.instruments)
    if "instrument_type" not in instruments:
        instruments["instrument_type"] = "UNKNOWN"

    st.sidebar.success(f"Cache key: {result.pdf_hash}")
    resolve_param = st.query_params.get("resolve_instrument")
    if resolve_param is not None:
        try:
            resolve_id = int(resolve_param)
        except (TypeError, ValueError):
            resolve_id = None
        if resolve_id is not None:
            resolved = _resolve_selected_line(result, config, resolve_id)
            if not resolved:
                return
        del st.query_params["resolve_instrument"]
        st.rerun()

    page = st.sidebar.selectbox("Page", sorted(result.page_images.keys()))
    page_df = instruments[instruments["page"] == page].copy()

    types = sorted(t for t in page_df["instrument_type"].dropna().unique())
    type_filter_key = f"instrument_type_filter:{result.pdf_hash}:{page}"
    if type_filter_key not in st.session_state:
        st.session_state[type_filter_key] = types
    else:
        st.session_state[type_filter_key] = [t for t in st.session_state[type_filter_key] if t in types]

    with st.sidebar.form(f"instrument_type_form_{result.pdf_hash}_{page}"):
        pending_types = st.multiselect(
            "Instrument type",
            types,
            default=st.session_state[type_filter_key],
            key=f"{type_filter_key}:pending",
        )
        show_types = st.form_submit_button("Show", use_container_width=True)
    if show_types:
        st.session_state[type_filter_key] = pending_types

    selected_types = st.session_state[type_filter_key]
    filtered = page_df[page_df["instrument_type"].isin(selected_types)] if selected_types else page_df.iloc[0:0]

    counts = filtered["instrument_type"].value_counts().rename_axis("type").reset_index(name="count")
    c1, c2 = st.columns([0.28, 0.72])
    with c1:
        st.subheader("Counts")
        st.dataframe(counts, hide_index=True, use_container_width=True)
    with c2:
        st.subheader("Instrument Register")
        display_cols = [
            "instrument_id",
            "tag_number",
            "instrument_type",
            "line_number",
        ]
        available = [col for col in display_cols if col in filtered.columns]
        register = filtered[available].reset_index(drop=True)
        if "instrument_id" in register:
            register["instrument_id"] = register["instrument_id"].astype(int) + 1
        edited_register = st.data_editor(
            register,
            hide_index=True,
            use_container_width=True,
            height=220,
            disabled=["instrument_id"],
            key=f"instrument_register_{page}",
            column_config={
                "instrument_id": st.column_config.NumberColumn("ID", disabled=True),
                "tag_number": st.column_config.TextColumn("Tag Number"),
                "instrument_type": st.column_config.TextColumn("Instrument Type"),
                "line_number": st.column_config.TextColumn("Line Number"),
            },
        )
        if not edited_register.empty and "instrument_id" in edited_register:
            edited_register = edited_register.copy()
            edited_by_id = edited_register.copy()
            edited_by_id["instrument_id"] = edited_by_id["instrument_id"].astype(int) - 1
            edited_by_id = edited_by_id.set_index("instrument_id")
            current_by_id = filtered.set_index("instrument_id")
            changed = False
            edited_types: set[str] = set()
            for col in [c for c in available if c != "instrument_id"]:
                filtered[col] = filtered["instrument_id"].map(edited_by_id[col]).fillna(filtered[col])
            for inst_id, edited_row in edited_by_id.iterrows():
                if inst_id not in current_by_id.index:
                    continue
                result_matches = result.instruments.index[result.instruments["instrument_id"] == inst_id].tolist()
                if not result_matches:
                    continue
                result_idx = result_matches[0]
                for col in [c for c in available if c != "instrument_id"]:
                    old_value = current_by_id.at[inst_id, col]
                    new_value = edited_row[col]
                    old_text = "" if pd.isna(old_value) else str(old_value)
                    new_text = "" if pd.isna(new_value) else str(new_value)
                    if old_text == new_text:
                        continue
                    result.instruments.at[result_idx, col] = new_value
                    changed = True
                    if col == "instrument_type" and new_text:
                        edited_types.add(new_text)
                    if col == "line_number":
                        line_number, match_score, line_x, line_y = locate_line_number(new_text, result.line_tags, page)
                        filtered.loc[filtered["instrument_id"] == inst_id, "line_number"] = line_number
                        filtered.loc[filtered["instrument_id"] == inst_id, "line_x"] = line_x
                        filtered.loc[filtered["instrument_id"] == inst_id, "line_y"] = line_y
                        result.instruments.at[result_idx, "line_number"] = line_number
                        result.instruments.at[result_idx, "line_match_score"] = match_score
                        result.instruments.at[result_idx, "line_mapping_source"] = "manual"
                        result.instruments.at[result_idx, "line_x"] = line_x
                        result.instruments.at[result_idx, "line_y"] = line_y
            if changed:
                if edited_types:
                    selected = list(st.session_state.get(type_filter_key, []))
                    st.session_state[type_filter_key] = selected + [t for t in sorted(edited_types) if t not in selected]
                save_result(result)
                st.rerun()

        st.download_button(
            "Download Excel",
            data=_excel_bytes(edited_register),
            file_name=f"instrument_register_{result.pdf_hash}_page_{page}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    tag_labels = {
        int(r.instrument_id): f"{r.tag_number or 'UNTAGGED'} | {r.instrument_type} | {r.line_number}"
        for r in filtered.itertuples(index=False)
    }
    highlight_key = f"highlight_instruments:{result.pdf_hash}:{page}"
    if highlight_key not in st.session_state:
        st.session_state[highlight_key] = []
    else:
        st.session_state[highlight_key] = [
            inst_id for inst_id in st.session_state[highlight_key] if inst_id in tag_labels
        ]

    with st.sidebar.form(f"highlight_instruments_form_{result.pdf_hash}_{page}"):
        pending_highlights = st.multiselect(
            "Highlight instruments",
            list(tag_labels),
            default=st.session_state[highlight_key],
            format_func=lambda inst_id: tag_labels.get(inst_id, str(inst_id)),
            key=f"{highlight_key}:pending",
        )
        apply_highlights = st.form_submit_button("Highlight", use_container_width=True)
    if apply_highlights:
        st.session_state[highlight_key] = pending_highlights

    selected_ids = set(st.session_state[highlight_key])

    st.subheader("P&ID Viewer")
    components.html(_viewer_html(result.page_images[page], filtered, selected_ids), height=800, scrolling=True)
    with st.expander("Resolve line numbers with Gemini"):
        _render_line_resolver_table(result, config, filtered, page)
