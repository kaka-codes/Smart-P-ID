from __future__ import annotations

import time

import streamlit as st
import streamlit.components.v1 as components

from smart_pid.config import AppConfig, ConfigError
from smart_pid.pipeline import PipelineError, process_pdf
from smart_pid.ui import render_app


st.set_page_config(
    page_title="Smart P&ID",
    page_icon=":material/account_tree:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    st.title("Smart P&ID")
    st.caption("Instrument detection, tag extraction, and AI-assisted line mapping from P&ID PDFs.")

    try:
        config = AppConfig.from_streamlit()
    except ConfigError as exc:
        st.error(str(exc))
        st.stop()

    uploaded = st.sidebar.file_uploader("Upload P&ID PDF", type=["pdf"])
    force = st.sidebar.button("Reprocess PDF", use_container_width=True)

    if uploaded is None:
        st.info("Upload a P&ID PDF to start.")
        st.stop()

    start_time = time.monotonic()
    timer_slot = st.empty()
    progress_slot = st.empty()
    stage_slot = st.empty()
    detail_slot = st.empty()

    with timer_slot.container():
        components.html(
            """
            <div style="font-family:Inter,Arial,sans-serif;font-size:15px;color:#0f172a;">
              Elapsed processing time: <strong id="pid-timer">0.0s</strong>
            </div>
            <script>
              const started = Date.now();
                const timer = document.getElementById("pid-timer");
                
                setInterval(() => {
                  const elapsed = (Date.now() - started) / 1000;
                
                  if (elapsed < 60) {
                    timer.textContent = elapsed.toFixed(1) + "s";
                  } else {
                    const minutes = Math.floor(elapsed / 60);
                    const seconds = Math.floor(elapsed % 60);
                    timer.textContent = `${minutes}m ${seconds}s`;
                  }
                }, 100);
            </script>
            """,
            height=32,
        )

    progress_bar = progress_slot.progress(0, text="Starting P&ID processing")

    def report_progress(stage: str, detail: str, progress: float) -> None:
        progress_bar.progress(progress, text=f"{stage} ({progress * 100:.0f}%)")
        stage_slot.markdown(f"**Current step:** {stage}")
        detail_slot.info(detail)

    try:
        result = process_pdf(
            uploaded.getvalue(),
            uploaded.name,
            config,
            force=force,
            progress_callback=report_progress,
        )
    except PipelineError as exc:
        st.error(str(exc))
        st.stop()

    total_elapsed = time.monotonic() - start_time
    timer_slot.success(f"P&ID processing complete in {total_elapsed:.1f}s.")
    progress_bar.progress(1.0, text="Complete")
    stage_slot.empty()
    detail_slot.empty()

    render_app(result, config)


if __name__ == "__main__":
    main()
