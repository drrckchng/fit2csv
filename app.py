from __future__ import annotations

import csv
from pathlib import Path

import streamlit as st

from fit_to_csv import parse_fit_record_rows, rows_to_ordered_csv_bytes


def _render_header() -> None:
    st.set_page_config(page_title="Garmin FIT Record to CSV", page_icon=":runner:", layout="centered")
    st.title("Garmin FIT Record to CSV")
    st.write(
        "Upload a Garmin `.fit` file. This app exports only `record` rows, "
        "with key running metrics first and all other decoded record columns appended."
    )


def _render_controls() -> tuple[st.runtime.uploaded_file_manager.UploadedFile | None, bool, int]:
    uploaded_file = st.file_uploader("Choose a .fit file", type=["fit"])
    include_dev_fields = st.checkbox("Include developer fields (for extra metrics)", value=True)
    preview_rows = st.slider("Preview rows", min_value=5, max_value=100, value=20, step=5)
    return uploaded_file, include_dev_fields, preview_rows


def _build_preview_rows(csv_bytes: bytes, limit: int) -> list[dict[str, str]]:
    decoded = csv_bytes.decode("utf-8")
    return list(csv.DictReader(decoded.splitlines()))[:limit]


def main() -> None:
    _render_header()
    uploaded_file, include_dev_fields, preview_rows = _render_controls()

    if uploaded_file is None:
        st.caption("Waiting for file upload...")
        return

    st.info(f"Selected file: {uploaded_file.name}")

    try:
        rows, units_map, detected_sport, cadence_note = parse_fit_record_rows(
            uploaded_file.getvalue(),
            include_developer_fields=include_dev_fields,
        )

        if not rows:
            st.error("No `record` rows were found in this FIT file.")
            return

        csv_bytes = rows_to_ordered_csv_bytes(rows, units_map)
        output_name = f"{Path(uploaded_file.name).stem}_record.csv"

        st.success(f"Parsed {len(rows)} record rows.")
        st.write(f"Detected sport: `{detected_sport}`")
        st.write(cadence_note)

        st.subheader("Preview")
        st.dataframe(
            _build_preview_rows(csv_bytes, preview_rows),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            label="Download record CSV",
            data=csv_bytes,
            file_name=output_name,
            mime="text/csv",
        )
    except Exception as error:
        st.error(f"Could not parse/convert file: {error}")


if __name__ == "__main__":
    main()
