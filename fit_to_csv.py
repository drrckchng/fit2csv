from __future__ import annotations

import csv
import io
import statistics
from datetime import datetime
from typing import Any

import fitdecode

SUPPORTED_SPORT = "running"

CoreRow = dict[str, str]
UnitsMap = dict[str, str]


def _normalize_value(value: Any) -> str:
    """Convert FIT field values to CSV-safe strings."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return ";".join(str(item) for item in value)
    return str(value)


def _field_name(field: Any, fallback_prefix: str) -> str:
    name = getattr(field, "name", None)
    if name:
        return str(name)

    def_num = getattr(field, "def_num", None)
    if def_num is not None:
        return f"{fallback_prefix}_{def_num}"

    return fallback_prefix


def _field_units(field: Any) -> str:
    units = getattr(field, "units", None)
    if units is None:
        return ""
    return str(units).strip()


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_datetime(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _format_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _header_with_unit(name: str, units_map: UnitsMap) -> str:
    unit = units_map.get(name, "").strip()
    return f"{name} [{unit}]" if unit else name


def _is_unknown_column(name: str) -> bool:
    return name.startswith("unknown_")


def _pace_min_per_km(speed_mps: float) -> float | None:
    if speed_mps <= 0:
        return None
    return 1000.0 / (speed_mps * 60.0)


def _pace_min_per_mi(speed_mps: float) -> float | None:
    if speed_mps <= 0:
        return None
    return 1609.344 / (speed_mps * 60.0)


def _step_length_to_stride_m(step_length: float, step_unit: str) -> float | None:
    """Convert Garmin step length to stride length in meters."""
    unit = step_unit.strip().lower()

    if unit in {"m", "meter", "meters"}:
        step_m = step_length
    elif unit in {"cm", "centimeter", "centimeters"}:
        step_m = step_length / 100.0
    elif unit in {"mm", "millimeter", "millimeters"}:
        step_m = step_length / 1000.0
    else:
        # Fallback heuristic when FIT unit metadata is missing.
        step_m = step_length / 1000.0 if step_length > 10 else step_length

    if step_m <= 0:
        return None
    return step_m * 2.0


def _resolve_cadence_factor(rows: list[CoreRow]) -> tuple[float, str]:
    cadence_values = [
        numeric
        for row in rows
        for numeric in [_to_float(row.get("cadence", ""))]
        if numeric is not None and numeric > 0
    ]

    if not cadence_values:
        return 1.0, "No cadence values found for conversion."

    median_cadence = statistics.median(cadence_values)
    if median_cadence <= 130:
        return 2.0, (
            "Auto cadence conversion applied: low median cadence "
            f"({median_cadence:.1f}) suggests strides/min, converted to steps/min."
        )

    return 1.0, (
        "Auto cadence conversion not applied: values already look like steps/min "
        f"(median cadence {median_cadence:.1f})."
    )


def _validate_running_sport(detected_sport: str) -> None:
    if detected_sport != SUPPORTED_SPORT:
        found = detected_sport or "unknown"
        raise ValueError(f"Only running FIT files are supported. Found sport: {found}.")


def _extract_session_sport(frame: Any) -> str:
    for field in frame.fields:
        if getattr(field, "name", "") == "sport":
            return _normalize_value(getattr(field, "value", ""))
    return ""


def _add_field_to_row(row: CoreRow, units_map: UnitsMap, field: Any, name: str) -> None:
    value = _normalize_value(getattr(field, "value", None))
    if value == "" or _is_unknown_column(name):
        return

    units = _field_units(field)
    if units and name not in units_map:
        units_map[name] = units

    row[name] = value


def _extract_record_row(frame: Any, units_map: UnitsMap, include_developer_fields: bool) -> CoreRow:
    row: CoreRow = {}

    for field in frame.fields:
        field_name = _field_name(field, "field")
        _add_field_to_row(row, units_map, field, field_name)

    if include_developer_fields:
        for dev_field in getattr(frame, "dev_fields", []):
            dev_name = _field_name(dev_field, "dev_field")
            if dev_name in row:
                dev_name = f"dev_{dev_name}"
            _add_field_to_row(row, units_map, dev_field, dev_name)

    # Running-dynamics aliases for consistency in downstream export.
    if "stance_time" in row and "gct_ms" not in row:
        row["gct_ms"] = row["stance_time"]
        units_map.setdefault("gct_ms", "ms")

    if "stance_time_balance" in row and "gct_balance_percent" not in row:
        row["gct_balance_percent"] = row["stance_time_balance"]
        units_map.setdefault("gct_balance_percent", "%")

    return row


def _add_cadence_spm(rows: list[CoreRow], units_map: UnitsMap) -> str:
    factor, cadence_note = _resolve_cadence_factor(rows)

    for row in rows:
        cadence_value = _to_float(row.get("cadence", ""))
        if cadence_value is None:
            continue
        row["cadence_spm"] = _format_number(cadence_value * factor)

    units_map.setdefault("cadence_spm", "spm")
    return cadence_note


def parse_fit_record_rows(
    fit_bytes: bytes,
    *,
    include_developer_fields: bool = True,
) -> tuple[list[CoreRow], UnitsMap, str, str]:
    """Parse running FIT bytes into `record` rows."""
    rows: list[CoreRow] = []
    units_map: UnitsMap = {}
    detected_sport = ""

    with fitdecode.FitReader(io.BytesIO(fit_bytes)) as fit_file:
        for frame in fit_file:
            if not isinstance(frame, fitdecode.records.FitDataMessage):
                continue

            if frame.name == "session" and not detected_sport:
                detected_sport = _extract_session_sport(frame)

            if frame.name != "record":
                continue

            row = _extract_record_row(frame, units_map, include_developer_fields)
            rows.append(row)

    _validate_running_sport(detected_sport)
    cadence_note = _add_cadence_spm(rows, units_map)

    return rows, units_map, detected_sport, cadence_note


def _compute_row_pace(
    row: CoreRow,
    prev_time: datetime | None,
    prev_distance: float | None,
) -> tuple[str, str, datetime | None, float | None]:
    timestamp = row.get("timestamp", "")
    distance_text = row.get("distance", "")

    speed_value = _to_float(row.get("enhanced_speed", ""))
    if speed_value is None:
        speed_value = _to_float(row.get("speed", ""))

    pace_km: float | None = None
    pace_mi: float | None = None

    if speed_value is not None:
        pace_km = _pace_min_per_km(speed_value)
        pace_mi = _pace_min_per_mi(speed_value)
    else:
        current_time = _to_datetime(timestamp)
        current_distance = _to_float(distance_text)

        if (
            prev_time is not None
            and current_time is not None
            and prev_distance is not None
            and current_distance is not None
        ):
            delta_seconds = (current_time - prev_time).total_seconds()
            delta_distance = current_distance - prev_distance
            if delta_seconds > 0 and delta_distance > 0:
                delta_speed = delta_distance / delta_seconds
                pace_km = _pace_min_per_km(delta_speed)
                pace_mi = _pace_min_per_mi(delta_speed)

        if current_time is not None:
            prev_time = current_time
        if current_distance is not None:
            prev_distance = current_distance

    pace_km_text = _format_number(pace_km) if pace_km is not None else ""
    pace_mi_text = _format_number(pace_mi) if pace_mi is not None else ""

    if speed_value is not None:
        current_time = _to_datetime(timestamp)
        current_distance = _to_float(distance_text)
        if current_time is not None:
            prev_time = current_time
        if current_distance is not None:
            prev_distance = current_distance

    return pace_km_text, pace_mi_text, prev_time, prev_distance


def rows_to_ordered_csv_bytes(rows: list[CoreRow], units_map: UnitsMap) -> bytes:
    """Export rows with core running columns first, then all other decoded columns."""
    if not rows:
        raise ValueError("No record rows were found in this FIT file.")

    distance_unit = units_map.get("distance", "m") or "m"
    vo_unit = units_map.get("vertical_oscillation", "mm") or "mm"
    vr_unit = units_map.get("vertical_ratio", "%") or "%"

    base_headers = [
        "timestamp",
        f"distance [{distance_unit}]",
        "pace [min/km]",
        "pace [min/mi]",
        "HR [bpm]",
        "cadence [spm]",
        "stride length [m]",
        f"vert oscillation [{vo_unit}]",
        "GCT [ms]",
        f"vert ratio [{vr_unit}]",
    ]

    consumed_columns = {
        "timestamp",
        "distance",
        "heart_rate",
        "cadence",
        "cadence_spm",
        "step_length",
        "vertical_oscillation",
        "gct_ms",
        "stance_time",
        "vertical_ratio",
        "enhanced_speed",
        "speed",
    }

    extra_columns = sorted({key for row in rows for key in row.keys()} - consumed_columns)
    headers = base_headers + [_header_with_unit(col, units_map) for col in extra_columns]

    csv_rows: list[CoreRow] = []
    prev_time: datetime | None = None
    prev_distance: float | None = None
    step_unit = units_map.get("step_length", "")

    for row in rows:
        pace_km_text, pace_mi_text, prev_time, prev_distance = _compute_row_pace(
            row, prev_time, prev_distance
        )

        stride_text = ""
        step_value = _to_float(row.get("step_length", ""))
        if step_value is not None:
            stride_m = _step_length_to_stride_m(step_value, step_unit)
            if stride_m is not None:
                stride_text = _format_number(stride_m)

        export_row: CoreRow = {
            "timestamp": row.get("timestamp", ""),
            f"distance [{distance_unit}]": row.get("distance", ""),
            "pace [min/km]": pace_km_text,
            "pace [min/mi]": pace_mi_text,
            "HR [bpm]": row.get("heart_rate", ""),
            "cadence [spm]": row.get("cadence_spm", row.get("cadence", "")),
            "stride length [m]": stride_text,
            f"vert oscillation [{vo_unit}]": row.get("vertical_oscillation", ""),
            "GCT [ms]": row.get("gct_ms", row.get("stance_time", "")),
            f"vert ratio [{vr_unit}]": row.get("vertical_ratio", ""),
        }

        for col in extra_columns:
            export_row[_header_with_unit(col, units_map)] = row.get(col, "")

        csv_rows.append(export_row)

    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=headers, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(csv_rows)

    return csv_buffer.getvalue().encode("utf-8")
