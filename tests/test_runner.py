from datetime import datetime

import pytest

from engine import EventDispatcher
from runner import (
    RunRequest,
    Runner,
    dates_from_csv,
    dates_from_range,
    parse_date,
)


class RecordingDispatcher(EventDispatcher):
    """Records notified messages instead of dispatching them."""

    def __init__(self):
        super().__init__()
        self.sent = []

    def notify(self, message):
        self.sent.append(message)


def legacy_gui_plan(
    lat, lon, dates, analysis_types, scope, gridded, custom_path, custom_name, data_dir
):
    """The message plan PrecipGUI.run_btn_clicked sent before the runner existed,
    copied from gui.py at 964cc00 with dispatcher.notify replaced by list.append."""
    sent = []
    is_huc = scope in ("HUC12", "HUC10", "HUC8")
    is_custom = scope == "Custom Polygon"
    if is_huc or is_custom:
        msg = {
            "message_type": "huc_analysis",
            "lat": lat,
            "lon": lon,
            "analysis_date": dates[0],
            "output_dir": "output",
            "data_dir": data_dir,
            "huc_level": {"HUC12": 12, "HUC10": 10, "HUC8": 8}.get(scope, 8),
            "analysis_types": analysis_types,
        }
        if gridded:
            msg["gridded"] = True
        if is_custom and custom_path:
            msg["custom_polygon"] = custom_path
            msg["custom_name"] = custom_name.strip() or "CUSTOM"
        sent.append(msg)
        return sent

    for current_date in dates:
        bulk = {
            "lat": lat,
            "lon": lon,
            "analysis_date": current_date,
            "output_dir": "output",
            "data_dir": data_dir,
        }
        if "precip" in analysis_types:
            msg = {"message_type": "precip_analysis", **bulk}
            if gridded:
                msg["gridded"] = True
            sent.append(msg)
        for t in ("usgs", "nwm", "wimp", "pdsi"):
            if t in analysis_types:
                sent.append({"message_type": f"{t}_analysis", **bulk})
        sent.append(
            {"message_type": "generate_pdf", "analysis_types": analysis_types, **bulk}
        )
    if len(dates) > 1:
        sent.append(
            {
                "message_type": "merge_pdfs",
                "lat": lat,
                "lon": lon,
                "start_date": dates[0],
                "end_date": dates[-1],
                "output_dir": "output",
            }
        )
    return sent


SCOPE_CODES = {
    "Single Point": "point",
    "HUC12": "huc12",
    "HUC10": "huc10",
    "HUC8": "huc8",
    "Custom Polygon": "polygon",
}
ONE_DATE = [datetime(2024, 6, 15)]
THREE_DATES = [datetime(2024, 6, 15), datetime(2024, 6, 16), datetime(2024, 6, 17)]
DEFAULT_TYPES = ["precip", "wimp", "pdsi"]
ALL_TYPES = ["precip", "usgs", "nwm", "wimp", "pdsi"]


@pytest.mark.parametrize("gui_scope", list(SCOPE_CODES))
@pytest.mark.parametrize("dates", [ONE_DATE, THREE_DATES], ids=["one", "three"])
@pytest.mark.parametrize("types", [DEFAULT_TYPES, ALL_TYPES], ids=["default", "all"])
@pytest.mark.parametrize("gridded", [False, True], ids=["ghcn", "gridded"])
def test_plan_matches_legacy_gui(gui_scope, dates, types, gridded):
    custom_path = "C:/shapes/site.shp"
    custom_name = "Site A"
    request = RunRequest(
        lat=44.95,
        lon=-93.1,
        dates=dates,
        scope=SCOPE_CODES[gui_scope],
        analysis_types=types,
        gridded=gridded,
        output_dir="output",
        data_dir="C:/apt/data",
    )
    if request.scope == "polygon":
        request.custom_polygon = custom_path
        request.custom_name = custom_name

    dispatcher = RecordingDispatcher()
    Runner(dispatcher).run(request)

    assert dispatcher.sent == legacy_gui_plan(
        44.95,
        -93.1,
        dates,
        types,
        gui_scope,
        gridded,
        custom_path,
        custom_name,
        "C:/apt/data",
    )


def test_point_run_sends_analyses_in_fixed_order():
    request = RunRequest(
        lat=44.95,
        lon=-93.1,
        dates=ONE_DATE,
        analysis_types=["pdsi", "precip", "nwm"],
    )
    dispatcher = RecordingDispatcher()
    Runner(dispatcher).run(request)

    assert [m["message_type"] for m in dispatcher.sent] == [
        "precip_analysis",
        "nwm_analysis",
        "pdsi_analysis",
        "generate_pdf",
    ]


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"lat": 91.0}, "Latitude"),
        ({"lon": -181.0}, "Longitude"),
        ({"dates": []}, "No valid dates"),
        ({"scope": "huc6"}, "Unknown scope"),
        ({"analysis_types": ["precip", "rain"]}, "Unknown analysis type"),
        ({"analysis_types": []}, "at least one"),
        ({"scope": "polygon"}, "shapefile"),
    ],
)
def test_invalid_requests_are_rejected(changes, message):
    fields = {"lat": 44.95, "lon": -93.1, "dates": ONE_DATE, **changes}
    dispatcher = RecordingDispatcher()

    with pytest.raises(ValueError, match=message):
        Runner(dispatcher).run(RunRequest(**fields))
    assert dispatcher.sent == []


def _dispatcher_with(handlers):
    dispatcher = EventDispatcher()
    for message_type, handler in handlers.items():
        dispatcher.register(message_type, handler)
    return dispatcher


def test_clean_run_is_ok():
    dispatcher = _dispatcher_with({})
    result = Runner(dispatcher).run(RunRequest(lat=44.95, lon=-93.1, dates=ONE_DATE))

    assert result.status == "ok"
    assert result.warnings == [] and result.errors == []


def test_adapter_warnings_are_collected():
    def failing_pdsi(message):
        return {
            "message_type": "warning",
            "warning_type": "pdsi_analysis_failed",
            "error": "boom",
            "original_message": message,
        }

    dispatcher = _dispatcher_with({"pdsi_analysis": failing_pdsi})
    result = Runner(dispatcher).run(RunRequest(lat=44.95, lon=-93.1, dates=ONE_DATE))

    assert result.status == "partial"
    assert [w["warning_type"] for w in result.warnings] == ["pdsi_analysis_failed"]
    assert result.errors == []


def test_handler_exceptions_are_collected():
    def crashing_wimp(message):
        raise RuntimeError("wimp exploded")

    dispatcher = _dispatcher_with({"wimp_analysis": crashing_wimp})
    result = Runner(dispatcher).run(RunRequest(lat=44.95, lon=-93.1, dates=ONE_DATE))

    assert result.status == "failed"
    assert len(result.errors) == 1
    assert result.errors[0]["error"] == "wimp exploded"
    assert result.errors[0]["original_message"]["message_type"] == "wimp_analysis"


def test_each_run_gets_its_own_result():
    calls = {"n": 0}

    def fails_first_time(message):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"message_type": "warning", "warning_type": "x", "error": "once"}

    dispatcher = _dispatcher_with({"pdsi_analysis": fails_first_time})
    runner = Runner(dispatcher)
    request = RunRequest(lat=44.95, lon=-93.1, dates=ONE_DATE)

    assert runner.run(request).status == "partial"
    assert runner.run(request).status == "ok"


def test_warning_outside_a_run_is_ignored():
    dispatcher = EventDispatcher()
    Runner(dispatcher)
    dispatcher.notify({"message_type": "warning", "warning_type": "stray"})


def test_dates_from_range_is_inclusive():
    dates = dates_from_range(parse_date("2024-02-27"), parse_date("2024-03-01"))
    assert [d.strftime("%Y-%m-%d") for d in dates] == [
        "2024-02-27",
        "2024-02-28",
        "2024-02-29",
        "2024-03-01",
    ]


def test_dates_from_range_rejects_reversed_range():
    with pytest.raises(ValueError, match="on or after"):
        dates_from_range(parse_date("2024-03-01"), parse_date("2024-02-01"))


def test_dates_from_csv_takes_first_date_per_row_sorted_and_unique(tmp_path):
    path = tmp_path / "dates.csv"
    path.write_text(
        "site,date,note\n"
        "A,2024-06-17,x\n"
        "B,2024-06-15,2024-01-01\n"
        "C,2024-06-17,dup\n"
        "D,not a date,\n",
        encoding="utf-8-sig",
    )
    assert dates_from_csv(str(path)) == [datetime(2024, 6, 15), datetime(2024, 6, 17)]


def test_dates_from_csv_requires_existing_file(tmp_path):
    with pytest.raises(ValueError, match="CSV"):
        dates_from_csv(str(tmp_path / "missing.csv"))
