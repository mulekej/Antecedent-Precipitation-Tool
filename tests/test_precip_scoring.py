from datetime import datetime, timedelta

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest
from matplotlib.backends.backend_pdf import PdfPages

from core.data_handler import (
    INSUFFICIENT_DATA,
    _compute_precip_condition,
    _score_precip_periods,
)
from core.report_plotter import _aggregate_batch_stats, _plot_precip_page

OBS_DATE = datetime(2026, 10, 21)
# The three 30-day periods end 0, 30 and 60 days before the observation date
PERIOD_DATES = [OBS_DATE - timedelta(days=d) for d in (0, 30, 60)]
LOW, HIGH = 1.0, 3.0
NORMAL, WET, DRY = 2.0, 4.0, 0.5


def _payload(observed):
    """Stored-JSON style payload; None in ``observed`` leaves that period missing.

    Missing values are absent from the dicts, as store_precip_data writes them.
    """
    key = lambda d: d.strftime("%Y-%m-%d")
    return {
        "obs_date": key(OBS_DATE),
        "rolling_total": {
            key(d): v for d, v in zip(PERIOD_DATES, observed) if v is not None
        },
        "normal_low": {key(d): LOW for d in PERIOD_DATES},
        "normal_high": {key(d): HIGH for d in PERIOD_DATES},
    }


def test_complete_data_scores_as_before():
    assert _compute_precip_condition(_payload([NORMAL, NORMAL, NORMAL])) == (
        "Normal Conditions",
        12,
    )
    assert _compute_precip_condition(_payload([WET, WET, WET])) == (
        "Wetter than Normal",
        18,
    )
    assert _compute_precip_condition(_payload([DRY, DRY, DRY])) == (
        "Drier than Normal",
        6,
    )


def test_measured_zero_is_scored_as_dry():
    periods, condition, score = _score_precip_periods(
        _payload([0.0, NORMAL, NORMAL])
    )
    assert periods[0]["condition"] == "Dry"
    assert periods[0]["value"] == 1
    assert (condition, score) == ("Drier than Normal", 9)


@pytest.mark.parametrize(
    "observed",
    [
        pytest.param([NORMAL, NORMAL, None], id="oldest_period_missing"),
        pytest.param([None, WET, WET], id="most_recent_missing_others_wet"),
        pytest.param([None, None, None], id="all_missing"),
    ],
)
def test_any_missing_period_is_insufficient_not_drier(observed):
    assert _compute_precip_condition(_payload(observed)) == (INSUFFICIENT_DATA, None)


def test_missing_normal_bound_is_insufficient():
    payload = _payload([NORMAL, NORMAL, NORMAL])
    del payload["normal_low"][PERIOD_DATES[1].strftime("%Y-%m-%d")]
    assert _compute_precip_condition(payload) == (INSUFFICIENT_DATA, None)


def test_missing_period_details():
    periods, _, _ = _score_precip_periods(_payload([None, WET, WET]))
    missing = periods[0]
    assert missing["condition"] == "Data Missing"
    assert missing["obs"] is None
    assert missing["value"] is None and missing["product"] is None
    assert [p["condition"] for p in periods[1:]] == ["Wet", "Wet"]


def test_precip_page_renders_with_missing_period(tmp_path):
    days = pd.date_range(OBS_DATE - timedelta(days=400), OBS_DATE)
    key = lambda d: d.strftime("%Y-%m-%d")
    payload = {
        "obs_date": key(OBS_DATE),
        "graph_end": key(OBS_DATE + timedelta(days=15)),
        "lat": 44.8638,
        "lon": -92.9155,
        "elev": 900.0,
        "local_stations_info": [],
        "daily_precip": {key(d): 0.1 for d in days},
        # The most recent 30-day total is missing, as stored after dropna
        "rolling_total": {key(d): NORMAL for d in days if d != OBS_DATE},
        "normal_low": {key(d): LOW for d in days},
        "normal_high": {key(d): HIGH for d in days},
    }
    with PdfPages(tmp_path / "page.pdf") as pdf:
        _plot_precip_page(
            payload, None, None, None, None, pdf, data_dir=str(tmp_path)
        )
    assert (tmp_path / "page.pdf").stat().st_size > 0


def _row(obs_date, score, condition):
    return {
        "lat": 44.0,
        "lon": -93.0,
        "obs_date": obs_date,
        "score": score,
        "condition": condition,
        "source": "GHCN",
    }


def test_batch_stats_count_insufficient_separately():
    rows = [
        _row("2026-10-21", 12, "Normal Conditions"),
        _row("2026-10-22", 18, "Wetter than Normal"),
        _row("2026-10-23", None, INSUFFICIENT_DATA),
    ]
    stats = _aggregate_batch_stats(rows)
    assert stats["avg_score"] == 15.0
    assert stats["n_scored"] == 2
    assert stats["n_insufficient"] == 1
    assert sum(stats["pie_sizes"]) == 2
    # Still listed in the breakdown table, with no score
    assert stats["table_vals"][-1][3:5] == ["—", INSUFFICIENT_DATA]
