# runner.py
"""
Builds and sends the message plan for one APT run.
Shared by the GUI and the command line so both send exactly the same messages,
and collects the warnings and handler errors raised along the way.
"""

import csv
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional

from engine import EventDispatcher

logger = logging.getLogger(__name__)

DATE_FORMAT = "%Y-%m-%d"

# Single-point runs send analyses in this order regardless of how they were selected
ANALYSIS_ORDER = ("precip", "usgs", "nwm", "wimp", "pdsi")
DEFAULT_ANALYSIS_TYPES = ("precip", "wimp", "pdsi")

SCOPES = ("point", "huc12", "huc10", "huc8", "polygon")
HUC_LEVELS = {"huc12": 12, "huc10": 10, "huc8": 8}


# ====================== DATE HELPERS ======================


def parse_date(text: str) -> datetime:
    """Parse a YYYY-MM-DD string."""
    return datetime.strptime(text.strip(), DATE_FORMAT)


def dates_from_range(start: datetime, end: datetime) -> List[datetime]:
    """Every day from start to end, inclusive."""
    if end < start:
        raise ValueError("End date must be on or after start date")
    return [start + timedelta(days=d) for d in range((end - start).days + 1)]


def dates_from_csv(path: str) -> List[datetime]:
    """First YYYY-MM-DD cell on each row of a CSV, sorted and de-duplicated."""
    if not path or not os.path.isfile(path):
        raise ValueError("Valid CSV file path required")
    dates = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.reader(fh):
            for cell in row:
                try:
                    dates.append(parse_date(cell))
                    break
                except ValueError:
                    continue
    return sorted(set(dates))


# ====================== REQUEST / RESULT ======================


@dataclass
class RunRequest:
    """Everything needed to run APT for one location."""

    lat: float
    lon: float
    dates: List[datetime]
    scope: str = "point"  # one of SCOPES
    analysis_types: List[str] = field(
        default_factory=lambda: list(DEFAULT_ANALYSIS_TYPES)
    )
    gridded: bool = False
    custom_polygon: Optional[str] = None
    custom_name: str = "CUSTOM"
    output_dir: str = "output"
    data_dir: str = "data"

    def validate(self):
        """Raise ValueError describing the first problem found."""
        if not -90 <= self.lat <= 90:
            raise ValueError(f"Latitude {self.lat} is outside -90 to 90")
        if not -180 <= self.lon <= 180:
            raise ValueError(f"Longitude {self.lon} is outside -180 to 180")
        if not self.dates:
            raise ValueError("No valid dates provided")
        if self.scope not in SCOPES:
            raise ValueError(
                f"Unknown scope '{self.scope}'. Use one of: {', '.join(SCOPES)}"
            )
        unknown = [t for t in self.analysis_types if t not in ANALYSIS_ORDER]
        if unknown:
            raise ValueError(
                f"Unknown analysis type(s) {unknown}. "
                f"Use any of: {', '.join(ANALYSIS_ORDER)}"
            )
        if not self.analysis_types:
            raise ValueError("Select at least one analysis type")
        if self.scope == "polygon" and not self.custom_polygon:
            raise ValueError("Custom Polygon scope needs a watershed shapefile")

    @property
    def is_area(self) -> bool:
        return self.scope != "point"


@dataclass
class RunResult:
    """What happened during a run, beyond the files it wrote."""

    request: RunRequest
    warnings: List[dict] = field(default_factory=list)
    errors: List[dict] = field(default_factory=list)

    @property
    def status(self) -> str:
        """'failed' if a handler raised, 'partial' if any analysis warned, else 'ok'."""
        if self.errors:
            return "failed"
        if self.warnings:
            return "partial"
        return "ok"


# ====================== RUNNER ======================


class Runner:
    """Turns a RunRequest into dispatcher messages and collects the outcome."""

    def __init__(self, dispatcher: EventDispatcher):
        self.dispatcher = dispatcher
        self._result: Optional[RunResult] = None
        dispatcher.register("warning", self._on_warning)
        dispatcher.add_error_listener(self._on_error)

    def run(self, request: RunRequest) -> RunResult:
        """Validate the request, send its messages, and return the outcome.
        Blocks until every message has been handled."""
        request.validate()
        self._result = RunResult(request=request)
        try:
            if request.is_area:
                self._run_area(request)
            else:
                self._run_point(request)
            return self._result
        finally:
            self._result = None

    # ---------------------------------------------------------------
    # Message plans
    # ---------------------------------------------------------------
    def _run_point(self, request: RunRequest):
        """Single point: every analysis + a PDF per date, then a merge if several dates."""
        for current_date in request.dates:
            bulk = {
                "lat": request.lat,
                "lon": request.lon,
                "analysis_date": current_date,
                "output_dir": request.output_dir,
                "data_dir": request.data_dir,
            }
            for analysis_type in ANALYSIS_ORDER:
                if analysis_type not in request.analysis_types:
                    continue
                msg = {"message_type": f"{analysis_type}_analysis", **bulk}
                if analysis_type == "precip" and request.gridded:
                    msg["gridded"] = True
                self.dispatcher.notify(msg)
            self.dispatcher.notify(
                {
                    "message_type": "generate_pdf",
                    "analysis_types": request.analysis_types,
                    **bulk,
                }
            )

        if len(request.dates) > 1:
            self.dispatcher.notify(
                {
                    "message_type": "merge_pdfs",
                    "lat": request.lat,
                    "lon": request.lon,
                    "start_date": request.dates[0],
                    "end_date": request.dates[-1],
                    "output_dir": request.output_dir,
                }
            )

    def _run_area(self, request: RunRequest):
        """HUC or custom polygon: one huc_analysis message for the first date.
        AreaAdapter expands it into per-point analyses."""
        if len(request.dates) > 1:
            logger.info(
                "Watershed runs use the first date only (%s); %d later date(s) ignored",
                request.dates[0].strftime(DATE_FORMAT),
                len(request.dates) - 1,
            )
        msg = {
            "message_type": "huc_analysis",
            "lat": request.lat,
            "lon": request.lon,
            "analysis_date": request.dates[0],
            "output_dir": request.output_dir,
            "data_dir": request.data_dir,
            "huc_level": HUC_LEVELS.get(request.scope, 8),
            "analysis_types": request.analysis_types,
        }
        if request.gridded:
            msg["gridded"] = True
        if request.scope == "polygon":
            msg["custom_polygon"] = request.custom_polygon
            msg["custom_name"] = request.custom_name or "CUSTOM"
        self.dispatcher.notify(msg)

    # ---------------------------------------------------------------
    # Outcome collection
    # ---------------------------------------------------------------
    def _on_warning(self, message: dict):
        """Handler for 'warning' messages, which adapters return on failure."""
        logger.warning(
            "Run warning (%s): %s",
            message.get("warning_type"),
            message.get("error", ""),
        )
        if self._result is not None:
            self._result.warnings.append(message)

    def _on_error(self, error_msg: dict):
        """Dispatcher error listener: a handler raised and the queue was cleared."""
        if self._result is not None:
            self._result.errors.append(error_msg)
