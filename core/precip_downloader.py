import logging
import os
import sys
import glob
from datetime import datetime, timedelta

import pandas as pd
import requests
from geopy.distance import great_circle
import time

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

from config import get_base_url

logger = logging.getLogger(__name__)

# Nearest stations downloaded when choosing the primary station
MAX_PRIMARY_CANDIDATES = 40
# Extra station downloads allowed while filling gaps in the composite
MAX_FILL_FETCHES = 60


# ====================== ELEVATION ======================
def get_elevation(lat: float, lon: float, unit: str = "feet") -> float:
    """Try multiple elevation APIs with fallbacks."""
    apis = [
        ("USGS EPQS", lambda: _get_epqs_elevation(lat, lon, unit)),
        ("OpenTopo", lambda: _get_opentopo_elevation(lat, lon, unit)),
        ("OpenElevation", lambda: _get_open_elevation(lat, lon, unit)),
    ]
    for name, func in apis:
        try:
            elev = func()
            if elev is not None and elev > -1000:
                logger.debug(f"Elevation from {name}: {elev:.1f} ft")
                return elev
        except Exception as e:
            logger.debug(f"{name} failed: {e}")
        time.sleep(0.3)
    logger.warning(f"All elevation APIs failed for ({lat}, {lon})")
    return -1.0


def _get_epqs_elevation(lat, lon, unit="feet"):
    url = get_base_url("usgs_epqs")
    params = {"x": lon, "y": lat, "units": unit}
    r = requests.get(url, params=params, timeout=10)
    return r.json()["value"] if r.ok else None


def _get_opentopo_elevation(lat, lon, unit="feet"):
    url = get_base_url("opentopo_elevation")
    r = requests.get(url, params={"locations": f"{lat},{lon}"}, timeout=10)
    if r.ok:
        elev_m = r.json()["results"][0]["elevation"]
        return elev_m * 3.28084 if unit == "feet" else elev_m
    return None


def _get_open_elevation(lat, lon, unit="feet"):
    url = get_base_url("open_elevation")
    r = requests.get(url, params={"locations": f"{lat},{lon}"}, timeout=10)
    if r.ok:
        elev_m = r.json()["results"][0]["elevation"]
        return elev_m * 3.28084 if unit == "feet" else elev_m
    return None


# ====================== GHCN CORE ======================
def get_ghcn_timeseries_data(
    lat: float, lon: float, start_date_str: str, end_date_str: str, data_dir: str
):
    """
    Matches original anteProcess.py logic as closely as possible,
    with advanced file-based caching for GHCN data.
    """
    cache_dir = os.path.join(data_dir, "ghcn_cache")
    os.makedirs(cache_dir, exist_ok=True)

    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)
    today_str = datetime.now().strftime("%Y%m%d")

    def _get_cached_reference_file(prefix: str, url_key: str, line_filter=None):
        """Download a GHCN reference file with a 30-day cache, returning its path."""
        pattern = os.path.join(cache_dir, f"{prefix}_*.txt")
        existing_files = glob.glob(pattern)
        filepath = None

        if existing_files:
            existing_files.sort()
            filepath = existing_files[-1]
            date_str = os.path.basename(filepath).split("_")[-1].split(".")[0]
            try:
                file_date = datetime.strptime(date_str, "%Y%m%d")
                if (datetime.now() - file_date).days > 30:
                    filepath = None  # Stale, force redownload
            except ValueError:
                filepath = None

        if not filepath:
            try:
                url = get_base_url(url_key)
                logger.info(f"Downloading {url}...")
                response = requests.get(url, timeout=60)
                response.raise_for_status()

                text = response.text
                if line_filter is not None:
                    text = "\n".join(l for l in text.splitlines() if line_filter(l))

                new_filepath = os.path.join(cache_dir, f"{prefix}_{today_str}.txt")
                with open(new_filepath, "w") as f:
                    f.write(text)

                # Delete superseded files
                for old_file in existing_files:
                    try:
                        os.remove(old_file)
                    except OSError:
                        pass
                filepath = new_filepath
            except Exception as e:
                if existing_files:
                    logger.debug(f"Failed to download {prefix}. Using stale cache. {e}")
                    filepath = existing_files[-1]
                else:
                    raise e

        return filepath

    def _get_station_list():
        filepath = _get_cached_reference_file("stations", "ghcn_stations")
        col_specs = [(0, 11), (12, 20), (21, 30), (31, 37), (38, 40), (41, 71)]
        names = ["id", "latitude", "longitude", "elevation", "state", "name"]
        df = pd.read_fwf(filepath, colspecs=col_specs, names=names, index_col="id")
        df["elevation"] = pd.to_numeric(df["elevation"], errors="coerce")
        logger.info(f"Loaded {len(df)} GHCN stations from cache/network.")
        return df

    def _get_prcp_inventory():
        """First/last year of PRCP record per station, from ghcnd-inventory.txt."""
        filepath = _get_cached_reference_file(
            "inventory_prcp", "ghcn_inventory", lambda l: l[31:35] == "PRCP"
        )
        col_specs = [(0, 11), (36, 40), (41, 45)]
        names = ["id", "first_year", "last_year"]
        df = pd.read_fwf(filepath, colspecs=col_specs, names=names, index_col="id")
        return df[~df.index.duplicated()]

    def _parse_dly_file(file_content: str) -> pd.Series | None:
        """Optimized high-performance parser bypassing individual pd.to_datetime calls."""
        dates = []
        values = []

        # Avoid repeatedly looking up methods/types in the loop
        pydatetime = datetime
        float_cast = float

        for line in file_content.splitlines():
            if line[17:21] != "PRCP":
                continue

            try:
                year = int(line[11:15])
                month = int(line[15:17])
            except ValueError:
                continue

            idx = 21
            for d in range(1, 32):
                val_str = line[idx : idx + 5]
                idx += 8

                if val_str.strip() == "-9999":
                    continue

                # Fast date validation before object creation
                try:
                    # Leverage standard datetime constructor (10x faster than pd.to_datetime)
                    dt = pydatetime(year, month, d)
                    dates.append(dt)
                    values.append(float_cast(val_str))
                except ValueError:
                    continue  # Invalid days (e.g., Feb 30th)

        if not dates:
            return None

        # Create Series in one batch conversion and sort index
        series = pd.Series(values, index=pd.DatetimeIndex(dates))
        return series.sort_index()

    def _has_sufficient_por(
        filepath: str, req_start: datetime, req_end: datetime
    ) -> bool:
        """Superfast seek-based POR checker that reads only the first and last lines."""
        try:
            with open(filepath, "rb") as f:
                # 1. Grab first line
                first_line = f.readline().decode("utf-8", errors="ignore")
                while first_line and first_line[17:21] != "PRCP":
                    first_line = f.readline().decode("utf-8", errors="ignore")

                if not first_line:
                    return False

                # 2. Seek to end to grab the last line
                f.seek(0, os.SEEK_END)
                file_size = f.tell()

                # Back up 2000 bytes (typical line is ~270 bytes) to find the last PRCP line
                seek_back = min(file_size, 2000)
                f.seek(file_size - seek_back, os.SEEK_SET)
                last_chunk = (
                    f.read(seek_back).decode("utf-8", errors="ignore").splitlines()
                )

                last_line = None
                for line in reversed(last_chunk):
                    if len(line) >= 21 and line[17:21] == "PRCP":
                        last_line = line
                        break

                if not last_line:
                    return False

                # Parse bounds
                first_date = datetime(int(first_line[11:15]), int(first_line[15:17]), 1)
                last_date = datetime(int(last_line[11:15]), int(last_line[15:17]), 1)

                start_month = datetime(req_start.year, req_start.month, 1)
                end_month = datetime(req_end.year, req_end.month, 1)

                return first_date <= start_month and last_date >= end_month

        except Exception as e:
            logger.debug(f"Fast POR check failed for {filepath}: {e}")
            return False

    def _fetch_station_data(station_id: str) -> pd.Series | None:
        pattern = os.path.join(cache_dir, f"{station_id}_*.dly")
        existing_files = glob.glob(pattern)
        cached_file = None
        is_stale = False
        has_por = False
        file_age_days = None
        FRESH_DAYS = 7  # ← skip POR check for anything this new

        if existing_files:
            existing_files.sort()
            cached_file = existing_files[-1]
            date_str = os.path.basename(cached_file).split("_")[1].split(".")[0]
            try:
                file_date = datetime.strptime(date_str, "%Y%m%d")
                file_age_days = (datetime.now() - file_date).days
                is_stale = file_age_days > 30
            except ValueError:
                is_stale = True
                logger.warning(
                    f"{station_id}: could not parse date from {os.path.basename(cached_file)}"
                )

            # Only run the (expensive / often-failing) POR check on older files
            if file_age_days is not None and file_age_days <= FRESH_DAYS:
                has_por = True  # treat fresh files as OK
                logger.debug(
                    f"{station_id}: age={file_age_days}d <= {FRESH_DAYS} - skipping POR check"
                )
            else:
                has_por = _has_sufficient_por(cached_file, start_date, end_date)

            logger.debug(
                f"{station_id}: cache={os.path.basename(cached_file)} | "
                f"age={file_age_days}d | stale={is_stale} | has_POR={has_por}"
            )

        needs_download = (not cached_file) or is_stale or (not has_por)

        if needs_download:
            reason = []
            if not cached_file:
                reason.append("no_cache")
            if is_stale:
                reason.append("stale")
            if not has_por:
                reason.append("insufficient_POR")
            logger.info(f"{station_id}: DOWNLOAD needed - {', '.join(reason)}")

            url = f"{get_base_url('ghcn_daily')}/all/{station_id}.dly"
            try:
                logger.debug(f"{station_id}: GET {url}")
                resp = requests.get(url, timeout=25)
                resp.raise_for_status()
                new_filepath = os.path.join(cache_dir, f"{station_id}_{today_str}.dly")

                with open(new_filepath, "w", encoding="utf-8") as f:
                    f.write(resp.text)
                logger.debug(
                    f"{station_id}: wrote {os.path.basename(new_filepath)} ({len(resp.text)} bytes)"
                )

                # Delete only older files (never the one we just wrote)
                for old_file in existing_files:
                    if os.path.abspath(old_file) != os.path.abspath(new_filepath):
                        try:
                            os.remove(old_file)
                        except OSError as e:
                            logger.warning(
                                f"{station_id}: failed to delete {old_file}: {e}"
                            )

                with open(new_filepath, "r", encoding="utf-8") as f:
                    series = _parse_dly_file(f.read())
                if series is not None:
                    logger.debug(f"{station_id}: parsed {len(series)} PRCP values")
                else:
                    logger.warning(f"{station_id}: parsed 0 PRCP values from new file")
                return series

            except Exception as e:
                if cached_file and os.path.isfile(cached_file):
                    if is_stale and not has_por:
                        raise RuntimeError(
                            f"Download failed for {station_id}, and local cached file is both "
                            "stale (>30 days) and lacks sufficient Period of Record (POR)."
                        ) from e
                    logger.debug(
                        f"{station_id}: download failed ({e}), falling back to "
                        f"{os.path.basename(cached_file)}"
                    )
                    with open(cached_file, "r", encoding="utf-8") as f:
                        return _parse_dly_file(f.read())
                else:
                    logger.debug(
                        f"{station_id}: download failed and no usable cache: {e}"
                    )
                    return None
        else:
            logger.debug(f"{station_id}: using existing cache (fresh + sufficient POR)")
            with open(cached_file, "r", encoding="utf-8") as f:
                series = _parse_dly_file(f.read())
            if series is not None:
                logger.debug(
                    f"{station_id}: parsed {len(series)} PRCP values from cache"
                )
            return series

    # ====================== Primary Station Logic ======================
    def _get_primary_station(stations_with_data: list[dict], end_date: datetime):
        min_antecedent_rows = int(90 * 0.75)
        huge_record = None
        huge_lowest = 10000
        medium_record = None
        medium_lowest = 10000
        minimum_record = None
        minimum_lowest = 10000
        tolerable = 0.75

        for st in stations_with_data:
            act = st.get("actual_rows", 0)
            ant = st.get("antecedent_rows", 0)
            wdiff = st.get("weighted_diff", 10000)

            if ant >= min_antecedent_rows:
                if act > 10000 and wdiff < huge_lowest:
                    huge_lowest, huge_record = wdiff, st
                elif act > 8000 and wdiff < medium_lowest:
                    medium_lowest, medium_record = wdiff, st
                elif act > 6000 and wdiff < minimum_lowest:
                    minimum_lowest, minimum_record = wdiff, st

        best = minimum_record
        if medium_record and (
            not best
            or medium_record["weighted_diff"] < best["weighted_diff"] + 2 * tolerable
        ):
            best = medium_record
        if huge_record and (
            not best
            or huge_record["weighted_diff"] < best["weighted_diff"] + 4 * tolerable
        ):
            if best == minimum_record and medium_record:
                if (
                    huge_record["weighted_diff"]
                    < medium_record["weighted_diff"] + 2 * tolerable
                ):
                    best = huge_record
            else:
                best = huge_record

        if best is None:
            # Fallback: any station with data, sorted by weighted diff
            candidates = [
                s
                for s in stations_with_data
                if s.get("data") is not None and not s["data"].empty
            ]
            if candidates:
                return sorted(candidates, key=lambda x: x.get("weighted_diff", 10000))[
                    0
                ]
            return stations_with_data[0] if stations_with_data else None

        return best

    # ====================== Main Logic ======================
    logger.info(f"Building GHCN composite for ({lat:.4f}, {lon:.4f})")

    target_elev_ft = get_elevation(lat, lon, "feet")

    all_stations = _get_station_list().dropna(
        subset=["latitude", "longitude", "elevation"]
    )
    local_stations = []
    for sid, row in all_stations.iterrows():
        dist = great_circle((lat, lon), (row["latitude"], row["longitude"])).miles
        if dist > 60:
            continue
        station_elev_ft = (row["elevation"] or 0) * 3.28084
        elev_diff = abs((target_elev_ft or 0) - station_elev_ft)
        wdiff = dist * ((elev_diff / 1000) + 0.45)

        local_stations.append(
            {
                **row.to_dict(),
                "id": sid,
                "distance": round(dist, 3),
                "weighted_diff": round(wdiff, 4),
                "elevation_ft": station_elev_ft,
            }
        )
    local_stations = sorted(local_stations, key=lambda x: x["weighted_diff"])

    stations_with_data = []
    norm_start = start_date
    norm_end = datetime(end_date.year - 1, 9, 30)
    ante_start = end_date - timedelta(days=90)
    ante_end = end_date

    # Attach PRCP period of record so stations can be chosen by coverage
    # instead of distance alone. Dense networks (e.g. CoCoRaHS) otherwise
    # crowd every long-record station out of the candidate pool.
    try:
        inventory = _get_prcp_inventory()
    except Exception as e:
        logger.warning(f"GHCN inventory unavailable, selecting by distance only: {e}")
        inventory = None
    for st in local_stations:
        if inventory is not None and st["id"] in inventory.index:
            st["first_year"] = int(inventory.at[st["id"], "first_year"])
            st["last_year"] = int(inventory.at[st["id"], "last_year"])
        elif inventory is not None:
            st["first_year"], st["last_year"] = None, None  # no PRCP record
        else:
            st["first_year"], st["last_year"] = start_date.year, end_date.year
    local_stations = [
        st
        for st in local_stations
        if st["first_year"] is not None
        and st["first_year"] <= end_date.year
        and st["last_year"] >= start_date.year
    ]

    def _covers_years(st, years) -> bool:
        return any(st["first_year"] <= y <= st["last_year"] for y in years)

    def _load(st):
        series = _fetch_station_data(st["id"])
        st["data"] = series
        if series is not None and not series.empty:
            st["actual_rows"] = int(series.loc[norm_start:norm_end].notna().sum())
            st["antecedent_rows"] = int(series.loc[ante_start:ante_end].notna().sum())
        else:
            st["actual_rows"] = 0
            st["antecedent_rows"] = 0

    # Primary candidates: nearest stations whose record could plausibly meet
    # the primary-station thresholds and that report during the antecedent period.
    def _normal_years(st):
        return max(
            0,
            min(st["last_year"], norm_end.year)
            - max(st["first_year"], norm_start.year)
            + 1,
        )

    primary_pool = [
        st
        for st in local_stations
        if _normal_years(st) * 365 > 6000 and st["last_year"] >= ante_start.year
    ][:MAX_PRIMARY_CANDIDATES]
    if not primary_pool:
        primary_pool = local_stations[:MAX_PRIMARY_CANDIDATES]
    logger.info(f"Fetching data for {len(primary_pool)} primary candidates...")

    for st in primary_pool:
        _load(st)
        stations_with_data.append(st)

    primary = _get_primary_station(stations_with_data, end_date)
    if not primary:
        logger.error("No usable primary station found.")
        return None, [], target_elev_ft or -1

    logger.info(f"Selected primary: {primary['id']} / {primary.get('name')}")

    # Re-sort others relative to primary
    other_stations = [s for s in local_stations if s["id"] != primary["id"]]
    for st in other_stations:
        dist = great_circle(
            (primary["latitude"], primary["longitude"]),
            (st["latitude"], st["longitude"]),
        ).miles
        elev_diff = abs(primary.get("elevation_ft", 0) - st.get("elevation_ft", 0))
        st["weighted_diff"] = round(dist * ((elev_diff / 1000) + 0.45), 4)

    sorted_stations = [primary] + sorted(
        other_stations, key=lambda x: x["weighted_diff"]
    )

    # === COMPOSITE BUILDING ===
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    composite = pd.Series(index=date_range, dtype="float64")
    final_used = []
    extra_fetches = 0

    for i, st in enumerate(sorted_stations):
        missing = composite.isna()
        if not missing.any():
            break
        # Skip stations whose record cannot fill any of the remaining gaps
        if i > 0 and not _covers_years(st, set(composite.index[missing].year)):
            continue
        if "data" not in st:
            if extra_fetches >= MAX_FILL_FETCHES:
                logger.warning(
                    f"Reached {MAX_FILL_FETCHES} additional station downloads "
                    "with gaps remaining."
                )
                break
            _load(st)
            extra_fetches += 1
        data = st.get("data")
        if data is not None and not data.empty:
            before = composite.isna().sum()
            # Reindex to ensure alignment
            station_data = data.reindex(date_range)
            composite = composite.fillna(station_data)
            filled = before - composite.isna().sum()

            if filled > 0 or i == 0:  # Always keep primary + any that contributed
                st_info = {k: v for k, v in st.items() if k not in ["data"]}
                st_info.update(
                    {
                        "days_normal": st.get("actual_rows", 0),  # Historical coverage
                        "days_antecedent": st.get("antecedent_rows", 0),
                    }
                )
                final_used.append(st_info)

    # Bridge single-day gaps only (as v3.0.9 did). Longer gaps stay NaN so they
    # are excluded from the normals rather than counted as zero precipitation.
    composite = composite.interpolate(method="time", limit=1, limit_area="inside")

    remaining = composite.isna()
    if remaining.any():
        logger.warning(
            f"{int(remaining.sum())} days have no station data after filling "
            f"({int(remaining[norm_start:norm_end].sum())} in the normal period, "
            f"{int(remaining[ante_start:ante_end].sum())} in the antecedent period)."
        )

    logger.info(f"Composite built with {len(final_used)} stations")
    return composite, final_used, target_elev_ft or -1.0


# ====================== Gridded (unchanged) ======================
def build_composite_precip_series_gridded(lat, lon, start_date, end_date):
    try:
        from core.gridded_downloader import (
            build_composite_precip_series_gridded as impl,
        )
    except ImportError:
        from gridded_downloader import build_composite_precip_series_gridded as impl
    return impl(lat, lon, start_date, end_date)


# ====================== Public API ======================
def build_composite_precip_series(
    lat: float,
    lon: float,
    start_date: datetime,
    end_date: datetime,
    use_gridded: bool = False,
    data_dir: str = ".",
):
    logger.info(f"build_composite_precip_series called - gridded={use_gridded}")

    if use_gridded:
        series, stations = build_composite_precip_series_gridded(
            lat, lon, start_date, end_date
        )
        return series, stations, -1.0

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    series, stations_info, elevation = get_ghcn_timeseries_data(
        lat, lon, start_str, end_str, data_dir
    )

    if series is None:
        series = pd.Series(dtype="float64")
        stations_info = []

    # Ensure full date range; days without data stay NaN (never zero-filled)
    full_range = pd.date_range(start=start_date, end=end_date, freq="D")
    series = series.reindex(full_range)

    return (
        series,
        stations_info or [],
        float(elevation) if elevation is not None else -1.0,
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python precip_downloader.py <path_to_data_dir>")
    else:
        # Standardize and sanitize directory paths for Windows/Linux
        data_directory = os.path.abspath(sys.argv[1])
        end_date = datetime.strptime("2023-11-11", "%Y-%m-%d")
        start_date = end_date - timedelta(days=365 * 30)

        series, info, elev = build_composite_precip_series(
            30.0, -90.0, start_date, end_date, data_dir=data_directory
        )
        print(f"GHCN: {len(series)} days, elevation={elev:.1f}")
        print(series.head(5))
        print("...")
        print(series.tail(5))
