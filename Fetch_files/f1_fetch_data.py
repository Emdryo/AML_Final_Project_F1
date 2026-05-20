"""
Script 2 of 2 — Fetch any OpenF1 endpoint for one or all sessions → save as HDF5
==================================================================================
Requires sessions.json produced by f1_fetch_sessions.py.

Usage:
    pip install requests pandas tables

    # Fetch car telemetry for one session
    python3 f1_fetch_data.py --session 9158 --endpoint car_data

    # Fetch car telemetry for ALL sessions (skips already-downloaded files)
    python3 f1_fetch_data.py --endpoint car_data --all

    # Fetch ALL sessions but only Race sessions
    python3 f1_fetch_data.py --endpoint laps --all --session-type Race

    # Fetch ALL sessions for a specific year
    python3 f1_fetch_data.py --endpoint weather --all --year 2023

    # Combine filters: only 2024 Race sessions
    python3 f1_fetch_data.py --endpoint car_data --all --year 2024 --session-type Race

    # List all available endpoints
    python3 f1_fetch_data.py --list-endpoints

    # Search sessions (e.g. find all 2023 races)
    python3 f1_fetch_data.py --search "2023 Race"

Available endpoints:
    car_data, laps, weather, location, intervals, position,
    pit, stints, race_control, session_result, starting_grid,
    drivers, overtakes, championship_drivers, championship_teams
"""

import argparse
import json
import time
import sys
import os
import requests
import pandas as pd

BASE_URL      = "https://api.openf1.org/v1"
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
SESSIONS_FILE = os.path.join(SCRIPT_DIR, "sessions.json")
OUTPUT_DIR    = os.path.join(SCRIPT_DIR, "f1_data")
DELAY_S       = 0.4   # seconds between requests

# Endpoints that must be fetched per driver (too large for one request)
PER_DRIVER_ENDPOINTS = {
    "car_data", "location", "intervals", "position",
}

# Endpoints where one request returns all session data
SESSION_LEVEL_ENDPOINTS = {
    "laps", "weather", "pit", "stints", "race_control",
    "session_result", "starting_grid", "drivers",
    "overtakes", "championship_drivers", "championship_teams",
}

ALL_ENDPOINTS = PER_DRIVER_ENDPOINTS | SESSION_LEVEL_ENDPOINTS

# Columns stored as HDF5 categories to save space
CATEGORY_COLS = {"compound", "flag", "scope", "category", "session_type",
                 "team_name", "name_acronym", "country_code"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_sessions() -> list[dict]:
    if not os.path.exists(SESSIONS_FILE):
        sys.exit(
            f"ERROR: {SESSIONS_FILE} not found.\n"
            "Run f1_fetch_sessions.py first to build the session index."
        )
    with open(SESSIONS_FILE, encoding="utf-8") as f:
        return json.load(f)


def get_session_info(sessions: list[dict], session_key: int) -> dict | None:
    for s in sessions:
        if s["session_key"] == session_key:
            return s
    return None


def fetch_json(url: str, retries: int = 3) -> list:
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 429:
                print("  Rate limited — waiting 60 s …", flush=True)
                time.sleep(60)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            print(f"  Attempt {attempt} failed: {e}", flush=True)
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)
    return []


def get_drivers(session_key: int) -> list[int]:
    data = fetch_json(f"{BASE_URL}/drivers?session_key={session_key}")
    return sorted(d["driver_number"] for d in data)


def make_output_path(session_info: dict, endpoint: str) -> str:
    """
    Build the full output path with subfolders:
        f1_data/{endpoint}/{year}/{session_key}_{country}_{label}.h5
    """
    country = session_info["country"].replace(" ", "_")
    label   = (session_info["label"]
               .replace(" ", "_")
               .replace("(", "")
               .replace(")", ""))
    folder   = os.path.join(OUTPUT_DIR, endpoint, str(session_info["year"]))
    filename = f"{session_info['session_key']}_{country}_{label}.h5"
    return os.path.join(folder, filename)


def to_hdf5(df: pd.DataFrame, path: str, key: str) -> None:
    df = df.copy()
    for col in df.columns:
        # Serialize list/dict columns (e.g. segments_sector_*) to JSON strings.
        # HDF5 table format cannot store mixed/list dtypes directly.
        # To restore on load: df[col] = df[col].apply(json.loads)
        if df[col].dtype == object:
            first_valid = df[col].dropna().iloc[0] if df[col].notna().any() else None
            if isinstance(first_valid, (list, dict)):
                df[col] = df[col].apply(lambda x: json.dumps(x) if x is not None else None)
        if col in CATEGORY_COLS and df[col].dtype == object:
            df[col] = df[col].astype("category")
        if "date" in col.lower() and df[col].dtype == object:
            try:
                df[col] = pd.to_datetime(df[col], utc=True)
            except Exception:
                pass
    try:
        df.to_hdf(path, key=key, mode="w", complevel=9,
                  complib="blosc:zstd", format="table")
    except Exception:
        df.to_hdf(path, key=key, mode="w", complevel=9,
                  complib="zlib", format="table")


# ── Fetch one session ─────────────────────────────────────────────────────────

def fetch_session(info: dict, endpoint: str, show_header: bool = True) -> str | None:
    """
    Fetch `endpoint` data for one session and save to HDF5.
    Returns the output path on success, None if no data.
    """
    out_path = make_output_path(info, endpoint)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if show_header:
        print(f"\n{'=' * 65}")
        print(f"  Session : {info['session_key']} — "
              f"{info['country']} {info['label']} ({info['date']})")
        print(f"  Endpoint: {endpoint}")
        print(f"{'=' * 65}")

    all_rows: list[dict] = []

    if endpoint in PER_DRIVER_ENDPOINTS:
        driver_numbers = get_drivers(info["session_key"])
        if show_header:
            print(f"  Drivers : {', '.join(str(n) for n in driver_numbers)}")
        for i, num in enumerate(driver_numbers, 1):
            url = (f"{BASE_URL}/{endpoint}"
                   f"?session_key={info['session_key']}&driver_number={num}")
            print(f"    [{i:2d}/{len(driver_numbers)}] driver #{num:3d} … ",
                  end="", flush=True)
            rows = fetch_json(url)
            print(f"{len(rows):,} rows")
            all_rows.extend(rows)
            if i < len(driver_numbers):
                time.sleep(DELAY_S)
    else:
        url = f"{BASE_URL}/{endpoint}?session_key={info['session_key']}"
        print(f"  Fetching … ", end="", flush=True)
        all_rows = fetch_json(url)
        print(f"{len(all_rows):,} rows")

    if not all_rows:
        print("  WARNING: no data returned — skipping.")
        return None

    df = pd.DataFrame(all_rows)
    to_hdf5(df, out_path, key=endpoint)

    size_mb = os.path.getsize(out_path) / 1_048_576
    print(f"  Saved → {os.path.basename(out_path)}  ({size_mb:.2f} MB)")
    return out_path


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_list_endpoints():
    print("\nPer-driver endpoints (fetched once per driver per session):")
    for e in sorted(PER_DRIVER_ENDPOINTS):
        print(f"  {e}")
    print("\nSession-level endpoints (one request per session):")
    for e in sorted(SESSION_LEVEL_ENDPOINTS):
        print(f"  {e}")


def cmd_search(sessions: list[dict], query: str):
    q = query.lower()
    matches = [s for s in sessions
               if q in s["country"].lower()
               or q in s["label"].lower()
               or q in s["session_name"].lower()
               or q in str(s["year"])
               or q in str(s["session_key"])]
    if not matches:
        print(f"No sessions matching '{query}'.")
        return
    print(f"\n{'Key':>7}  {'Date':10}  {'Year':5}  {'Country':20}  {'Session'}")
    print("─" * 75)
    for s in matches:
        print(f"{s['session_key']:>7}  {s['date']:10}  {s['year']:5}  "
              f"{s['country']:20}  {s['label']}")
    print(f"\n{len(matches)} match(es).")


def cmd_fetch_one(sessions: list[dict], session_key: int, endpoint: str):
    if endpoint not in ALL_ENDPOINTS:
        sys.exit(f"ERROR: Unknown endpoint '{endpoint}'. "
                 "Run --list-endpoints to see options.")
    info = get_session_info(sessions, session_key)
    if info is None:
        sys.exit(f"ERROR: Session key {session_key} not found in {SESSIONS_FILE}.")
    fetch_session(info, endpoint, show_header=True)


def cmd_fetch_all(sessions: list[dict], endpoint: str,
                  year_filter: int | None, type_filter: str | None,
                  skip_existing: bool):
    if endpoint not in ALL_ENDPOINTS:
        sys.exit(f"ERROR: Unknown endpoint '{endpoint}'. "
                 "Run --list-endpoints to see options.")

    # Apply filters
    targets = sessions
    if year_filter:
        targets = [s for s in targets if s["year"] == year_filter]
    if type_filter:
        targets = [s for s in targets
                   if type_filter.lower() in s["session_type"].lower()
                   or type_filter.lower() in s["session_name"].lower()]

    if not targets:
        print("No sessions match the given filters.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n{'=' * 65}")
    print(f"  Bulk fetch: {endpoint}")
    filters = []
    if year_filter: filters.append(f"year={year_filter}")
    if type_filter: filters.append(f"type={type_filter}")
    print(f"  Filters   : {', '.join(filters) if filters else 'none (all sessions)'}")
    print(f"  Sessions  : {len(targets)}")
    print(f"  Skip exist: {skip_existing}")
    print(f"{'=' * 65}\n")

    done, skipped, failed, empty = 0, 0, 0, 0

    for i, info in enumerate(targets, 1):
        out_path = make_output_path(info, endpoint)

        print(f"[{i:3d}/{len(targets)}] {info['date']}  "
              f"{info['country']:20}  {info['label']:25} (key {info['session_key']})")

        if skip_existing and os.path.exists(out_path):
            print(f"         → already exists, skipping.")
            skipped += 1
            continue

        try:
            result = fetch_session(info, endpoint, show_header=False)
            if result:
                done += 1
            else:
                empty += 1
        except Exception as e:
            print(f"         ERROR: {e}")
            failed += 1

        # Small pause between sessions to be polite to the API
        if i < len(targets):
            time.sleep(DELAY_S)

    # Summary
    print(f"\n{'─' * 65}")
    print(f"Bulk fetch complete.")
    print(f"  Saved  : {done}")
    print(f"  Skipped: {skipped}  (already existed)")
    print(f"  Empty  : {empty}   (no data for that session)")
    print(f"  Failed : {failed}")
    print(f"  Output : {OUTPUT_DIR}/")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fetch OpenF1 data for one or all sessions → HDF5.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--session",       type=int,  help="Single session key")
    parser.add_argument("--endpoint",      type=str,  help="API endpoint (e.g. car_data)")
    parser.add_argument("--all",           action="store_true",
                        help="Fetch all sessions (optionally filtered)")
    parser.add_argument("--year",          type=int,  help="Filter to a specific year (e.g. 2023)")
    parser.add_argument("--session-type",  type=str,  dest="session_type",
                        help="Filter by session type (e.g. Race, Practice, Qualifying, Sprint)")
    parser.add_argument("--no-skip",       action="store_true",
                        help="Re-download even if the file already exists")
    parser.add_argument("--list-endpoints", action="store_true")
    parser.add_argument("--search",        type=str)
    args = parser.parse_args()

    if args.list_endpoints:
        cmd_list_endpoints()
        return

    sessions = load_sessions()

    if args.search:
        cmd_search(sessions, args.search)
        return

    if not args.endpoint:
        parser.print_help()
        print("\nTip: use --list-endpoints or --search to explore.")
        return

    if args.all:
        cmd_fetch_all(
            sessions,
            endpoint     = args.endpoint,
            year_filter  = args.year,
            type_filter  = args.session_type,
            skip_existing= not args.no_skip,
        )
    elif args.session:
        cmd_fetch_one(sessions, args.session, args.endpoint)
    else:
        parser.print_help()
        print("\nProvide --session KEY for one session, or --all for all sessions.")


if __name__ == "__main__":
    main()