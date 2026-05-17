"""
Script 2 of 2 — Fetch any OpenF1 endpoint for a session → save as HDF5
========================================================================
Requires sessions.json produced by f1_fetch_sessions.py.

Usage:
    pip install requests pandas tables

    # Fetch car telemetry for session 9158 (2023 Bahrain Race)
    python f1_fetch_data.py --session 9158 --endpoint car_data

    # Fetch lap data for session 9158
    python f1_fetch_data.py --session 9158 --endpoint laps

    # Fetch weather for a session — no per-driver loop needed
    python f1_fetch_data.py --session 9158 --endpoint weather

    # List all available endpoints
    python f1_fetch_data.py --list-endpoints

    # Search sessions (e.g. find all 2023 races)
    python f1_fetch_data.py --search "2023 Race"

Available endpoints (all supported by this script):
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

BASE_URL       = "https://api.openf1.org/v1"
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
SESSIONS_FILE  = os.path.join(SCRIPT_DIR, "sessions.json")
OUTPUT_DIR     = os.path.join(SCRIPT_DIR, "f1_data")
DELAY_S        = 0.4     # between requests

# Endpoints that have one row per driver per timepoint → fetch per-driver
PER_DRIVER_ENDPOINTS = {
    "car_data", "location", "intervals", "position",
}

# Endpoints that return session-level data (one request is enough)
SESSION_LEVEL_ENDPOINTS = {
    "laps", "weather", "pit", "stints", "race_control",
    "session_result", "starting_grid", "drivers",
    "overtakes", "championship_drivers", "championship_teams",
}

ALL_ENDPOINTS = PER_DRIVER_ENDPOINTS | SESSION_LEVEL_ENDPOINTS

# HDF5 dtypes: columns that should be stored as categories to save space
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


def to_hdf5(df: pd.DataFrame, path: str, key: str = "data") -> None:
    """Save DataFrame to HDF5 with compression. Categorical columns save space."""
    for col in df.columns:
        if col in CATEGORY_COLS and df[col].dtype == object:
            df[col] = df[col].astype("category")
        # Parse ISO timestamps
        if "date" in col.lower() and df[col].dtype == object:
            try:
                df[col] = pd.to_datetime(df[col], utc=True)
            except Exception:
                pass

    df.to_hdf(
        path,
        key=key,
        mode="w",
        complevel=9,          # max zlib compression
        complib="blosc:zstd", # fast + good ratio; falls back to zlib if unavailable
        format="table",       # queryable; use "fixed" if you hit dtype issues
    )


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
    matches = [
        s for s in sessions
        if q in s["country"].lower()
        or q in s["label"].lower()
        or q in s["session_name"].lower()
        or q in str(s["year"])
        or q in str(s["session_key"])
    ]
    if not matches:
        print(f"No sessions matching '{query}'.")
        return
    print(f"\n{'Key':>7}  {'Date':10}  {'Year':5}  {'Country':20}  {'Session'}")
    print("─" * 75)
    for s in matches:
        print(f"{s['session_key']:>7}  {s['date']:10}  {s['year']:5}  "
              f"{s['country']:20}  {s['label']}")
    print(f"\n{len(matches)} match(es).")


def cmd_fetch(sessions: list[dict], session_key: int, endpoint: str):
    if endpoint not in ALL_ENDPOINTS:
        sys.exit(
            f"ERROR: Unknown endpoint '{endpoint}'.\n"
            "Run with --list-endpoints to see all options."
        )

    info = get_session_info(sessions, session_key)
    if info is None:
        sys.exit(
            f"ERROR: Session key {session_key} not found in {SESSIONS_FILE}.\n"
            "Run f1_fetch_sessions.py again or check the key."
        )

    label   = info["label"]
    country = info["country"]
    date    = info["date"]
    year    = info["year"]

    print("=" * 65)
    print(f"OpenF1 data fetcher")
    print(f"  Session : {session_key} — {country} {label} ({date})")
    print(f"  Endpoint: {endpoint}")
    print("=" * 65)

    all_rows: list[dict] = []

    if endpoint in PER_DRIVER_ENDPOINTS:
        # Fetch per driver
        print(f"\n[1/2] Getting driver list …", flush=True)
        driver_numbers = get_drivers(session_key)
        print(f"      {len(driver_numbers)} drivers: "
              f"{', '.join(str(n) for n in driver_numbers)}")

        print(f"\n[2/2] Fetching {endpoint} per driver …", flush=True)
        for i, num in enumerate(driver_numbers, 1):
            url = f"{BASE_URL}/{endpoint}?session_key={session_key}&driver_number={num}"
            print(f"  [{i:2d}/{len(driver_numbers)}] driver #{num:3d} … ", end="", flush=True)
            rows = fetch_json(url)
            print(f"{len(rows):,} rows")
            all_rows.extend(rows)
            if i < len(driver_numbers):
                time.sleep(DELAY_S)
    else:
        # Single request for the whole session
        print(f"\nFetching {endpoint} for session {session_key} …", flush=True)
        url = f"{BASE_URL}/{endpoint}?session_key={session_key}"
        all_rows = fetch_json(url)
        print(f"  {len(all_rows):,} rows returned")

    if not all_rows:
        print("\nWARNING: No data returned. The endpoint may not have data for this session.")
        return

    df = pd.DataFrame(all_rows)
    print(f"\nTotal rows: {len(df):,}  |  Columns: {list(df.columns)}")

    # Build output path:  f1_data/9158_car_data_2023_Bahrain_Race.h5
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    safe_country = country.replace(" ", "_")
    safe_label   = label.replace(" ", "_").replace("(", "").replace(")", "")
    filename = f"{session_key}_{endpoint}_{year}_{safe_country}_{safe_label}.h5"
    out_path = os.path.join(OUTPUT_DIR, filename)

    print(f"\nSaving to {out_path} …", flush=True)
    try:
        to_hdf5(df, out_path, key=endpoint)
    except Exception as e:
        # blosc may not be available everywhere; fall back to zlib
        print(f"  (blosc unavailable, falling back to zlib: {e})")
        df.to_hdf(out_path, key=endpoint, mode="w", complevel=9, complib="zlib",
                  format="table")

    size_mb = os.path.getsize(out_path) / 1_048_576
    print(f"Done. File size: {size_mb:.2f} MB  →  {out_path}")

    # Quick preview
    print(f"\nFirst 3 rows:\n{df.head(3).to_string(index=False)}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fetch OpenF1 data for a session and save as HDF5.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--session",          type=int,   help="Session key (from sessions.json)")
    parser.add_argument("--endpoint",         type=str,   help="API endpoint name (e.g. car_data)")
    parser.add_argument("--list-endpoints",   action="store_true", help="List all available endpoints")
    parser.add_argument("--search",           type=str,   help="Search sessions by keyword")
    args = parser.parse_args()

    if args.list_endpoints:
        cmd_list_endpoints()
        return

    sessions = load_sessions()

    if args.search:
        cmd_search(sessions, args.search)
        return

    if not args.session or not args.endpoint:
        parser.print_help()
        print("\nTip: run --list-endpoints or --search to explore available data.")
        return

    cmd_fetch(sessions, args.session, args.endpoint)


if __name__ == "__main__":
    main()