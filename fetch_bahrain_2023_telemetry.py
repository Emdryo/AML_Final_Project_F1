"""
Fetch all car telemetry data for the 2023 Bahrain Grand Prix (Race session).

OpenF1 API: https://openf1.org/docs
- Meeting: 2023 Bahrain Grand Prix  (meeting_key = 1141)
- Session: Race                     (session_key = 9158)
- Endpoint: /v1/car_data            (3.7 Hz, ~20 drivers)

The free tier rate limit is 3 req/s and 30 req/min.
This script fetches one driver at a time with a small delay to stay well
within those limits, then combines everything into a single CSV file.
"""

import time
import csv
import sys

try:
    import requests
except ImportError:
    sys.exit("Install requests first:  pip install requests")

# ── Config ────────────────────────────────────────────────────────────────────

SESSION_KEY  = 9158        # 2023 Bahrain GP – Race
MEETING_KEY  = 1141        # 2023 Bahrain GP
BASE_URL     = "https://api.openf1.org/v1"
OUTPUT_FILE  = "bahrain_2023_race_telemetry.csv"
DELAY_S      = 0.4         # seconds between requests (well under 3 req/s limit)

# ── Helpers ───────────────────────────────────────────────────────────────────

def fetch_json(url: str, retries: int = 3) -> list:
    """GET a URL, return parsed JSON. Retries on transient errors."""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 429:
                print(f"  Rate limited (429). Waiting 60s …", flush=True)
                time.sleep(60)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            print(f"  Request error on attempt {attempt}: {e}", flush=True)
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)
    return []


def get_drivers(session_key: int) -> list[dict]:
    """Return all drivers who took part in the session."""
    url = f"{BASE_URL}/drivers?session_key={session_key}"
    return fetch_json(url)


def get_telemetry_for_driver(session_key: int, driver_number: int) -> list[dict]:
    """Return all car_data rows for one driver in a session."""
    url = f"{BASE_URL}/car_data?session_key={session_key}&driver_number={driver_number}"
    return fetch_json(url)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("OpenF1 – 2023 Bahrain GP Race telemetry downloader")
    print("=" * 60)

    # 1. Get the driver list for this session
    print(f"\n[1/3] Fetching driver list for session {SESSION_KEY} …", flush=True)
    drivers = get_drivers(SESSION_KEY)
    if not drivers:
        print("ERROR: no drivers returned. Check session_key.", file=sys.stderr)
        sys.exit(1)

    driver_map = {d["driver_number"]: d for d in drivers}
    driver_numbers = sorted(driver_map.keys())
    print(f"      Found {len(driver_numbers)} drivers: "
          f"{', '.join(str(n) for n in driver_numbers)}")

    # 2. Fetch telemetry per driver
    print(f"\n[2/3] Fetching car telemetry ({len(driver_numbers)} drivers) …", flush=True)

    all_rows: list[dict] = []
    for i, num in enumerate(driver_numbers, start=1):
        name = driver_map[num].get("full_name", f"#{num}")
        team = driver_map[num].get("team_name", "")
        print(f"  [{i:2d}/{len(driver_numbers)}] {name:30s} ({team}) … ", end="", flush=True)

        rows = get_telemetry_for_driver(SESSION_KEY, num)

        # Enrich each row with driver meta for convenience
        for row in rows:
            row["full_name"]   = name
            row["name_acronym"] = driver_map[num].get("name_acronym", "")
            row["team_name"]   = team

        all_rows.extend(rows)
        print(f"{len(rows):,} rows", flush=True)

        if i < len(driver_numbers):
            time.sleep(DELAY_S)

    print(f"\n  Total rows collected: {len(all_rows):,}")

    if not all_rows:
        print("ERROR: no data collected.", file=sys.stderr)
        sys.exit(1)

    # 3. Write to CSV
    print(f"\n[3/3] Writing to {OUTPUT_FILE} …", flush=True)

    # Canonical column order (OpenF1 fields first, then enrichment)
    fieldnames = [
        "date", "driver_number", "full_name", "name_acronym", "team_name",
        "session_key", "meeting_key",
        "speed", "rpm", "n_gear", "throttle", "brake", "drs",
    ]
    # Add any extra fields the API might return that aren't in our list
    extra = [k for k in all_rows[0].keys() if k not in fieldnames]
    fieldnames += extra

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"  Done. Saved {len(all_rows):,} rows → {OUTPUT_FILE}")
    print("\nColumns written:")
    for col in fieldnames:
        print(f"  • {col}")
    print("\nDone! 🏁")


if __name__ == "__main__":
    main()