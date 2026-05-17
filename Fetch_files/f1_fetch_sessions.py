"""
Script 1 of 2 — Fetch & save all OpenF1 session keys (2023–2025)
=================================================================
Run this once to build a local sessions index and driver lookup table.
Outputs:
  sessions.json   — all sessions with keys, dates, circuit info
  drivers.json    — lookup: (year, driver_number) → name, acronym, team

Usage:
    pip install requests
    python f1_fetch_sessions.py
"""

import json
import time
import os
import requests

BASE_URL     = "https://api.openf1.org/v1"
YEARS        = [2023, 2024, 2025]
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
OUT_SESSIONS = os.path.join(SCRIPT_DIR, "sessions.json")
OUT_DRIVERS  = os.path.join(SCRIPT_DIR, "drivers.json")

# Session type → short readable label
SESSION_TYPE_LABEL = {
    "Practice":           "Practice",
    "Qualifying":         "Qualifying",
    "Sprint":             "Sprint Race",
    "Sprint Qualifying":  "Sprint Qualifying",
    "Sprint Shootout":    "Sprint Shootout",
    "Race":               "Grand Prix (Race)",
}


def fetch(url: str, retries: int = 3) -> list:
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 429:
                print("  Rate limited — waiting 60 s …")
                time.sleep(60)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            print(f"  Attempt {attempt} failed: {e}")
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)
    return []


# ── Sessions ──────────────────────────────────────────────────────────────────

def fetch_sessions() -> list[dict]:
    print("=" * 60)
    print("Step 1/2 — Fetching sessions")
    print("=" * 60)
    all_sessions = []
    for year in YEARS:
        print(f"  Year {year} …", end=" ", flush=True)
        sessions = fetch(f"{BASE_URL}/sessions?year={year}")
        sessions = [s for s in sessions if not s.get("is_cancelled", False)]
        all_sessions.extend(sessions)
        print(f"{len(sessions)} sessions")
        time.sleep(0.4)

    all_sessions.sort(key=lambda s: s.get("date_start", ""))

    index = []
    for s in all_sessions:
        stype  = s.get("session_type", s.get("session_name", "Unknown"))
        label  = SESSION_TYPE_LABEL.get(stype, stype)
        date   = s.get("date_start", "")[:10]
        index.append({
            "session_key":  s["session_key"],
            "meeting_key":  s.get("meeting_key"),
            "year":         s.get("year", ""),
            "date":         date,
            "country":      s.get("country_name", ""),
            "circuit":      s.get("circuit_short_name", ""),
            "session_name": s.get("session_name", ""),
            "session_type": stype,
            "label":        label,
            "date_start":   s.get("date_start", ""),
            "date_end":     s.get("date_end", ""),
        })

    with open(OUT_SESSIONS, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(index)} sessions → {OUT_SESSIONS}")
    return index


# ── Drivers ───────────────────────────────────────────────────────────────────

def fetch_drivers(sessions: list[dict]) -> dict:
    """
    Build a cross-season driver lookup table saved to drivers.json.

    Strategy: use one session per meeting (preferring Race) to get each
    meeting's driver roster. This captures both regulars and substitutes
    while keeping the number of API requests minimal.

    Output structure:
        {
          "by_year_and_number": {
            "2023": { "1": { driver info }, "44": { ... }, ... },
            "2024": { ... },
            ...
          },
          "by_acronym": {
            "VER": { full_name, team_name, years_active: [2023, 2024, 2025], ... },
            ...
          }
        }

    Usage in your ML code:
        import json
        drivers = json.load(open("drivers.json"))

        # Name from year + number (safe cross-season)
        drivers["by_year_and_number"]["2023"]["44"]["full_name"]   # → "Lewis HAMILTON"

        # Look up a driver across all seasons by acronym
        drivers["by_acronym"]["VER"]["years_active"]               # → [2023, 2024, 2025]
    """
    print("\n" + "=" * 60)
    print("Step 2/2 — Building driver lookup table")
    print("=" * 60)
    print("  (one request per meeting — preferring Race session)\n")

    # Pick one session per meeting, preferring Race for the definitive roster
    meetings_seen: dict[int, dict] = {}
    for s in sessions:
        mk = s["meeting_key"]
        if mk not in meetings_seen:
            meetings_seen[mk] = s
        elif s["session_type"] == "Race":
            meetings_seen[mk] = s

    by_year_and_number: dict[str, dict[str, dict]] = {}
    by_acronym: dict[str, dict] = {}

    meeting_list = sorted(meetings_seen.values(), key=lambda s: s["date_start"])
    total = len(meeting_list)

    for i, s in enumerate(meeting_list, 1):
        sk   = s["session_key"]
        year = str(s["year"])
        print(f"  [{i:3d}/{total}] {s['date']:10}  {s['country']:20}  "
              f"{s['label']:25} (key {sk}) … ", end="", flush=True)

        drivers = fetch(f"{BASE_URL}/drivers?session_key={sk}")
        print(f"{len(drivers)} drivers")

        if year not in by_year_and_number:
            by_year_and_number[year] = {}

        for d in drivers:
            num     = str(d.get("driver_number", ""))
            acronym = d.get("name_acronym", "")
            entry   = {
                "driver_number":  d.get("driver_number"),
                "full_name":      d.get("full_name", ""),
                "first_name":     d.get("first_name", ""),
                "last_name":      d.get("last_name", ""),
                "name_acronym":   acronym,
                "team_name":      d.get("team_name", ""),
                "team_colour":    d.get("team_colour", ""),
                "broadcast_name": d.get("broadcast_name", ""),
            }
            # year+number lookup — most recent team name wins if a driver
            # changes teams mid-season (very rare but possible)
            by_year_and_number[year][num] = entry

            # acronym lookup — accumulate years active
            if acronym:
                if acronym not in by_acronym:
                    by_acronym[acronym] = {**entry, "years_active": []}
                yr = int(year)
                if yr not in by_acronym[acronym]["years_active"]:
                    by_acronym[acronym]["years_active"].append(yr)

        time.sleep(0.4)

    lookup = {
        "by_year_and_number": by_year_and_number,
        "by_acronym":         by_acronym,
    }

    with open(OUT_DRIVERS, "w", encoding="utf-8") as f:
        json.dump(lookup, f, indent=2, ensure_ascii=False)

    total_entries = sum(len(v) for v in by_year_and_number.values())
    print(f"\nSaved {total_entries} driver-year entries "
          f"({len(by_acronym)} unique drivers) → {OUT_DRIVERS}")
    return lookup


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(sessions: list[dict], drivers: dict):
    print("\n" + "─" * 82)
    print(f"{'Key':>7}  {'Date':10}  {'Year':5}  {'Country':20}  {'Session'}")
    print("─" * 82)
    for s in sessions:
        print(f"{s['session_key']:>7}  {s['date']:10}  {s['year']:5}  "
              f"{s['country']:20}  {s['label']}")
    print("─" * 82)
    print(f"\n{len(sessions)} sessions  |  "
          f"{len(drivers['by_acronym'])} unique drivers across {len(YEARS)} seasons")
    print(f"\nFiles written:")
    print(f"  {OUT_SESSIONS}")
    print(f"  {OUT_DRIVERS}")
    print("\nQuick reference — load drivers in your ML code:")
    print("  import json")
    print('  drivers = json.load(open("drivers.json"))')
    print('  drivers["by_year_and_number"]["2023"]["44"]["full_name"]  # Lewis HAMILTON')
    print('  drivers["by_acronym"]["VER"]["years_active"]              # [2023, 2024, 2025]')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    sessions = fetch_sessions()
    drivers  = fetch_drivers(sessions)
    print_summary(sessions, drivers)


if __name__ == "__main__":
    main()