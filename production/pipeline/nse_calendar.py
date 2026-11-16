"""Authoritative, versioned NSE sessions, including exceptional weekend sessions."""

import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


class CalendarUnavailable(ValueError):
    pass


class NSECalendar:
    def __init__(self, directory=None):
        self.directory = Path(directory or Path(__file__).with_name("calendars"))
        self.years = {}

    def year(self, year):
        if year not in self.years:
            path = self.directory / f"{year}.json"
            if not path.is_file():
                raise CalendarUnavailable(f"NSE calendar for {year} requires published exchange dates")
            data = json.loads(path.read_text(encoding="utf-8"))
            if data["year"] != year or data["timezone"] != "Asia/Kolkata":
                raise CalendarUnavailable("NSE calendar identity is invalid")
            self.years[year] = data
        return self.years[year]

    def is_session(self, day):
        data = self.year(day.year)
        if str(day) in data.get("excluded_sessions", {}):
            return False
        return str(day) in data["special_sessions"] or (day.weekday() < 5 and str(day) not in data["holidays"])

    def exclusions(self, start, end):
        """Explicit production exclusions; exchange-session metadata is retained."""
        return sorted(day for year in range(start.year, end.year + 1)
                      if (self.directory / f"{year}.json").is_file()
                      for value in self.year(year).get("excluded_sessions", {})
                      if start <= (day := date.fromisoformat(value)) <= end)

    def request_windows(self, expected):
        """Split provider ranges so an excluded day is never requested."""
        expected = sorted(expected)
        if not expected:
            raise ValueError("Provider windows require planned sessions")
        excluded = self.exclusions(expected[0], expected[-1])
        if set(expected).intersection(excluded):
            raise ValueError("Excluded sessions cannot be requested")
        windows, current = [], []
        for day in expected:
            if current and any(current[-1] < omitted < day for omitted in excluded):
                windows.append(current)
                current = []
            current.append(day)
        return windows + [current]

    def sessions(self, start, end):
        result = []
        day = start
        while day <= end:
            if self.is_session(day):
                result.append(day)
            day += timedelta(days=1)
        return result

    def cutoff(self, day, *, ready=False):
        data = self.year(day.year)
        field = "ready_at" if ready else "close"
        special = data["special_sessions"].get(str(day), {})
        value = special.get(field, data["ready_at"] if ready else data["normal_close"])
        return datetime.combine(day, time.fromisoformat(value), tzinfo=IST)

    def latest(self, now=None, *, ready=False):
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None or now.utcoffset() is None:
            raise CalendarUnavailable("Session selection requires a timezone-aware clock")
        local = now.astimezone(IST)
        day = local.date()
        while not self.is_session(day) or local < self.cutoff(day, ready=ready):
            day -= timedelta(days=1)
        return day
