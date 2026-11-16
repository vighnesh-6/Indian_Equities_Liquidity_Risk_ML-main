"""Strict adapter for the approved yfinance daily OHLCV provider.

The research DataFetcher fills null volumes and drops null closes. Production
validates the raw daily response before applying its approved float32 convention.
"""

import logging
import math
import tempfile
import time
import warnings
from datetime import datetime, timedelta, timezone
from numbers import Real

import numpy as np
import pandas as pd

from .nse_calendar import IST, NSECalendar
from .repository import Bar, Provenance, prepare_batch


class ProviderIncomplete(ValueError):
    def __init__(self, ticker, code, *, missing=(), detail=None):
        self.ticker = ticker
        self.code = code
        self.missing = sorted(missing)
        self.detail = detail
        super().__init__(f"{ticker}: {code}" + (f"; missing {','.join(map(str, self.missing))}" if missing else ""))

    def report(self):
        return {"ticker": self.ticker, "code": self.code, "missing_sessions": list(map(str, self.missing)), "detail": self.detail}


def normalize_history(ticker, frame, expected):
    if frame is None or frame.empty:
        raise ProviderIncomplete(ticker, "EMPTY_RESPONSE", missing=expected)
    frame = frame.copy()
    frame.columns = [str(name).strip().lower() for name in frame.columns]
    required = ["open", "high", "low", "close", "volume"]
    if not set(required).issubset(frame.columns) or frame.columns.duplicated().any():
        raise ProviderIncomplete(ticker, "INVALID_COLUMNS")
    records, raw_records = {}, {}
    for timestamp, *values in frame.loc[:, required].itertuples(index=True, name=None):
        stamp = pd.Timestamp(timestamp)
        if pd.isna(stamp):
            raise ProviderIncomplete(ticker, "INVALID_DATE")
        if stamp.tzinfo is not None:
            stamp = stamp.tz_convert(IST)
        if stamp != stamp.normalize():
            raise ProviderIncomplete(ticker, "INTRADAY_TIMESTAMP")
        day = stamp.date()
        if day not in expected:
            raise ProviderIncomplete(ticker, "UNEXPECTED_SESSION", detail=str(day))
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
               or not math.isfinite(value) for value in values):
            raise ProviderIncomplete(ticker, "MISSING_OR_NONFINITE_OHLCV", detail=str(day))
        volume = values[4]
        integer_volume = int(volume)
        if not 0 <= integer_volume <= 2**63 - 1 or integer_volume != volume:
            raise ProviderIncomplete(ticker, "INVALID_VOLUME", detail=str(day))
        raw = (*values[:4], integer_volume)
        if day in raw_records and raw_records[day] != raw:
            raise ProviderIncomplete(ticker, "CONFLICTING_DUPLICATE", detail=str(day))
        raw_records[day] = raw
        opening, high, low, close = values[:4]
        if low <= 0 or not low <= min(opening, close) <= max(opening, close) <= high:
            raise ProviderIncomplete(ticker, "INVALID_OHLCV_BOUNDS", detail=str(day))
        bar = Bar(day, *[float(np.float32(value)) for value in values[:4]], integer_volume)
        if day in records and records[day] != bar:
            raise ProviderIncomplete(ticker, "CONFLICTING_DUPLICATE", detail=str(day))
        records[day] = bar
    missing = set(expected) - set(records)
    if missing:
        raise ProviderIncomplete(ticker, "MISSING_SESSIONS", missing=missing, detail={"observed_rows": len(records)})
    provenance = Provenance("yfinance", "validation", "adjusted", datetime.now(timezone.utc))
    bars = [records[day] for day in sorted(records)]
    try:
        prepare_batch(bars, sorted(records), None, provenance)
    except (TypeError, ValueError):
        raise ProviderIncomplete(ticker, "INVALID_OHLCV_BOUNDS") from None
    return records


class YahooProvider:
    def __init__(self, *, attempts=3, delay=1.0, backoff=2.0, timeout=20, sleeper=time.sleep):
        if attempts < 1 or delay < 0 or backoff < 1 or timeout <= 0:
            raise ValueError("Provider retry and timeout settings are invalid")
        import yfinance as yf
        self.yf = yf
        self.version = yf.__version__
        self.attempts, self.delay, self.backoff, self.timeout = attempts, delay, backoff, timeout
        self.sleep = sleeper
        self.calls = 0
        self.retry_events = []
        self.yf.set_tz_cache_location(tempfile.mkdtemp(prefix="liquidity-yahoo-"))
        logging.getLogger("yfinance").setLevel(logging.CRITICAL)

    def fetch(self, ticker, expected):
        expected = sorted(expected)
        if not expected:
            raise ValueError("Fetching requires an explicit nonempty session plan")
        windows = NSECalendar().request_windows(expected)
        for attempt in range(self.attempts):
            try:
                result = {}
                for window in windows:
                    if self.calls:
                        self.sleep(self.delay)
                    self.calls += 1
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", DeprecationWarning)
                        frame = self.yf.Ticker(ticker).history(
                            start=str(window[0]), end=str(window[-1] + timedelta(days=1)),
                            interval="1d", auto_adjust=True, actions=False, repair=False,
                            timeout=self.timeout, raise_errors=True,
                        )
                    result.update(normalize_history(ticker, frame, window))
                return result, Provenance("yfinance", self.version, "adjusted", datetime.now(timezone.utc))
            except ProviderIncomplete as error:
                failure = error
            except Exception as error:
                failure = ProviderIncomplete(ticker, "PROVIDER_REQUEST_FAILED", detail=type(error).__name__)
            self.retry_events.append({"attempt": attempt + 1, **failure.report()})
            if attempt + 1 < self.attempts:
                self.sleep(min(self.backoff ** (attempt + 1), 30))
        raise failure
