"""Bounded rolling context and persistent recursive Wilder RSI state."""

from dataclasses import asdict, dataclass, field
import json

import numpy as np
import pandas as pd

from .contract import CONTRACT
from .engine import calculate_prepared, prepare_stock, validate_market_frame
from .kernels import _rsi_from_averages, compute_liquidity_features, rolling_zscore


@dataclass
class FeatureState:
    source_version: str
    ticker: str
    count: int = 0
    last_date: str | None = None
    previous_close: float | None = None
    gains: list = field(default_factory=list)
    losses: list = field(default_factory=list)
    average_gain: float | None = None
    average_loss: float | None = None
    context: list = field(default_factory=list)
    amihud_history: list = field(default_factory=list)

    def to_json(self):
        return json.dumps(asdict(self), allow_nan=False, sort_keys=True)

    @classmethod
    def from_json(cls, value):
        return cls(**json.loads(value))


def advance_rsi(state, close, masked):
    previous = state.previous_close
    state.previous_close = float(close)
    if previous is None or masked:
        return np.nan
    delta = float(close) - previous
    gain, loss = max(delta, 0.0), max(-delta, 0.0)
    if state.average_gain is None:
        state.gains.append(gain)
        state.losses.append(loss)
        if len(state.gains) < 14:
            return np.nan
        state.average_gain = sum(state.gains) / 14.0
        state.average_loss = sum(state.losses) / 14.0
        state.gains.clear()
        state.losses.clear()
    else:
        state.average_gain = (13.0 * state.average_gain + gain) / 14.0
        state.average_loss = (13.0 * state.average_loss + loss) / 14.0
    return _rsi_from_averages(state.average_gain, state.average_loss)


def initialize(stock, benchmark, vix, *, source_version):
    """Calculate a full historical block and retain its exact recursive state."""
    prepared = prepare_stock(stock, benchmark)
    state = FeatureState(source_version=source_version, ticker=str(prepared.ticker.iloc[0]))
    output = calculate_prepared(prepared, benchmark, vix)
    for row in prepared.itertuples(index=False):
        advance_rsi(state, row.close, bool(row.is_corporate_action_gap))
    state.count = len(prepared)
    amihud = compute_liquidity_features(prepared)["amihud_ratio"]
    state.amihud_history = [None if pd.isna(value) else float(value) for value in amihud]
    state.last_date = str(prepared.date.iloc[-1].date())
    tail = prepared.iloc[-CONTRACT["context_rows"]:].copy()
    tail["date"] = tail.date.dt.strftime("%Y-%m-%d")
    state.context = tail.replace({np.nan: None}).to_dict("records")
    return output, state


def update(state, observations, benchmark, vix, *, source_version):
    """Return new dates only and a replacement state, leaving caller state intact."""
    state = FeatureState.from_json(state.to_json())
    new = validate_market_frame(observations, stock=True)
    if source_version != state.source_version or new.ticker.iloc[0] != state.ticker:
        raise ValueError("Changed ticker or input vintage requires a versioned historical rebuild")
    if len(state.amihud_history) != state.count:
        raise ValueError("Missing rolling-variance history requires a versioned historical rebuild")
    if state.last_date is not None and new.date.min() <= pd.Timestamp(state.last_date):
        raise ValueError("Incremental observations must strictly follow the checkpoint")
    context = pd.DataFrame(state.context)
    if not context.empty:
        context["date"] = pd.to_datetime(context.date)
    # Re-enrich the bounded input context with the frozen rules. The cumulative
    # warm-up count and recursive RSI are restored independently below.
    raw_columns = ["date", "ticker", "open", "high", "low", "close", "volume"]
    combined = pd.concat([context.reindex(columns=raw_columns), new[raw_columns]], ignore_index=True)
    prepared = prepare_stock(combined, benchmark)
    prepared["is_warmup"] = np.arange(len(prepared)) + state.count - len(context) < 80
    prepared["is_short_history"] = state.count + len(new) < 200
    result = calculate_prepared(prepared, benchmark, vix).iloc[len(context):].copy()
    # Pandas 2.2.3 carries Welford/Kahan roundoff from earlier windows. Replay
    # this one cached float64 input stream to reproduce the frozen variance,
    # including large historical scale changes. Historical feature rows are
    # never recalculated or rewritten. JSON round-trips preserve these doubles.
    new_amihud = compute_liquidity_features(prepared)["amihud_ratio"].iloc[len(context):]
    state.amihud_history.extend(None if pd.isna(value) else float(value) for value in new_amihud)
    replayed = rolling_zscore(pd.Series(state.amihud_history, dtype="float64"), 20, min_periods=15)
    result["amihud_z20"] = replayed.iloc[-len(new):].to_numpy(dtype="float32")
    for index, row in prepared.iloc[len(context):].iterrows():
        result.loc[index, "rsi_14"] = np.float32(advance_rsi(
            state, row.close, bool(row.is_corporate_action_gap)))
    state.count += len(new)
    state.last_date = str(new.date.iloc[-1].date())
    tail = prepared.iloc[-CONTRACT["context_rows"]:].copy()
    tail["date"] = tail.date.dt.strftime("%Y-%m-%d")
    state.context = tail.replace({np.nan: None}).to_dict("records")
    return result.reset_index(drop=True), state
