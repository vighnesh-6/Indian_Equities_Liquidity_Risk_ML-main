"""Production adapter for the approved Phase 7 mathematical contract."""

import hashlib
import json

import numpy as np
import pandas as pd

from production.pipeline.config import ROOT
from src.labels.lrs import FROZEN_LRS_THRESHOLDS, compute_lrs
from src.labels.targets import compute_forward_max_lrs, construct_hre_target


def contract_hash():
    paths = ['src/labels/lrs.py', 'src/labels/targets.py', 'production/phase1/contract.json']
    hashes = {path: hashlib.sha256((ROOT / path).read_bytes().replace(b'\r\n', b'\n')).hexdigest() for path in paths}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def calculate_labels(features):
    """Rank only post-warmup peers and expose outcomes after five future rows."""
    frame = features.sort_values(['ticker', 'date']).reset_index(drop=True).copy()
    if frame.duplicated(['ticker', 'date']).any():
        raise ValueError('Duplicate instrument dates')
    frame['date'] = pd.to_datetime(frame.date)
    if frame.date.eq(pd.Timestamp('2026-11-08')).any():
        raise ValueError('Excluded session')
    eligible = ~frame.is_warmup
    frame['lrs'] = np.nan
    frame.loc[eligible, 'lrs'] = compute_lrs(frame.loc[eligible])
    frame['forward_lrs'] = np.nan
    frame.loc[eligible, 'forward_lrs'] = compute_forward_max_lrs(frame.loc[eligible])
    frame['hre_label'] = np.nan
    frame.loc[eligible, 'hre_label'] = construct_hre_target(
        frame.loc[eligible], thresholds=FROZEN_LRS_THRESHOLDS, fwd_max_col='forward_lrs')
    frame['horizon_end'] = frame.groupby('ticker').date.shift(-5)
    frame.loc[~eligible, 'horizon_end'] = pd.NaT
    return frame
