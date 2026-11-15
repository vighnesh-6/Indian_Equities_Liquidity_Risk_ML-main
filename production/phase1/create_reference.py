"""Generate fixed parity fixtures once using the existing Windows environment."""

import json
import platform
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from validate_runtime import (
    HERE, ROOT, MODEL_HASHES, THRESHOLD, environment, load_models, predict, sha256,
)


def write_new(name, value):
    with (HERE / name).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    if platform.system() != "Windows" or platform.python_version() != "3.13.5":
        raise RuntimeError("Generate references with the verified Windows Python 3.13.5")
    for name in ["inputs.json", "windows_reference.json", "contract.json"]:
        if (HERE / name).exists():
            raise FileExistsError(f"Reference already exists: {name}")
    hgbt, detector = load_models()
    features = {
        "hist_gradient_boosting": list(hgbt.feature_names_in_),
        "isolation_forest": list(detector.feature_subset),
    }
    columns = list(dict.fromkeys(features["hist_gradient_boosting"] + features["isolation_forest"]))
    source = ROOT / "data" / "processed" / "features.parquet"
    parquet = pq.ParquetFile(source)
    samples = []
    for batch in parquet.iter_batches(batch_size=16384, columns=columns + ["date", "ticker", "cap_segment"]):
        chunk = batch.to_pandas()
        valid = np.isfinite(chunk[columns].to_numpy()).all(axis=1)
        for _, group in chunk.loc[valid].groupby("cap_segment", sort=True):
            positions = np.unique(np.linspace(0, len(group) - 1, min(16, len(group)), dtype=int))
            samples.append(group.iloc[positions])
    historical = pd.concat(samples, ignore_index=True)
    if historical.empty:
        raise AssertionError("No eligible historical inputs")
    candidate = historical[columns].astype("float64")
    candidate_predictions = predict(candidate, hgbt, detector, features)
    probabilities = np.array(candidate_predictions["hgbt_probability"])
    decisions = np.array(candidate_predictions["isolation_decision"])
    selected = np.unique(np.linspace(0, len(candidate) - 1, min(384, len(candidate)), dtype=int)).tolist()
    selected.extend(np.argsort(np.abs(probabilities - THRESHOLD))[:8].tolist())
    selected.extend(np.argsort(np.abs(decisions))[:8].tolist())
    selected.extend([int(np.argmin(probabilities)), int(np.argmax(probabilities)),
                     int(np.argmin(decisions)), int(np.argmax(decisions))])
    selected = list(dict.fromkeys(selected))
    vectors = [candidate.iloc[i].to_numpy().copy() for i in selected]
    cases = [f"historical:{historical.iloc[i]['date']}:{historical.iloc[i]['ticker']}" for i in selected]
    base = candidate.median().to_numpy()

    def add(label, vector):
        cases.append(label)
        vectors.append(np.array(vector, dtype="float64"))

    for i, name in enumerate(columns):
        vector = base.copy()
        vector[i] = np.nan
        add(f"missing:{name}", vector)
    add("missing:all", np.full(len(columns), np.nan))
    add("synthetic:zero", np.zeros(len(columns)))
    add("synthetic:historical_minima", candidate.min().to_numpy())
    add("synthetic:historical_maxima", candidate.max().to_numpy())
    for feature_index, cuts in enumerate(hgbt.named_steps["classifier"]._bin_mapper.bin_thresholds_):
        if not len(cuts):
            continue
        boundary = float(cuts[len(cuts) // 2])
        column_index = columns.index(features["hist_gradient_boosting"][feature_index])
        for direction, value in [("below", np.nextafter(boundary, -np.inf)),
                                 ("equal", boundary), ("above", np.nextafter(boundary, np.inf))]:
            vector = base.copy()
            vector[column_index] = value
            add(f"hgbt_bin:{columns[column_index]}:{direction}", vector)
    for name, values, threshold in [("hgbt", probabilities, THRESHOLD), ("if", decisions, 0.0)]:
        lower = np.flatnonzero(values < threshold)
        upper = np.flatnonzero(values >= threshold)
        if not len(lower) or not len(upper):
            raise AssertionError(f"Historical sample does not span the {name} decision boundary")
        lo = candidate.iloc[lower[np.argmax(values[lower])]].to_numpy().copy()
        hi = candidate.iloc[upper[np.argmin(values[upper])]].to_numpy().copy()
        for _ in range(40):
            mid = (lo + hi) / 2
            frame = pd.DataFrame([mid], columns=columns)
            if name == "hgbt":
                score = hgbt.predict_proba(frame[features["hist_gradient_boosting"]])[0, 1]
            else:
                score = detector.pipeline.decision_function(frame[features["isolation_forest"]])[0]
            if score < threshold:
                lo = mid
            else:
                hi = mid
        add(f"{name}_decision:below", lo)
        add(f"{name}_decision:above", hi)
    frame = pd.DataFrame(vectors, columns=columns)
    values = [[None if np.isnan(value) else float(value) for value in row] for row in frame.to_numpy()]
    write_new("inputs.json", {"columns": columns, "cases": cases, "values": values})
    write_new("windows_reference.json", {
        "environment": environment(),
        "predictions": predict(frame, hgbt, detector, features),
    })
    coverage = {
        "rows": len(frame), "historical_rows": len(selected),
        "sampled_cap_segments": sorted(historical["cap_segment"].unique().tolist()),
        "sampled_start": str(historical["date"].min()),
        "sampled_end": str(historical["date"].max()),
        "single_feature_missing_cases": len(columns), "all_missing_case": True,
        "bin_boundary_cases": sum(label.startswith("hgbt_bin:") for label in cases),
        "decision_boundary_cases": 4,
        "note": "Synthetic bin and decision probes test runtime behavior and are not feature-engineering outputs.",
    }
    write_new("contract.json", {
        "model_hashes": MODEL_HASHES, "features": features,
        "if_score_bounds": [detector.score_min_, detector.score_max_],
        "if_offset": float(detector.model.offset_), "coverage": coverage,
        "source": {"path": "data/processed/features.parquet", "sha256": sha256(source)},
        "files": {name: sha256(HERE / name) for name in ["inputs.json", "windows_reference.json"]},
    })
    print(json.dumps(coverage, indent=2))


if __name__ == "__main__":
    main()
