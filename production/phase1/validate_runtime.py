"""Read frozen inputs and compare inference with the Windows reference."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import warnings
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

MODEL_HASHES = {
    "hist_gradient_boosting": "7848d911b5ca8afb04114675d5507ac91f104efaca57a47a44706e8fd37921b9",
    "isolation_forest": "9d5a05088dfb2083da2fce9bd430d8ca6f1e752f7bd8f8a7016be14a02de7377",
}
RTOL = 1e-7
ATOL = 1e-9
THRESHOLD = 0.534


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot():
    """Include tracked and untracked files, with outputs kept outside this tree."""
    return {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in ROOT.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(ROOT).parts
    }


def environment():
    pins = {}
    for line in (HERE / "requirements.txt").read_text().splitlines():
        if "==" in line and not line.startswith("#"):
            name, version = line.strip().split("==")
            pins[name] = {"expected": version, "actual": importlib.metadata.version(name)}
    return {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "packages": pins,
    }


def load_models():
    import joblib
    from sklearn.exceptions import InconsistentVersionWarning
    from src.modeling.anomaly import AnomalyDetector

    loaded = {}
    for name, expected in MODEL_HASHES.items():
        path = ROOT / "models" / "phase8_batch5" / name / "model.joblib"
        if sha256(path) != expected:
            raise AssertionError(f"Frozen artifact hash mismatch for {name}")
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            loaded[name] = (
                AnomalyDetector.load_artifact(path)
                if name == "isolation_forest"
                else joblib.load(path)
            )
    return loaded["hist_gradient_boosting"], loaded["isolation_forest"]


def predict(frame, hgbt, detector, features):
    import numpy as np

    probability = hgbt.predict_proba(frame[features["hist_gradient_boosting"]])[
        :, list(hgbt.classes_).index(1.0)
    ]
    anomaly = detector.score(frame)
    return {
        "hgbt_probability": probability.tolist(),
        "hgbt_flag": (probability >= THRESHOLD).astype(int).tolist(),
        "hgbt_native_class": hgbt.predict(
            frame[features["hist_gradient_boosting"]]
        ).astype(int).tolist(),
        "isolation_score": anomaly["isolation_score"].tolist(),
        "isolation_flag": anomaly["isolation_flag"].tolist(),
        "anomaly_score_scaled": anomaly["anomaly_score_scaled"].tolist(),
        "isolation_decision": detector.pipeline.decision_function(
            frame[features["isolation_forest"]]
        ).tolist(),
    }


def validate(report, require_linux):
    import numpy as np
    import pandas as pd

    checks = report["checks"]

    def check(name, condition):
        checks[name] = "PASS" if condition else "FAIL"
        if not condition:
            raise AssertionError(name)

    info = environment()
    report["environment"] = info
    check("python_version", info["python"] == "3.13.5")
    check("x64_architecture", info["machine"].lower() in {"amd64", "x86_64"})
    check("isolated_virtual_environment", sys.prefix != sys.base_prefix)
    if require_linux:
        check("linux_runner", info["system"] == "Linux")
    check("pinned_versions", all(v["actual"] == v["expected"] for v in info["packages"].values()))
    result = subprocess.run(
        [sys.executable, "-B", "-m", "pip", "check"], capture_output=True, text=True
    )
    report["pip_check"] = result.stdout + result.stderr
    check("dependency_consistency", result.returncode == 0)

    contract = json.loads((HERE / "contract.json").read_text())
    for name, expected_hash in contract["files"].items():
        check(f"input_hash:{name}", sha256(HERE / name) == expected_hash)
    check("fixed_model_hashes", contract["model_hashes"] == MODEL_HASHES)
    fixtures = json.loads((HERE / "inputs.json").read_text())
    reference = json.loads((HERE / "windows_reference.json").read_text())
    features = contract["features"]
    frame = pd.DataFrame(fixtures["values"], columns=fixtures["columns"], dtype="float64")
    check("fixture_shape", len(frame) == len(fixtures["cases"]) and len(frame) > 0)
    check("finite_or_missing_inputs", not np.isinf(frame.to_numpy()).any())
    check("unique_columns", frame.columns.is_unique)
    hgbt, detector = load_models()
    check("artifact_hashes_and_loading", True)
    for name, model in [("hist_gradient_boosting", hgbt), ("isolation_forest", detector.pipeline)]:
        path = ROOT / "models" / "phase8_batch5" / name / "feature_names.json"
        metadata = ROOT / "reports" / "phase8_batch5" / "training_metadata" / f"{name}_training_metadata.json"
        check(f"feature_order:{name}", list(model.feature_names_in_) == features[name])
        check(f"feature_file:{name}", json.loads(path.read_text())["features"] == features[name])
        check(f"training_features:{name}", json.loads(metadata.read_text())["feature_list"] == features[name])
        check(f"feature_count:{name}", model.n_features_in_ == len(features[name]))
    check("if_custom_feature_order", detector.feature_subset == features["isolation_forest"])
    check("if_preprocessing", list(detector.pipeline.named_steps) == ["imputer", "model"])
    check("if_fitted_medians", detector.pipeline.named_steps["imputer"].strategy == "median"
          and len(detector.pipeline.named_steps["imputer"].statistics_) == 12)
    check("if_score_bounds", [detector.score_min_, detector.score_max_] == contract["if_score_bounds"])
    check("if_decision_offset", float(detector.model.offset_) == contract["if_offset"])
    actual = predict(frame, hgbt, detector, features)
    report["rows"] = len(frame)
    report["coverage"] = contract["coverage"]
    report["tolerances"] = {"rtol": RTOL, "atol": ATOL}
    report["comparisons"] = {}
    for name, values in actual.items():
        observed = np.asarray(values)
        expected = np.asarray(reference["predictions"][name])
        check(f"finite_output:{name}", np.isfinite(observed).all())
        check(f"output_shape:{name}", observed.shape == expected.shape)
        binary = name in {"hgbt_flag", "hgbt_native_class", "isolation_flag"}
        equal = observed == expected if binary else np.isclose(observed, expected, rtol=RTOL, atol=ATOL)
        mismatches = np.flatnonzero(~equal)
        report["comparisons"][name] = {
            "max_absolute_difference": float(np.max(np.abs(observed - expected))),
            "mismatch_cases": [fixtures["cases"][i] for i in mismatches],
        }
        check(f"prediction_parity:{name}", not len(mismatches))
    check("hgbt_probability_range", ((np.array(actual["hgbt_probability"]) >= 0)
          & (np.array(actual["hgbt_probability"]) <= 1)).all())
    check("if_scaled_range", ((np.array(actual["anomaly_score_scaled"]) >= 0)
          & (np.array(actual["anomaly_score_scaled"]) <= 100)).all())
    reordered = predict(frame[frame.columns[::-1]], hgbt, detector, features)
    check("adapter_column_order", reordered == actual)
    for name, function in [
        ("hgbt", lambda: hgbt.predict_proba(frame[features["hist_gradient_boosting"]].iloc[:, ::-1])),
        ("if", lambda: detector.score(frame.drop(columns=[features["isolation_forest"][0]]))),
    ]:
        try:
            function()
        except ValueError:
            checks[f"invalid_schema_rejected:{name}"] = "PASS"
        else:
            check(f"invalid_schema_rejected:{name}", False)
    check("hgbt_threshold_boundary", (np.array([np.nextafter(THRESHOLD, -np.inf), THRESHOLD,
          np.nextafter(THRESHOLD, np.inf)]) >= THRESHOLD).astype(int).tolist() == [0, 1, 1])
    report["predictions"] = actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-linux", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(ROOT):
        parser.error("Validation outputs must use a temporary directory outside the repository")
    before = snapshot()
    report = {"status": "FAIL", "checks": {}}
    try:
        validate(report, args.require_linux)
        report["status"] = "PASS"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        after = snapshot()
        changes = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
        report["changed_repository_files"] = changes
        report["checks"]["repository_integrity"] = "FAIL" if changes else "PASS"
        if changes:
            report["status"] = "FAIL"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "predictions"}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
