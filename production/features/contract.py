"""Immutable feature ordering and reproducible implementation identity."""

import hashlib
import json
from pathlib import Path
from uuid import UUID

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[1]
CONTRACT = json.loads((DIRECTORY / "contract.json").read_text(encoding="utf-8"))
FEATURES = tuple(CONTRACT["core_features"] + CONTRACT["supplementary_features"])
FLAGS = tuple(CONTRACT["quality_flags"])
FLOAT32_RTOL = 1e-6
FLOAT32_ATOL = 1e-8


def text_sha256(path):
    """Preserve text identities across Windows and Ubuntu checkouts."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def verify_contract():
    if len(FEATURES) != 58 or len(set(FEATURES)) != 58:
        raise ValueError("The frozen contract must contain 58 distinct features")
    for relative, expected in CONTRACT["source_hashes"].items():
        if text_sha256(ROOT / relative) != expected:
            raise ValueError("Frozen feature source differs: " + relative)


def implementation_hash():
    paths = [DIRECTORY / name for name in
             ["contract.json", "kernels.py", "inputs.py", "engine.py", "state.py"]]
    hashes = {path.name: text_sha256(path) for path in paths}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def calculation_identity(source_version):
    return hashlib.sha256((implementation_hash() + ":" + str(source_version)).encode()).hexdigest()


def calculation_version(source_version):
    # Compact relational keys retain the complete SHA-256 in feature_versions.
    return str(UUID(hex=calculation_identity(source_version)[:32]))
