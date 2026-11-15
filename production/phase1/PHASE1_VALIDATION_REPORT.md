# Phase 1 runtime and frozen model compatibility report

Audit date 16 November 2026.

**Overall status — BLOCKED pending the actual GitHub Actions Ubuntu run.**

The production dependency specification, fixed Windows reference inputs and predictions, inference validator and manual Ubuntu workflow are prepared. A clean Windows environment installed successfully and passed 53 validation checks across 551 cases. Linux wheel resolution passed. Ubuntu installation and Windows–Linux prediction parity await workflow execution, so the production runtime is awaiting Phase 2 acceptance.

## Runtime specification

The target is standard CPython 3.13.5 on Ubuntu 24.04 x64. The baseline is the existing Anaconda Python 3.13.5 Windows AMD64 environment. The independently created temporary Windows virtual environment uses official binary wheels and disables system site packages.

| Package | Pinned version |
| --- | --- |
| Python | 3.13.5 |
| pip bootstrap | 25.1.1 |
| NumPy | 2.1.3 |
| pandas | 2.2.3 |
| scikit-learn | 1.8.0 |
| SciPy | 1.15.3 |
| joblib | 1.4.2 |
| threadpoolctl | 3.5.0 |
| PyArrow | 19.0.0 |
| PyYAML | 6.0.3 |
| python-dateutil | 2.9.0.post0 |
| pytz | 2026.3.post1 |
| tzdata | 2026.3 |
| six | 1.17.0 |

`requirements.txt` lists the complete 12-package inference/configuration/Parquet runtime closure. `requirements.lock` pins those versions and SHA-256 hashes for the downloaded Windows and Linux distributions. pip is the separately pinned environment bootstrap tool.

Both research dependency files declare NumPy below 2, scikit-learn below 1.6 and PyArrow below 18. These conflict with the verified environment. Their yfinance declarations also disagree. The separate production lock resolves the model-runtime conflicts by matching the verified numerical stack. The workflow installs this lock and imports the existing research modules directly from the checkout.

Training metadata explicitly records NumPy 2.1.3, pandas 2.2.3 and scikit-learn 1.8.0. Other pins reproduce the currently working environment. Their original training-time versions are unrecorded in those model metadata files. Scikit-learn recommends preserving dependency versions for persisted models. [Scikit-learn model persistence](https://scikit-learn.org/stable/model_persistence.html)

## Completed installation evidence

1. Created a fresh Python 3.13.5 virtual environment in the system temporary directory.
2. Installed all 12 pinned packages from scratch with binary wheels. Installation exited successfully.
3. Reinstalled using the final hash lock with `--require-hashes` and `--only-binary` to verify distribution hashes. Installation exited successfully.
4. Ran `pip check` successfully with no broken requirements.
5. Resolved and downloaded all 12 CPython 3.13-compatible Linux x64 wheels. Inspected their runtime requirements using Linux environment markers. All requirements were satisfied by the pinned set.

Linux wheel resolution evidence is recorded in `linux_wheel_resolution.json`. Actual Linux installation and inference remain pending.

Temporary installation evidence is under `C:\Users\vighn\AppData\Local\Temp\liquidity-phase1-49t5ctr8`, including `install-windows.json`, `install-windows-hashed.json` and `validation-windows-final.json`. Temporary storage may be cleaned by the operating system. Persistent local validation evidence is in `windows_validation.json`.

## Artifact and interface verification

| Artifact | SHA-256 |
| --- | --- |
| `models/phase8_batch5/hist_gradient_boosting/model.joblib` | `7848d911b5ca8afb04114675d5507ac91f104efaca57a47a44706e8fd37921b9` |
| `models/phase8_batch5/isolation_forest/model.joblib` | `9d5a05088dfb2083da2fce9bd430d8ca6f1e752f7bd8f8a7016be14a02de7377` |

HGBT loads through `joblib.load` and its saved pipeline exposes `predict_proba` and `predict`. Production flags use the approved probability threshold 0.534. The existing `ModelScorer` is a scaffold, so this validation exercises the saved estimator interface.

Isolation Forest loads through the existing `AnomalyDetector.load_artifact` interface. Its `src.modeling.anomaly.AnomalyDetector` class resolves from the repository root. Inference uses its existing `score` method. Its saved pipeline contains the fitted 12-feature median imputer and Isolation Forest, with no external preprocessing file required. The manifest reference to a separate `preprocessor.joblib` remains a documentation inconsistency.

The saved IF decision offset is -0.5366245497779618. Its saved score bounds are -0.7758002614394625 and -0.34533976879598105. Validation checks these values, embedded imputation statistics and model-specific ordered feature lists against the fitted objects, feature JSON and training metadata.

HGBT has 36 ordered features and IF has 12, with 37 distinct inputs in their union. The exact lists are recorded in `contract.json`. Model loading promotes scikit-learn serialization-version warnings to failures.

## Fixed inputs and Windows comparison

`inputs.json` stores identical float64 feature vectors for every environment. Missing values are represented as JSON null and restored to NaN. `windows_reference.json` contains predictions generated using the existing Windows environment. The contract records input and reference hashes and the source Parquet hash. A scoped `.gitattributes` file preserves LF endings for the new JSON fixtures across platform checkouts.

The fixture contains 551 rows comprising 398 historical observations, 37 single-feature missing-value cases, one all-missing case, three synthetic zero/extrema cases, 108 HGBT bin-boundary probes and four decision-boundary probes. Historical sampling spans large, mid and small segments between 30 April 2018 and 23 October 2026.

Bin-boundary probes use values immediately below, at and above saved histogram cut points. HGBT and IF decision probes use feature vectors on opposite sides of their decisions. The validator also checks the HGBT comparator immediately below, at and above 0.534, rejects invalid schemas and verifies explicit model-specific column selection. Synthetic probes exercise runtime behavior.

| Output | Maximum absolute difference, existing Windows vs clean Windows | Result |
| --- | --- | --- |
| HGBT positive probability | 2.220446049250313e-16 | PASS |
| HGBT flag at 0.534 | 0 | PASS |
| HGBT native predicted class | 0 | PASS |
| IF raw score | 0 | PASS |
| IF anomaly flag | 0 | PASS |
| IF scaled score | 0 | PASS |
| IF decision function | 0 | PASS |

All numerical comparisons use rtol=1e-7 and atol=1e-9. Binary outputs require exact equality. Every comparison passed. No tolerance adjustment occurred.

Negative controls confirmed that the Linux-required check rejects a Windows process and that an injected 1e-4 probability mismatch fails validation. The injection existed only in process memory during validator testing. Results are recorded in `validation_controls.json`.

A Windows joblib warning reported unavailable WMIC physical-core detection and used logical-core fallback. Inference completed successfully.

## Validation decisions

| Validation | Status |
| --- | --- |
| Clean Windows Python 3.13.5 environment | PASS |
| Complete pinned dependency installation | PASS |
| Final distribution hash verification | PASS |
| Dependency consistency | PASS |
| Linux wheel availability and dependency closure | PASS |
| HGBT artifact identity and loading | PASS on Windows |
| IF artifact identity, custom import and loading | PASS on Windows |
| Embedded preprocessing and saved score constants | PASS on Windows |
| Feature names, counts and ordering | PASS on Windows |
| Historical, missing-value and boundary inference | PASS on Windows |
| Original Windows vs clean Windows parity | PASS |
| Validator failure controls | PASS |
| New workflow YAML, embedded Python and Bash syntax | PASS |
| Ubuntu clean installation | BLOCKED pending execution |
| Ubuntu model loading | BLOCKED pending execution |
| Windows–Linux numerical and binary parity | BLOCKED pending execution |
| Existing research-file preservation | See final `preservation_verification.json` |
| Phase 2 runtime readiness | HOLD |

## Prepared Ubuntu workflow and required action

The new `.github/workflows/phase1-runtime-validation.yml` runs exclusively through `workflow_dispatch` on Ubuntu 24.04 x64. It pins Python 3.13.5 and action commit revisions, uses a read-only repository token, creates a fresh virtual environment, installs the hash lock, verifies artifacts and compares all outputs with Windows references. It writes outputs to the runner temporary directory, verifies repository hashes and Git status, then uploads a seven-day evidence artifact named `phase1-runtime-evidence`.

GitHub CLI and the inspected GitHub token environment variables were unavailable. Docker was unavailable and WSL reported no installed Linux subsystem. Repository publication is reserved for the user under the standing instruction to perform commits and pushes personally.

The user must publish only `production/phase1/` and `.github/workflows/phase1-runtime-validation.yml` to the repository default branch. The two pre-existing untracked Parquet files remain outside this publication step. In GitHub, open Actions, select **Phase 1 frozen model runtime validation**, choose **Run workflow**, select the published branch and run it. This workflow requires no project secrets. GitHub requires a manually dispatched workflow to exist on the default branch. [GitHub manual workflow instructions](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow)

Review the completed run and its `phase1-runtime-evidence` artifact. Acceptance requires a successful job, a PASS in `phase1-validation-linux.json`, zero binary mismatches, every numerical comparison within the fixed tolerances and PASS in `phase1-integrity-linux.json`. A failed comparison must be investigated with its reported case identifiers and actual outputs before acceptance.

The runtime can advance to Phase 2 after the actual Ubuntu run satisfies these checks and the user approves progression. The separate Phase 0 data-permission and infrastructure questions retain their own release requirements.

## Preservation verification

The initial tracked and staged diffs were empty. The initial Git status contained only `data/processed/features.parquet` and `data/processed/features_with_targets.parquet` as untracked files. A baseline captured all 318 existing repository files outside `.git`, including existing bytecode/cache files.

All permanent additions use the two explicitly approved paths. Final hash comparisons, additions, Git HEAD and tracked/staged diff results are recorded in `preservation_verification.json`. Validation scripts disable bytecode writes and use inference-only model APIs.
