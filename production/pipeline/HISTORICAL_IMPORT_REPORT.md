# Phase 2 historical import completion report

Completed on 16 November 2026.

**PASS.** The historical import committed **569,517 observations for exactly 302 instruments**. The ingestion run is `SUCCESS`. The importer's fresh-source reconciliation and a separate read-only binary-digest audit both passed. The final database/source discrepancy list is empty.

## Verified production target

| Field | Verified value |
| --- | --- |
| Project | liquidity-risk-production |
| Project ID | `super-flower-30495255` |
| Branch | production |
| Branch ID | `br-winter-thunder-azx88rt3` |
| Endpoint | `ep-misty-shadow-azik30ky` |
| Database | `neondb` |
| Schema | `liquidity` |
| User | `neondb_owner` |
| PostgreSQL | 18.7 |
| Post-import verification session | READ ONLY |

The existing connection module loaded only `LIQUIDITY_NEON_DATABASE_URL` from the project-root `.env.local`. Its hostname and server identity gate ran before exposing each transaction cursor.

## Sources and imported observations

| Source Parquet | Imported rows | First observation | Last observation |
| --- | ---: | --- | --- |
| `data/interim/stocks/ohlcv_clean.parquet` | 565,190 | 2018-01-01 | 2026-10-23 |
| `data/interim/market/nifty50_clean.parquet` | 2,171 | 2018-01-02 | 2026-10-23 |
| `data/interim/market/india_vix_clean.parquet` | 2,156 | 2018-01-01 | 2026-10-23 |
| **Total** | **569,517** | **2018-01-01** | **2026-10-23** |

The stock dimension contains exactly 100 large-cap, 150 mid-cap and 50 small-cap equities. The two market instruments are `^NSEI` and `^INDIAVIX`. All canonical company names, ISINs, segments, sectors, universe version and normalized universe checksums match the approved configuration.

| Source Parquet | Original and final SHA-256 |
| --- | --- |
| `data/interim/stocks/ohlcv_clean.parquet` | `feaefaac2ded41478f067ece25ad3860affc3aa683026b8a925ba431a1b93421` |
| `data/interim/market/nifty50_clean.parquet` | `0f9faf4c03f23f5a87e6d30451eeaf8ce243494d93b2e5fe7dc5fa41af1a66c5` |
| `data/interim/market/india_vix_clean.parquet` | `3b71021c2f4ff69f3ad58ba00b2e1f3873bcb12680650933519d2dfa181307d4` |

The canonical universe file SHA-256 is `c9f52526718685736459b5238f0b012d09dd715ba4b883697f814438db2d9054`.

## Field mapping and preservation

Ticker, date, open, high, low, close and volume map directly to `liquidity.market_eod`. Float32 source prices promote losslessly to double precision. Trading dates retain the source calendar day. The source datasets have no turnover column, and all 569,517 database turnover values are SQL NULL.

The auxiliary return, benchmark, warmup and cleaning-flag columns retain their original values in the protected Parquet files. The approved market table stores the OHLCV projection. All source bytes match their original SHA-256 checksums.

| Validation item | Result |
| --- | ---: |
| Imported observations | 569,517 |
| Distinct instruments | 302 |
| Stock dimension rows | 300 |
| Instrument dimension rows | 302 |
| Required OHLCV nulls | 0 |
| Nullable turnover values | 569,517 |
| Duplicate ticker/date keys | 0 |
| Retained zero-volume observations | 2,934 |
| Equity zero-volume observations | 751 |
| NIFTY zero-volume observations | 27 |
| INDIA VIX zero-volume observations | 2,156 |
| Source/database discrepancies | 0 |

## Ingestion, provenance and recovery

| Item | Result |
| --- | --- |
| Run ID | `22404942-d6b6-42ce-9d3d-b25a98bde777` |
| Request key | `historical-parquet-v1:807f410baa503a82bd9d677d07082595ab854363465d83c71cfffdf1f4c9570b` |
| Run mode | `bootstrap` |
| Run status | `SUCCESS` |
| Started at | 2026-11-16 03:40:48.766968+00:00 |
| Finished at | 2026-11-16 04:20:34.773720+00:00 |
| Maximum rows per batch | 4,096 |
| Committed batches | 302 |
| Replayed production batches | 0 |
| Provenance records | 302 |
| Instrument checkpoints | 302 |

All writes used the existing `Repository.register_universe`, `start_run`, `persist_batch` and `complete_run` APIs. Registration gained an optional market-instrument selection argument. Its default behavior still registers all configured market instruments, and this import selected the two requested indices.

Each batch committed its observations, provenance, batch receipt and checkpoint atomically. The deterministic request binds the source hashes, universe hash, adjustment convention and batch size. Batch keys and payload checksums support exact replay. An interrupted run remains `RUNNING` so the same command can resume committed receipts and retry the rolled-back batch.

Provenance records use `cleaned_parquet` and `sha256.<source-file-hash>`. The adjustment convention follows the existing `auto_adjust=true` project configuration. The stable `retrieved_at` value is the snapshot-import run timestamp. Original provider retrieval timestamps are unavailable in these cleaned snapshots.

The independent audit verified every source-observed session plan, batch boundary, row count, canonical batch checksum, provenance hash and final checkpoint. Each checkpoint references its instrument's final receipt and source maximum date with revision 1.

## Independent database/source verification

The importer first reloaded the source Parquet files and performed a full key/value comparison after committing the run. That audit passed.

The separate final audit independently computed ordered per-instrument SHA-256 digests from decoded Parquet values and from PostgreSQL's stored values. The binary representation includes the date ordinal, all four float8 prices, int8 volume, and a distinct nullable-turnover marker plus its optional float8 value. PostgreSQL's core `int8send` and `float8send` representations match Python's network-order `struct` encoding. Every database observation contributes to its instrument digest.

All **302** database digests, counts, date bounds and zero-volume counts match the source profiles. Comparison uses exact stored values. The audit also verified duplicates, null counts, dimension membership, session plans, source provenance, receipt checksums and checkpoints. Full per-instrument evidence is recorded in the independent validation JSON.

Additional read-only checks confirmed all 37 frozen feature columns match the HGBT/Isolation Forest feature union. There are zero unvalidated constraints, disabled triggers or invalid indexes. Row counts in `daily_features`, `daily_predictions` and `pipeline_runs` are 0. The observed database size after import was 105,873,408 bytes.

## Source coverage conditions

Observed source dates define the stored history. The import retained instrument-specific listing ranges and gaps.

The equity history ranges from 91 to 2,176 rows per instrument. Five equities have 46 missing observations relative to NIFTY source sessions within their individual observed date bounds. INDIA VIX has 17 such missing benchmark sessions and two source sessions outside the NIFTY snapshot.

| Instrument | Benchmark sessions without source rows | Source sessions outside benchmark |
| --- | ---: | ---: |
| `AEGISLOG.NS` | 1 | 4 |
| `EMBASSY.NS` | 1 | 3 |
| `FORCEMOT.NS` | 41 | 3 |
| `ITCHOTELS.NS` | 1 | 1 |
| `RRKABEL.NS` | 2 | 1 |
| `^INDIAVIX` | 17 | 2 |

The equity snapshot also contains 1,219 observations on dates absent from the NIFTY snapshot. Their original dates and OHLCV values are retained.

| Short source history | Rows | First observation | Last observation |
| --- | ---: | --- | --- |
| `VAML.NS` | 91 | 2026-06-15 | 2026-10-23 |
| `VISL.NS` | 91 | 2026-06-15 | 2026-10-23 |
| `VOGL.NS` | 91 | 2026-06-15 | 2026-10-23 |

These conditions are present in the source data and are reproduced by the database. Every source observation is represented once.

## Failures and corrections

All 302 production ingestion batches committed successfully. Import recovery was exercised through isolated regression tests.

A later independent full-table read was observed in an aborted read-only transaction and was stopped. Its paginated replacement was also stopped after prolonged transfer. The exact original database error code and root cause were unavailable. The validator was changed to return compact per-instrument profiles using exact server-side binary SHA-256, and the separate final audit passed. Production ingestion stayed `SUCCESS` throughout this read-only recovery.

## Tests and file integrity

The complete production pipeline suite passed **62 tests in 25.856 seconds** using the existing isolated PGlite PostgreSQL test backend and Python 3.13.5. New regression coverage includes invalid/duplicate source rows, exact float32 preservation, zero volumes, source-session gaps, nullable-turnover encoding, exact 302-instrument registration, partial-import recovery, repeated replay, and detection of a 0.000000001 price discrepancy through PostgreSQL/source digest comparison.

Protected research file hashes match the pre-import baseline. This includes datasets, notebooks, research source and configurations, models, artifacts, results, existing reports, and the three migration SQL files. Git HEAD and Git status outside `production/pipeline/` match the baseline.

Updated production files are `repository.py` and `README.md`. New implementation files are `historical_import.py` and `tests/test_historical_import.py`. The source plan, execution evidence, independent validation and this completion report are the new audit deliverables.

## Audit artifacts and commands

- Source plan and file hashes are in `import_results/historical_import_plan.json`.
- Original execution and first reconciliation are in `import_results/historical_import_execution.json`.
- Independent final database/source evidence is in `import_results/historical_import_validation.json`.
- This report is `HISTORICAL_IMPORT_REPORT.md`.

A read-only verification can be repeated from the repository root with the production environment.

```powershell
python -B -m production.pipeline.historical_import --validate-only --report production/pipeline/import_results/historical_import_validation.json
```

The delivered scope is the approved historical snapshot import and its validation.

