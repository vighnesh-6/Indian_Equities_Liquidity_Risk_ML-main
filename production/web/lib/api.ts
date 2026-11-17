import "server-only";
import { query, queryMany, type Row } from "./database";
import { latestReadySession } from "./calendar";

export class InputError extends Error {}

export function tradingDate(value: string | null): string | null {
  if (!value) return null;
  if (
    !/^\d{4}-\d{2}-\d{2}$/.test(value) ||
    !Number.isFinite(Date.parse(value)) ||
    new Date(value).toISOString().slice(0, 10) !== value
  ) {
    throw new InputError("Use a valid YYYY-MM-DD date.");
  }
  if (value === "2026-11-08")
    throw new InputError("The 8 November 2026 Muhurat session is excluded.");
  return value;
}

function integer(
  value: string | null,
  fallback: number,
  maximum: number,
): number {
  if (value === null) return fallback;
  if (!/^\d+$/.test(value))
    throw new InputError("Pagination must contain positive integers.");
  const result = Number(value);
  if (result < 1 || result > maximum)
    throw new InputError("Pagination is outside the supported range.");
  return result;
}

const VERSION_SQL = `SELECT v.feature_version::text FROM liquidity.feature_versions v
 WHERE v.source_version='canonical-basis-'||(SELECT coalesce(max(generation_id),0)::text FROM liquidity.price_basis_generations WHERE state='ACTIVE')
 ORDER BY v.created_at DESC LIMIT 1`;

async function context(params: URLSearchParams) {
  const requestedDate = tradingDate(params.get("date"));
  const metadata = await query(`SELECT (${VERSION_SQL}) AS version,
    (SELECT max(date)::text FROM liquidity.feature_publications WHERE feature_version=(${VERSION_SQL})::uuid) AS latest_feature_date,
    (SELECT max(date)::text FROM liquidity.intelligence_publications WHERE predictions_at IS NOT NULL AND feature_version=(${VERSION_SQL})::uuid) AS latest_prediction_date,
    (SELECT min(date)::text FROM liquidity.feature_publications) AS first_date,
    (SELECT max(last_date)::text FROM liquidity.instrument_checkpoints) AS latest_market_date`);
  const meta = metadata[0];
  const selected =
    requestedDate || meta.latest_prediction_date || meta.latest_feature_date;
  if (!meta.version || !selected)
    throw new Error("ACTIVE_FEATURE_VINTAGE_PENDING");
  return {
    ...meta,
    version: meta.version,
    expected_session: latestReadySession(),
    selected_date: selected,
    stale:
      selected >= meta.latest_market_date &&
      (selected !== meta.latest_prediction_date ||
        (latestReadySession() !== null &&
          meta.latest_market_date < latestReadySession()!)),
  };
}

async function screener(params: URLSearchParams, anomalies = false) {
  const meta = await context(params);
  const page = integer(params.get("page"), 1, 10000);
  const limit = integer(params.get("limit"), 50, 300);
  const search = (params.get("search") || "").trim().slice(0, 100);
  const cap = params.get("cap") || "";
  const band = params.get("risk") || "";
  if (cap && !["large", "mid", "small"].includes(cap))
    throw new InputError("Unknown capitalisation segment.");
  if (band && !["LOW", "MODERATE", "HIGH", "VERY_HIGH"].includes(band))
    throw new InputError("Unknown risk band.");
  const sorts: Record<string, string> = {
    lrs: "r.lrs",
    probability: "r.hre_probability",
    anomaly: "r.anomaly_score_scaled",
    ticker: "s.ticker",
    name: "s.company_name",
  };
  const sort = sorts[params.get("sort") || (anomalies ? "anomaly" : "lrs")];
  if (!sort) throw new InputError("Unknown sorting field.");
  const direction = params.get("direction") === "asc" ? "ASC" : "DESC";
  const flagged = anomalies && params.get("flagged") === "1";
  const from = `FROM liquidity.stocks s LEFT JOIN liquidity.risk_intelligence r ON r.ticker=s.ticker AND r.date=$1::date AND r.feature_version=$2::uuid
    LEFT JOIN liquidity.market_eod_consistent m ON m.ticker=s.ticker AND m.date=$1::date
    WHERE s.is_active AND ($3='' OR s.ticker ILIKE '%'||$3||'%' OR s.company_name ILIKE '%'||$3||'%')
      AND ($4='' OR s.cap_segment=$4) AND ($5='' OR r.lrs_band=$5) ${flagged ? "AND r.is_anomaly IS TRUE" : ""}`;
  const values = [meta.selected_date, meta.version, search, cap, band];
  const [rows, total, distribution] = await queryMany(
    [
      {
        text: `SELECT s.ticker,s.company_name,s.cap_segment,s.sector,m.close,m.volume,r.lrs,r.lrs_band,r.hre_probability,r.hre_flag,
      r.isolation_score,r.anomaly_score_scaled,r.is_anomaly,r.hre_label,r.horizon_end,r.label_status,r.model_period,
      coalesce(r.prediction_status,CASE WHEN m.ticker IS NULL THEN 'NO_OBSERVATION' ELSE 'PENDING_FEATURES' END) AS prediction_status
      ${from} ORDER BY ${sort} ${direction} NULLS LAST,s.ticker LIMIT $6 OFFSET $7`,
        values: [...values, limit, (page - 1) * limit],
      },
      { text: `SELECT count(*)::int AS total ${from}`, values },
      {
        text: "SELECT lrs_band,count(*)::int AS count FROM liquidity.risk_intelligence WHERE feature_version=$1::uuid AND date=$2::date GROUP BY lrs_band",
        values: [meta.version, meta.selected_date],
      },
    ],
    meta.version,
  );
  const timeline = anomalies
    ? await query(
        `SELECT date,count(*) FILTER(WHERE is_anomaly)::int AS flagged FROM liquidity.risk_intelligence WHERE feature_version=$1::uuid AND date<=$2::date AND isolation_score IS NOT NULL AND predictions_at IS NOT NULL GROUP BY date ORDER BY date DESC LIMIT 90`,
        [meta.version, meta.selected_date],
      )
    : [];
  return {
    meta,
    rows,
    total: total[0].total,
    page,
    limit,
    distribution,
    timeline: timeline.reverse(),
  };
}

async function overview(params: URLSearchParams) {
  const meta = await context(params);
  const values = [meta.version, meta.selected_date];
  const [summary, benchmarks, trend, leaders, caps] = await queryMany(
    [
      {
        text: `SELECT count(*)::int AS observed,count(lrs)::int AS scored,avg(lrs) AS average_lrs,
      count(*) FILTER(WHERE lrs>=75)::int AS very_high,count(*) FILTER(WHERE hre_flag=1)::int AS hre_flags,
      count(*) FILTER(WHERE is_anomaly)::int AS anomaly_flags,avg(hre_probability) AS average_probability
      FROM liquidity.risk_intelligence WHERE feature_version=$1::uuid AND date=$2::date`,
        values,
      },
      {
        text: `SELECT m.ticker,m.date,m.close,(m.close/p.close-1)*100 AS change_pct FROM liquidity.market_eod_consistent m
      LEFT JOIN LATERAL (SELECT close FROM liquidity.market_eod_consistent WHERE ticker=m.ticker AND date<m.date ORDER BY date DESC LIMIT 1) p ON true
      WHERE m.date=$1::date AND m.ticker IN ('^NSEI','^INDIAVIX')`,
        values: [meta.selected_date],
      },
      {
        text: `SELECT date,avg(lrs) AS average_lrs,100.0*count(*) FILTER(WHERE lrs_90p_exceeded)/nullif(count(lrs),0) AS risk_breadth,count(*) FILTER(WHERE is_anomaly)::int AS anomalies FROM liquidity.risk_intelligence
      WHERE feature_version=$1::uuid AND date<=$2::date GROUP BY date ORDER BY date DESC LIMIT 90`,
        values,
      },
      {
        text: `SELECT ticker,company_name,lrs,lrs_band,hre_probability,anomaly_score_scaled,cap_segment FROM liquidity.risk_intelligence
      WHERE feature_version=$1::uuid AND date=$2::date AND lrs IS NOT NULL ORDER BY lrs DESC,ticker`,
        values,
      },
      {
        text: `SELECT cap_segment,count(*)::int AS observed,avg(lrs) AS average_lrs,avg(hre_probability) AS probability,
      count(*) FILTER(WHERE is_anomaly)::int AS anomalies FROM liquidity.risk_intelligence
      WHERE feature_version=$1::uuid AND date=$2::date GROUP BY cap_segment ORDER BY cap_segment`,
        values,
      },
    ],
    meta.version,
  );
  return {
    meta,
    summary: summary[0],
    benchmarks,
    trend: trend.reverse(),
    leaders: leaders.slice(0, 8),
    lowest_risk: leaders.slice(-5).reverse(),
    caps,
  };
}

async function stock(params: URLSearchParams) {
  const meta = await context(params);
  const ticker = (params.get("ticker") || "RELIANCE.NS").toUpperCase();
  if (!/^[A-Z0-9&.-]{1,30}\.NS$/.test(ticker))
    throw new InputError("Unknown ticker format.");
  const from = tradingDate(params.get("from")) || "2018-01-01";
  const [detail, history, snapshot] = await queryMany(
    [
      {
        text: "SELECT ticker,company_name,cap_segment,sector,isin FROM liquidity.stocks WHERE ticker=$1",
        values: [ticker],
      },
      {
        text: `SELECT m.date,m.open,m.high,m.low,m.close,m.volume,r.lrs,r.hre_probability,r.anomaly_score_scaled,r.is_anomaly,r.hre_flag,r.prediction_status
      FROM liquidity.market_eod_consistent m LEFT JOIN liquidity.risk_intelligence r
      ON r.ticker=m.ticker AND r.date=m.date AND r.feature_version=$1::uuid
      WHERE m.ticker=$2 AND m.date BETWEEN $3::date AND $4::date ORDER BY m.date LIMIT 3000`,
        values: [meta.version, ticker, from, meta.selected_date],
      },
      {
        text: `SELECT r.*,f.amihud_ma20,f.cs_spread_ma10,f.volume_z20,f.downside_vol_20d,f.rsi_14,f.vol_20d,f.volume_ratio,
      f.is_warmup,f.is_benchmark_available,f.is_valid_ohlcv FROM liquidity.risk_intelligence r JOIN liquidity.feature_values f USING(ticker,date,feature_version)
      WHERE r.ticker=$1 AND r.date=$2::date AND r.feature_version=$3::uuid`,
        values: [ticker, meta.selected_date, meta.version],
      },
    ],
    meta.version,
  );
  if (!detail.length)
    throw new InputError(
      "Ticker is outside the canonical 300-equity universe.",
    );
  return { meta, stock: detail[0], history, snapshot: snapshot[0] || null };
}

async function events(params: URLSearchParams) {
  const meta = await context(params);
  const all = params.get("all") === "1";
  const rows = await query(
    `SELECT e.*,m.close AS nifty_close FROM liquidity.market_events e
    LEFT JOIN liquidity.market_eod_consistent m ON m.date=e.event_date AND m.ticker='^NSEI'
    WHERE ($1::boolean OR e.event_date BETWEEN $2::date-90 AND $2::date+30)
    ORDER BY e.event_date DESC LIMIT 300`,
    [all, meta.selected_date],
  );
  return { meta, rows };
}

async function status() {
  const [
    coverage,
    features,
    intelligence,
    checkpoints,
    runs,
    ingestion,
    versions,
    labels,
  ] = await queryMany([
    {
      text: "SELECT count(*)::int AS instruments,min(last_date)::text AS earliest_checkpoint,max(last_date)::text AS latest_checkpoint FROM liquidity.instrument_checkpoints",
    },
    {
      text: "SELECT min(last_date)::text AS checkpoint,count(*)::int AS equities,feature_version::text FROM liquidity.feature_checkpoints GROUP BY feature_version",
    },
    {
      text: "SELECT max(date)::text AS latest_date,count(*)::int AS published_dates FROM liquidity.intelligence_publications WHERE predictions_at IS NOT NULL",
    },
    {
      text: "SELECT c.ticker,i.instrument_type,c.last_date,c.revision,c.updated_at FROM liquidity.instrument_checkpoints c JOIN liquidity.instruments i USING(ticker) ORDER BY c.ticker",
    },
    {
      text: "SELECT stage,through_date,status,started_at,finished_at,rows_processed,details FROM liquidity.intelligence_runs ORDER BY started_at DESC LIMIT 10",
    },
    {
      text: "SELECT mode,status,range_end,finished_at,error_code FROM liquidity.ingestion_runs ORDER BY started_at DESC LIMIT 1",
    },
    {
      text: "SELECT feature_version::text,contract_sha256,model_hashes,thresholds FROM liquidity.intelligence_versions",
    },
    {
      text: "SELECT count(*)::int AS observations,count(hre_label)::int AS matured_labels,max(horizon_end)::text AS latest_outcome_date FROM liquidity.intelligence_observations",
    },
  ]);
  let github: Row = {
    status: "UNAVAILABLE",
    note: "GitHub execution details could not be retrieved.",
  };
  try {
    const response = await fetch(
      "https://api.github.com/repos/vighnesh-6/Indian_Equities_Liquidity_Risk_ML-main/actions/workflows/liquidity-risk-daily-ingestion.yml/runs?per_page=1",
      {
        headers: { Accept: "application/vnd.github+json" },
        next: { revalidate: 60 },
        signal: AbortSignal.timeout(3500),
      },
    );
    if (response.ok) {
      const result = await response.json();
      const run = result.workflow_runs?.[0];
      if (run)
        github = {
          status: run.status,
          conclusion: run.conclusion,
          url: run.html_url,
          run_number: run.run_number,
          updated_at: run.updated_at,
        };
    }
  } catch {
    /* The database health remains independently visible. */
  }
  return {
    connectivity: "VERIFIED",
    target: "liquidity-risk-production",
    schema: "liquidity",
    coverage: coverage[0],
    features,
    intelligence: intelligence[0],
    checkpoints,
    runs,
    ingestion: ingestion[0] || null,
    versions,
    labels: labels[0],
    github,
    schedule: "Weekdays 16:00 IST / 10:30 UTC",
    expected_session: latestReadySession(),
    generated_at: new Date().toISOString(),
  };
}

export async function getResource(resource: string, params: URLSearchParams) {
  if (resource === "overview") return overview(params);
  if (resource === "screener") return screener(params);
  if (resource === "anomalies") return screener(params, true);
  if (resource === "stock") return stock(params);
  if (resource === "events") return events(params);
  if (resource === "status") return status();
  if (resource === "dates") {
    const meta = await context(params);
    const rows = await query(
      "SELECT date::text FROM liquidity.feature_publications WHERE feature_version=$1::uuid ORDER BY date DESC",
      [meta.version],
    );
    return { meta, dates: rows.map((row) => row.date) };
  }
  throw new InputError("Unknown API resource.");
}
