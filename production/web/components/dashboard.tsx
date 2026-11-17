"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import {
  Activity,
  ArrowDownRight,
  ArrowUpRight,
  CalendarDays,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Database,
  ExternalLink,
  Layers,
  LayoutDashboard,
  Loader2,
  Menu,
  Radar,
  RefreshCw,
  Search,
  ShieldCheck,
  X,
  AlertTriangle,
} from "lucide-react";
import { Candles, RiskChart, VolumeChart } from "./charts";

type Row = Record<string, any>;
const navigation = [
  { id: "overview", name: "Market Overview", icon: LayoutDashboard, path: "/" },
  {
    id: "screener",
    name: "Liquidity Screener",
    icon: Layers,
    path: "/screener",
  },
  { id: "stock", name: "Stock Deep-Dive", icon: Activity, path: "/stock" },
  {
    id: "anomalies",
    name: "Anomaly Surveillance",
    icon: Radar,
    path: "/anomalies",
  },
  {
    id: "events",
    name: "Event Intelligence",
    icon: CalendarDays,
    path: "/events",
  },
  {
    id: "status",
    name: "Data & System Status",
    icon: Database,
    path: "/status",
  },
];
const descriptions: Record<string, string> = {
  overview:
    "A cross-sectional view of liquidity, forward risk and market unusualness.",
  screener:
    "Explore the canonical 300-equity universe with date-specific risk intelligence.",
  stock:
    "Investigate price action, liquidity conditions and frozen-model signals.",
  anomalies: "Find unusual market behaviour with the frozen Isolation Forest.",
  events:
    "Verified market events, original sources and observed market context.",
  status:
    "Coverage, pipeline execution and the provenance behind every published signal.",
};
const n = (value: any, digits = 1) =>
  value == null || !Number.isFinite(Number(value))
    ? "\u2014"
    : Number(value).toLocaleString("en-IN", {
        maximumFractionDigits: digits,
        minimumFractionDigits: digits,
      });
const count = (value: any) =>
  value == null ? "\u2014" : Number(value).toLocaleString("en-IN");
const pct = (value: any) =>
  value == null ? "\u2014" : n(Number(value) * 100, 1) + "%";
const dateLabel = (value: string) =>
  value
    ? new Date(value.slice(0, 10) + "T12:00:00Z").toLocaleDateString("en-GB", {
        day: "2-digit",
        month: "short",
        year: "numeric",
      })
    : "\u2014";
const symbol = (ticker: string) => ticker.replace(".NS", "");
function bandClass(band: string) {
  return (
    (
      {
        LOW: "low",
        MODERATE: "moderate",
        HIGH: "high",
        VERY_HIGH: "very-high",
      } as Record<string, string>
    )[band] || "unavailable"
  );
}

function Badge({
  band,
  children,
}: {
  band?: string;
  children?: React.ReactNode;
}) {
  return (
    <span className={"badge " + bandClass(band || "")}>
      {children || band?.replace("_", " ") || "UNAVAILABLE"}
    </span>
  );
}
function Metric({
  label,
  value,
  detail,
  accent = false,
  icon,
}: {
  label: string;
  value: string;
  detail?: React.ReactNode;
  accent?: boolean;
  icon?: React.ReactNode;
}) {
  return (
    <section className={"metric " + (accent ? "accent" : "")}>
      <div className="metric-label">
        {label}
        {icon}
      </div>
      <div className="metric-value">{value}</div>
      <div className="metric-detail">{detail || "\u00a0"}</div>
    </section>
  );
}
function Panel({
  title,
  subtitle,
  children,
  action,
  className = "",
}: {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
  action?: React.ReactNode;
  className?: string;
}) {
  return (
    <section className={"panel " + className}>
      <div className="panel-heading">
        <div>
          <h2>{title}</h2>
          {subtitle && <p>{subtitle}</p>}
        </div>
        {action}
      </div>
      {children}
    </section>
  );
}

export default function Dashboard({ page }: { page: string }) {
  const [selectedDate, setDate] = useState("");
  const [ticker, setTicker] = useState("RELIANCE.NS");
  const [search, setSearch] = useState("");
  const [cap, setCap] = useState("");
  const [risk, setRisk] = useState("");
  const [sort, setSort] = useState(page === "anomalies" ? "anomaly" : "lrs");
  const [direction, setDirection] = useState("desc");
  const [tablePage, setTablePage] = useState(1);
  const [flagged, setFlagged] = useState(false);
  const [allEvents, setAllEvents] = useState(false);
  const [range, setRange] = useState("1Y");
  const [data, setData] = useState<Row | null>(null);
  const [dates, setDates] = useState<Row | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [refresh, setRefresh] = useState(0);
  const [menu, setMenu] = useState(false);

  useEffect(() => {
    const query = new URLSearchParams(window.location.search);
    if (query.get("date")) setDate(query.get("date")!);
    if (query.get("ticker")) setTicker(query.get("ticker")!);
    setSort(page === "anomalies" ? "anomaly" : "lrs");
    setTablePage(1);
    setMenu(false);
    fetch("/api/dates")
      .then((response) => response.json())
      .then((result) => {
        if (result.meta) {
          setDates(result);
          setDate(
            (previous) =>
              previous ||
              result.meta.latest_prediction_date ||
              result.meta.latest_feature_date,
          );
        }
      })
      .catch(() => {});
  }, [page]);

  useEffect(() => {
    const controller = new AbortController();
    const timer = setTimeout(
      async () => {
        setLoading(true);
        setError("");
        const params = new URLSearchParams({
          page: String(tablePage),
          limit: "25",
          sort,
          direction,
        });
        if (selectedDate) params.set("date", selectedDate);
        if (search) params.set("search", search);
        if (cap) params.set("cap", cap);
        if (risk) params.set("risk", risk);
        if (flagged) params.set("flagged", "1");
        if (allEvents) params.set("all", "1");
        if (page === "stock") {
          params.set("ticker", ticker);
          if (selectedDate && range !== "ALL") {
            const start = new Date(selectedDate + "T12:00:00Z");
            start.setUTCDate(
              start.getUTCDate() -
                ({ "3M": 90, "6M": 180, "1Y": 365 } as Record<string, number>)[
                  range
                ],
            );
            params.set("from", start.toISOString().slice(0, 10));
          }
        }
        try {
          const response = await fetch("/api/" + page + "?" + params, {
            signal: controller.signal,
            cache: "no-store",
          });
          const result = await response.json();
          if (!response.ok)
            throw new Error(result.message || "Data request failed.");
          if (!controller.signal.aborted) setData(result);
        } catch (failure) {
          if (!controller.signal.aborted) {
            setError(
              failure instanceof Error
                ? failure.message
                : "Data request failed.",
            );
            setData(null);
          }
        } finally {
          if (!controller.signal.aborted) setLoading(false);
        }
      },
      search ? 250 : 0,
    );
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [
    page,
    selectedDate,
    ticker,
    search,
    cap,
    risk,
    sort,
    direction,
    tablePage,
    flagged,
    allEvents,
    range,
    refresh,
  ]);

  const changeDate = (value: string) => {
    setData(null);
    setDate(value);
    setTablePage(1);
    const params = new URLSearchParams(window.location.search);
    params.set("date", value);
    window.history.replaceState(
      null,
      "",
      window.location.pathname + "?" + params,
    );
  };
  const href = (path: string, stock?: string) =>
    path +
    "?" +
    new URLSearchParams({
      ...(selectedDate ? { date: selectedDate } : {}),
      ...(stock ? { ticker: stock } : {}),
    });
  const current = navigation.find((item) => item.id === page)!;
  const meta = data?.meta || dates?.meta;
  const stale = Boolean(meta?.stale);

  return (
    <div className="application">
      <aside className={"sidebar " + (menu ? "open" : "")}>
        <Link href={href("/")} className="brand">
          <span className="brand-mark">
            <Activity size={22} />
          </span>
          <span>
            LIQUIDITY<small>RISK INTELLIGENCE</small>
          </span>
        </Link>
        <div className="workspace">
          <span className="workspace-icon">IN</span>
          <div>
            Indian Equities<small>NSE · 300-stock universe</small>
          </div>
          <ChevronRight size={15} />
        </div>
        <div className="nav-label">RESEARCH TERMINAL</div>
        <nav aria-label="Main navigation">
          {navigation.map((item) => (
            <Link
              key={item.id}
              href={href(item.path)}
              className={page === item.id ? "active" : ""}
            >
              <item.icon size={17} />
              <span>{item.name}</span>
              {page === item.id && <span className="nav-active-dot" />}
            </Link>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="model-note">
            <ShieldCheck size={18} />
            <div>
              Frozen research models<small>HGBT + Isolation Forest</small>
            </div>
          </div>
          <div className="sidebar-date">
            <span className="dot" /> Historical intelligence · NSE
          </div>
          <p>
            Research and decision support.
            <br />
            Not investment advice.
          </p>
        </div>
      </aside>
      {menu && (
        <button
          aria-label="Close navigation"
          className="mobile-shade"
          onClick={() => setMenu(false)}
        />
      )}
      <div className="main-shell">
        <header className="topbar">
          <div className="breadcrumb">
            <button
              className="icon-button mobile-toggle"
              aria-label="Open navigation"
              onClick={() => setMenu(!menu)}
            >
              {menu ? <X size={20} /> : <Menu size={20} />}
            </button>
            <span>Research Terminal</span>
            <ChevronRight size={13} />
            <strong>{current.name}</strong>
          </div>
          <div className="topbar-right">
            <span className="exchange-label">
              <span className="dot" /> NSE INDIA
            </span>
            <span className="avatar">LR</span>
          </div>
        </header>
        <main>
          <div className="page-heading">
            <div>
              <div className="eyebrow">
                INDIAN EQUITIES / LIQUIDITY INTELLIGENCE
              </div>
              <h1>
                {page === "stock" && data?.stock
                  ? symbol(data.stock.ticker)
                  : current.name}
              </h1>
              <p>
                {page === "stock" && data?.stock
                  ? data.stock.company_name
                  : descriptions[page]}
              </p>
            </div>
            <div className="page-actions">
              {page !== "status" && (
                <label className="date-picker">
                  <CalendarDays size={15} />
                  <input
                    aria-label="Historical trading date"
                    type="date"
                    value={selectedDate}
                    min={dates?.meta?.first_date}
                    max={dates?.meta?.latest_market_date}
                    onChange={(event) => changeDate(event.target.value)}
                  />
                </label>
              )}
              <button
                className="icon-button refresh"
                onClick={() => setRefresh((value) => value + 1)}
                aria-label="Refresh production data"
              >
                <RefreshCw size={16} className={loading ? "spin" : ""} />
              </button>
            </div>
          </div>
          {page !== "status" && (
            <div className="context-line">
              <span className="dot" />
              <span>
                Observations as of <strong>{dateLabel(selectedDate)}</strong>
              </span>
              <span className="context-divider" />
              <span>Frozen HGBT · HRE-5D horizon</span>
              <span className="context-divider" />
              <span>
                {selectedDate && selectedDate < "2025-10-20"
                  ? "Historical training-period model replay"
                  : "Out-of-sample model inference"}
              </span>
            </div>
          )}
          {stale && page !== "status" && (
            <div className="notice">
              <AlertTriangle size={16} />
              <span>
                Predictions for this session are pending. Latest completed
                predictions {dateLabel(meta.latest_prediction_date)}.
              </span>
            </div>
          )}
          {error && (
            <div className="error-panel">
              <AlertTriangle size={26} />
              <h2>Production data unavailable</h2>
              <p>{error}</p>
              <button
                className="button"
                onClick={() => setRefresh((value) => value + 1)}
              >
                Retry connection <RefreshCw size={14} />
              </button>
            </div>
          )}
          {loading && !data && !error && (
            <div className="loading-panel">
              <Loader2 className="spin" size={26} />
              <p>Reading verified production data…</p>
            </div>
          )}
          {data && !error && (
            <div className={loading ? "content updating" : "content"}>
              {page === "overview" && <Overview data={data} link={href} />}
              {(page === "screener" || page === "anomalies") && (
                <>
                  {page === "anomalies" && (
                    <div className="notice neutral">
                      <Radar size={16} />
                      <span>
                        Isolation Forest measures historical unusualness. HRE-5D
                        is the separate forward-risk forecast.
                      </span>
                    </div>
                  )}
                  {page === "anomalies" && data.timeline?.length > 0 && (
                    <Panel
                      title="Historical anomaly surveillance"
                      subtitle="Flagged equities by date, using the frozen model threshold"
                    >
                      <RiskChart
                        rows={data.timeline}
                        field="flagged"
                        height={210}
                        color="#b5a0ed"
                      />
                    </Panel>
                  )}
                  <div className="filterbar">
                    <label className="search">
                      <Search size={16} />
                      <input
                        aria-label="Search stocks"
                        placeholder="Search company or ticker…"
                        value={search}
                        onChange={(event) => {
                          setSearch(event.target.value);
                          setTablePage(1);
                        }}
                      />
                    </label>
                    <select
                      aria-label="Capitalisation filter"
                      value={cap}
                      onChange={(event) => {
                        setCap(event.target.value);
                        setTablePage(1);
                      }}
                    >
                      <option value="">All capitalisations</option>
                      <option value="large">Large cap</option>
                      <option value="mid">Mid cap</option>
                      <option value="small">Small cap</option>
                    </select>
                    <select
                      aria-label="Risk filter"
                      value={risk}
                      onChange={(event) => {
                        setRisk(event.target.value);
                        setTablePage(1);
                      }}
                    >
                      <option value="">All risk bands</option>
                      <option value="LOW">Low</option>
                      <option value="MODERATE">Moderate</option>
                      <option value="HIGH">High</option>
                      <option value="VERY_HIGH">Very high</option>
                    </select>
                    {page === "anomalies" && (
                      <label className="checkbox">
                        <input
                          type="checkbox"
                          checked={flagged}
                          onChange={(event) => {
                            setFlagged(event.target.checked);
                            setTablePage(1);
                          }}
                        />{" "}
                        Flagged only
                      </label>
                    )}
                  </div>
                  <Panel
                    title={
                      page === "anomalies"
                        ? "Anomaly surveillance"
                        : "Equity liquidity screener"
                    }
                    subtitle={
                      count(data.total) + " instruments match this selection"
                    }
                    action={
                      <span className="panel-caption">
                        {dateLabel(selectedDate)}
                      </span>
                    }
                  >
                    <StockTable
                      rows={data.rows}
                      link={href}
                      anomaly={page === "anomalies"}
                      sort={(field) => {
                        setSort(field);
                        setDirection((value) =>
                          value === "desc" ? "asc" : "desc",
                        );
                      }}
                    />
                    <div className="table-footer">
                      <span>
                        {data.total ? (tablePage - 1) * 25 + 1 : 0}–
                        {Math.min(tablePage * 25, data.total)} of{" "}
                        {count(data.total)}
                      </span>
                      <div>
                        <button
                          aria-label="Previous page"
                          className="icon-button"
                          disabled={tablePage === 1}
                          onClick={() => setTablePage((value) => value - 1)}
                        >
                          <ChevronLeft size={16} />
                        </button>
                        <span>Page {tablePage}</span>
                        <button
                          aria-label="Next page"
                          className="icon-button"
                          disabled={tablePage * 25 >= data.total}
                          onClick={() => setTablePage((value) => value + 1)}
                        >
                          <ChevronRight size={16} />
                        </button>
                      </div>
                    </div>
                  </Panel>
                </>
              )}
              {page === "stock" && (
                <StockDetail
                  data={data}
                  range={range}
                  setRange={(value) => {
                    setData(null);
                    setRange(value);
                  }}
                  ticker={ticker}
                  setTicker={(value) => {
                    setData(null);
                    setTicker(value);
                  }}
                />
              )}
              {page === "events" && (
                <>
                  <div className="filterbar">
                    <div className="muted">
                      Verified research event registry · original source links
                    </div>
                    <label className="checkbox">
                      <input
                        type="checkbox"
                        checked={allEvents}
                        onChange={(event) => setAllEvents(event.target.checked)}
                      />{" "}
                      Full historical timeline
                    </label>
                  </div>
                  <div className="event-grid">
                    {data.rows.map((event: Row) => (
                      <article className="event-card" key={event.event_id}>
                        <div className="event-date">
                          <CalendarDays size={15} />
                          {dateLabel(event.event_date)}
                          <span className="verified">
                            <CheckCircle2 size={12} /> Verified
                          </span>
                        </div>
                        <span className="event-category">
                          {event.category.replaceAll("_", " ")}
                        </span>
                        <h2>{event.name}</h2>
                        <p>{event.description}</p>
                        <div className="event-footer">
                          <a
                            href={event.source_url}
                            target="_blank"
                            rel="noreferrer"
                          >
                            {event.source_name}
                            <ExternalLink size={12} />
                          </a>
                          {event.nifty_close != null && (
                            <span>NIFTY {n(event.nifty_close, 2)}</span>
                          )}
                        </div>
                      </article>
                    ))}
                  </div>
                  {!data.rows.length && (
                    <div className="empty-panel">
                      No verified events in this window. Select the full
                      historical timeline to browse available events.
                    </div>
                  )}
                  <div className="method-note">
                    Event dates and descriptions come from the verified stored
                    registry. Observed market levels provide context and do not
                    establish causation.
                  </div>
                </>
              )}
              {page === "status" && <SystemStatus data={data} />}
            </div>
          )}
          <footer className="footer">
            <span>
              LIQUIDITY <span className="footer-divider">/</span> INDIAN
              EQUITIES RISK INTELLIGENCE
            </span>
            <span>
              Research decision support · daily OHLCV liquidity proxies
            </span>
          </footer>
        </main>
      </div>
    </div>
  );
}

function Overview({
  data,
  link,
}: {
  data: Row;
  link: (path: string, ticker?: string) => string;
}) {
  const nifty = data.benchmarks.find((row: Row) => row.ticker === "^NSEI");
  const vix = data.benchmarks.find((row: Row) => row.ticker === "^INDIAVIX");
  const summary = data.summary;
  const change = (row: Row | undefined) => (
    <span className={Number(row?.change_pct) >= 0 ? "positive" : "negative"}>
      {row?.change_pct == null ? (
        "Unavailable"
      ) : (
        <>
          {Number(row.change_pct) >= 0 ? (
            <ArrowUpRight size={13} />
          ) : (
            <ArrowDownRight size={13} />
          )}{" "}
          {n(row.change_pct, 2)}%
        </>
      )}
      <span className="muted"> vs previous session</span>
    </span>
  );
  return (
    <>
      <div className="metrics">
        <Metric
          label="NIFTY 50"
          value={n(nifty?.close, 2)}
          detail={change(nifty)}
          icon={<Activity size={15} />}
        />
        <Metric
          label="INDIA VIX"
          value={n(vix?.close, 2)}
          detail={change(vix)}
          icon={<Radar size={15} />}
        />
        <Metric
          label="MEAN LIQUIDITY RISK"
          value={n(summary.average_lrs)}
          detail={count(summary.scored) + " stocks with available LRS"}
          accent
          icon={<Layers size={15} />}
        />
        <Metric
          label="FORWARD RISK FLAGS"
          value={count(summary.hre_flags)}
          detail={
            <>
              <span className="amber">HRE-5D</span> · frozen probability
              threshold 53.4%
            </>
          }
          icon={<ArrowUpRight size={15} />}
        />
      </div>
      <div className="overview-grid">
        <Panel
          title="High-risk liquidity breadth"
          subtitle="Equities above frozen segment thresholds · selected historical window"
          action={
            <span className="chart-key">
              <span className="dot" /> High-risk share
            </span>
          }
        >
          <RiskChart rows={data.trend} field="risk_breadth" />
          <div className="chart-footer">
            <span>
              {count(summary.observed)} observations / 300-equity universe
            </span>
            <span>Share of scored equities · 0–100</span>
          </div>
        </Panel>
        <Panel
          title="Risk by capitalisation"
          subtitle="Daily ranks within each static segment"
        >
          <div className="cap-list">
            {data.caps.map((row: Row) => (
              <div key={row.cap_segment} className="cap-item">
                <div>
                  <span className="cap-icon">
                    {row.cap_segment.slice(0, 1).toUpperCase()}
                  </span>
                  <div className="cap-name">
                    {row.cap_segment} cap
                    <small>{count(row.observed)} observed equities</small>
                  </div>
                  <strong>{n(row.average_lrs)}</strong>
                </div>
                <div className="track">
                  <span style={{ width: (row.average_lrs || 0) + "%" }} />
                </div>
              </div>
            ))}
          </div>
          <div className="mini-summary">
            <span>Isolation Forest flags</span>
            <strong>{count(summary.anomaly_flags)}</strong>
          </div>
          <div className="mini-summary">
            <span>LRS ≥ 75</span>
            <strong>{count(summary.very_high)}</strong>
          </div>
        </Panel>
      </div>
      <Panel
        title="Highest liquidity risk"
        subtitle="Stocks ranked by the selected date’s composite LRS"
        action={
          <Link className="text-link" href={link("/screener")}>
            View all equities <ChevronRight size={14} />
          </Link>
        }
      >
        <StockTable rows={data.leaders} link={link} />
      </Panel>
      <Panel
        title="Lowest liquidity risk"
        subtitle="Lowest composite LRS for the selected date"
      >
        <StockTable rows={data.lowest_risk || []} link={link} />
      </Panel>
      <div className="method-note">
        <ShieldCheck size={15} />
        <span>
          LRS combines within-segment illiquidity, spread, inverse volume and
          downside-volatility ranks. HRE-5D forecasts the next five
          observations. Anomaly scores measure unusualness against the frozen
          training distribution.
        </span>
      </div>
    </>
  );
}

function StockTable({
  rows,
  link,
  anomaly = false,
  sort,
}: {
  rows: Row[];
  link: (path: string, ticker?: string) => string;
  anomaly?: boolean;
  sort?: (value: string) => void;
}) {
  return (
    <div className="table-scroll">
      <table>
        <thead>
          <tr>
            <th>
              <button onClick={() => sort?.("ticker")}>Company / ticker</button>
            </th>
            <th>Segment</th>
            <th className="numeric">
              <button onClick={() => sort?.("lrs")}>LRS ↕</button>
            </th>
            <th>Risk band</th>
            <th className="numeric">
              <button onClick={() => sort?.("probability")}>
                HRE-5D probability ↕
              </button>
            </th>
            <th className="numeric">
              <button onClick={() => sort?.("anomaly")}>
                Anomaly score ↕
              </button>
            </th>
            {anomaly && <th>Isolation flag</th>}
            <th />
          </tr>
        </thead>
        <tbody>
          {rows.map((row: Row) => (
            <tr key={row.ticker}>
              <td>
                <Link href={link("/stock", row.ticker)} className="stock-name">
                  <span className="ticker-square">
                    {symbol(row.ticker).slice(0, 2)}
                  </span>
                  <span>
                    <strong>{symbol(row.ticker)}</strong>
                    <small>{row.company_name}</small>
                  </span>
                </Link>
              </td>
              <td>
                <span className="segment">{row.cap_segment}</span>
              </td>
              <td className="numeric">
                <span className="score-number">{n(row.lrs)}</span>
                <div className="score-track">
                  <span
                    className={bandClass(row.lrs_band)}
                    style={{ width: (row.lrs || 0) + "%" }}
                  />
                </div>
              </td>
              <td>
                <Badge band={row.lrs_band} />
              </td>
              <td
                className={
                  "numeric " + (row.hre_probability >= 0.534 ? "amber" : "")
                }
              >
                {pct(row.hre_probability)}
                {row.prediction_status &&
                  row.prediction_status !== "AVAILABLE" && (
                    <small className="cell-status">
                      {row.prediction_status.replaceAll("_", " ")}
                    </small>
                  )}
              </td>
              <td className="numeric mono">{n(row.anomaly_score_scaled)}</td>
              {anomaly && (
                <td>
                  {row.is_anomaly == null ? (
                    <span className="muted">Unavailable</span>
                  ) : (
                    <Badge band={row.is_anomaly ? "HIGH" : "LOW"}>
                      {row.is_anomaly ? "FLAGGED" : "NORMAL"}
                    </Badge>
                  )}
                </td>
              )}
              <td>
                <Link
                  aria-label={"Open " + row.ticker}
                  href={link("/stock", row.ticker)}
                >
                  <ChevronRight size={16} />
                </Link>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {!rows.length && (
        <div className="empty-panel">No observations match this selection.</div>
      )}
    </div>
  );
}

function StockDetail({
  data,
  range,
  setRange,
  ticker,
  setTicker,
}: {
  data: Row;
  range: string;
  setRange: (value: string) => void;
  ticker: string;
  setTicker: (value: string) => void;
}) {
  const snapshot = data.snapshot;
  const latest = data.history[data.history.length - 1];
  const [draft, setDraft] = useState(ticker);
  return (
    <>
      <div className="stock-toolbar">
        <form
          className="search"
          onSubmit={(event) => {
            event.preventDefault();
            setTicker(
              draft.toUpperCase().endsWith(".NS")
                ? draft.toUpperCase()
                : draft.toUpperCase() + ".NS",
            );
          }}
        >
          <Search size={16} />
          <input
            aria-label="Stock ticker"
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            placeholder="RELIANCE.NS"
          />
          <button type="submit" className="text-link">
            Go <ChevronRight size={14} />
          </button>
        </form>
        <Badge>{data.stock.cap_segment.toUpperCase() + " CAP"}</Badge>
        <span className="muted">
          {data.stock.sector || "Sector not supplied"} · {data.stock.isin}
        </span>
        <div className="range-controls">
          {["3M", "6M", "1Y", "ALL"].map((value) => (
            <button
              key={value}
              onClick={() => setRange(value)}
              className={range === value ? "selected" : ""}
            >
              {value}
            </button>
          ))}
        </div>
      </div>
      <div className="metrics">
        <Metric
          label="CLOSE PRICE"
          value={latest ? "₹" + n(latest.close, 2) : "\u2014"}
          detail={dateLabel(latest?.date)}
        />
        <Metric
          label="LIQUIDITY RISK SCORE"
          value={n(snapshot?.lrs)}
          detail={<Badge band={snapshot?.lrs_band} />}
          accent
        />
        <Metric
          label="HRE-5D PROBABILITY"
          value={pct(snapshot?.hre_probability)}
          detail={
            snapshot?.prediction_status === "AVAILABLE"
              ? snapshot.hre_flag
                ? "High-risk event forecast"
                : "Below frozen decision threshold"
              : (snapshot?.prediction_status || "Unavailable").replaceAll(
                  "_",
                  " ",
                )
          }
        />
        <Metric
          label="ANOMALY SCORE"
          value={n(snapshot?.anomaly_score_scaled)}
          detail={
            snapshot?.is_anomaly == null
              ? "Unavailable"
              : snapshot.is_anomaly
                ? "Flagged by Isolation Forest"
                : "Within frozen training distribution"
          }
        />
      </div>
      <Panel
        title="Price & traded volume"
        subtitle="Observed OHLCV · candlesticks show the most recent 90 observations"
      >
        <Candles rows={data.history} />
        <VolumeChart rows={data.history.slice(-90)} />
      </Panel>
      <div className="two-columns">
        <Panel
          title="Historical liquidity risk"
          subtitle="Stored date-specific LRS · 0–100"
        >
          <RiskChart rows={data.history} field="lrs" height={220} />
        </Panel>
        <Panel
          title="Historical anomaly score"
          subtitle="Frozen Isolation Forest · higher is more unusual"
        >
          <RiskChart
            rows={data.history}
            field="anomaly_score_scaled"
            height={220}
            color="#b5a0ed"
          />
        </Panel>
      </div>
      <div className="two-columns">
        <Panel
          title="Liquidity & volatility inputs"
          subtitle="Approved feature values for the selected date"
        >
          <div className="key-values">
            {[
              ["Amihud MA20", "amihud_ma20"],
              ["Spread proxy MA10", "cs_spread_ma10"],
              ["Volume z-score", "volume_z20"],
              ["Downside volatility", "downside_vol_20d"],
              ["RSI 14", "rsi_14"],
              ["Realised volatility", "vol_20d"],
            ].map(([label, key]) => (
              <div key={key}>
                <span>{label}</span>
                <strong>
                  {snapshot?.[key] == null
                    ? "\u2014"
                    : Number(snapshot[key]).toPrecision(5)}
                </strong>
              </div>
            ))}
          </div>
        </Panel>
        <Panel
          title="Forecast & realised outcome"
          subtitle="Future outcomes are separate from prediction inputs"
        >
          <div className="key-values">
            <div>
              <span>HGBT decision threshold</span>
              <strong>53.4%</strong>
            </div>
            <div>
              <span>Model evaluation period</span>
              <strong>
                {snapshot?.model_period === "TRAINING_PERIOD_REPLAY"
                  ? "Retrospective replay"
                  : "Out of sample"}
              </strong>
            </div>
            <div>
              <span>Actual HRE-5D outcome</span>
              <strong>
                {snapshot?.hre_label == null
                  ? "Pending / unavailable"
                  : snapshot.hre_label
                    ? "High-risk event"
                    : "No high-risk event"}
              </strong>
            </div>
            <div>
              <span>Outcome horizon ended</span>
              <strong>{dateLabel(snapshot?.horizon_end)}</strong>
            </div>
            <div>
              <span>Label availability</span>
              <strong>
                {snapshot?.label_status?.replaceAll("_", " ") || "Unavailable"}
              </strong>
            </div>
          </div>
        </Panel>
      </div>
    </>
  );
}

function SystemStatus({ data }: { data: Row }) {
  const good =
    data.coverage.instruments === 302 &&
    data.coverage.earliest_checkpoint === data.coverage.latest_checkpoint &&
    (!data.expected_session ||
      data.coverage.latest_checkpoint >= data.expected_session) &&
    data.intelligence.latest_date === data.coverage.latest_checkpoint;
  return (
    <>
      <div className={"system-banner " + (good ? "healthy" : "")}>
        <ShieldCheck size={25} />
        <div>
          <strong>
            {good ? "Market coverage verified" : "Coverage needs attention"}
          </strong>
          <span>
            Dedicated Liquidity Risk database ·{" "}
            {dateLabel(data.coverage.latest_checkpoint)} market checkpoint
          </span>
        </div>
        <Badge band={good ? "LOW" : "HIGH"}>
          {good ? "302 / 302" : "CHECK COVERAGE"}
        </Badge>
      </div>
      <div className="metrics">
        <Metric
          label="MARKET INSTRUMENTS"
          value={count(data.coverage.instruments)}
          detail="300 equities + NIFTY 50 + INDIA VIX"
        />
        <Metric
          label="FEATURE EQUITIES"
          value={count(data.features[0]?.equities)}
          detail={dateLabel(data.features[0]?.checkpoint)}
        />
        <Metric
          label="MATURED HRE-5D LABELS"
          value={count(data.labels.matured_labels)}
          detail={
            count(data.labels.observations) + " intelligence observations"
          }
        />
        <Metric
          label="PUBLISHED SCORE DATES"
          value={count(data.intelligence.published_dates)}
          detail={"Latest " + dateLabel(data.intelligence.latest_date)}
          accent
        />
      </div>
      <div className="two-columns">
        <Panel
          title="Production pipeline"
          subtitle="Weekday automation and current completion evidence"
        >
          <div className="key-values">
            <div>
              <span>Neon connectivity</span>
              <strong className="positive">Identity verified</strong>
            </div>
            <div>
              <span>Daily schedule</span>
              <strong>16:00 IST · weekdays</strong>
            </div>
            <div>
              <span>Last ingestion</span>
              <strong>{data.ingestion?.status || "Unavailable"}</strong>
            </div>
            <div>
              <span>Latest intelligence</span>
              <strong>{dateLabel(data.intelligence.latest_date)}</strong>
            </div>
            <div>
              <span>GitHub Actions</span>
              <strong>{data.github.conclusion || data.github.status}</strong>
            </div>
            {data.github.url && (
              <a
                href={data.github.url}
                target="_blank"
                rel="noreferrer"
                className="text-link"
              >
                View actual workflow run <ExternalLink size={13} />
              </a>
            )}
          </div>
        </Panel>
        <Panel
          title="Frozen model contract"
          subtitle="Original fitted models · inference only"
        >
          <div className="key-values">
            <div>
              <span>Champion</span>
              <strong>CFG_05_HGBT · 36 inputs</strong>
            </div>
            <div>
              <span>Anomaly model</span>
              <strong>CFG_06_IF · 12 inputs</strong>
            </div>
            <div>
              <span>HGBT threshold</span>
              <strong>0.534</strong>
            </div>
            <div>
              <span>Feature version</span>
              <strong className="version-text">
                {data.versions[0]?.feature_version || "Unavailable"}
              </strong>
            </div>
            <div>
              <span>Excluded special session</span>
              <strong>8 Nov 2026</strong>
            </div>
          </div>
        </Panel>
      </div>
      <Panel
        title="Recent intelligence executions"
        subtitle="Actual persisted worker runs"
      >
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Stage</th>
                <th>Through date</th>
                <th>Status</th>
                <th>Rows processed</th>
                <th>Finished</th>
              </tr>
            </thead>
            <tbody>
              {data.runs.map((row: Row, index: number) => (
                <tr key={index}>
                  <td>{row.stage}</td>
                  <td>{dateLabel(row.through_date)}</td>
                  <td>
                    <Badge
                      band={
                        row.status === "SUCCESS"
                          ? "LOW"
                          : row.status === "FAILED"
                            ? "HIGH"
                            : ""
                      }
                    >
                      {row.status}
                    </Badge>
                  </td>
                  <td className="mono">{count(row.rows_processed)}</td>
                  <td>
                    {row.finished_at
                      ? new Date(row.finished_at).toLocaleString("en-GB", {
                          timeZone: "Asia/Kolkata",
                        })
                      : "Running"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {!data.runs.length && (
            <div className="empty-panel">
              No intelligence execution records are available.
            </div>
          )}
        </div>
      </Panel>
      <Panel
        title="Instrument checkpoints"
        subtitle="Market ingestion state for every canonical instrument"
      >
        <div className="checkpoint-grid">
          {data.checkpoints.map((row: Row) => (
            <div key={row.ticker}>
              <span
                className={
                  "dot " +
                  (String(row.last_date).slice(0, 10) !==
                  data.coverage.latest_checkpoint
                    ? "warning"
                    : "")
                }
              />
              <strong>{symbol(row.ticker)}</strong>
              <span>{dateLabel(row.last_date)}</span>
            </div>
          ))}
        </div>
      </Panel>
    </>
  );
}
