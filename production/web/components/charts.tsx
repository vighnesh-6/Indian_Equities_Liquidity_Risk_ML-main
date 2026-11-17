"use client";

import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type Row = Record<string, any>;
const axis = { fill: "#82939b", fontSize: 10, fontFamily: "monospace" };
const tooltip = {
  background: "#142129",
  border: "1px solid #2a3b46",
  borderRadius: 8,
  color: "#e5edf1",
  fontSize: 12,
};
const shortDate = (value: string) => String(value).slice(5, 10);

export function RiskChart({
  rows,
  field = "average_lrs",
  height = 260,
  color = "#70dfba",
}: {
  rows: Row[];
  field?: string;
  height?: number;
  color?: string;
}) {
  if (!rows.some((row) => row[field] != null))
    return (
      <div className="chart-empty">No available values in this date range.</div>
    );
  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart
        data={rows}
        margin={{ top: 12, right: 10, left: -25, bottom: 0 }}
      >
        <defs>
          <linearGradient id={"fill-" + field} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor={color} stopOpacity={0.2} />
            <stop offset="1" stopColor={color} stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid
          stroke="#1d2b34"
          strokeDasharray="3 5"
          vertical={false}
        />
        <XAxis
          dataKey="date"
          tickFormatter={shortDate}
          tick={axis}
          axisLine={false}
          tickLine={false}
          minTickGap={35}
        />
        <YAxis
          tick={axis}
          axisLine={false}
          tickLine={false}
          domain={
            field.includes("lrs") || field.includes("score")
              ? [0, 100]
              : ["auto", "auto"]
          }
        />
        <Tooltip
          contentStyle={tooltip}
          formatter={(value: any) => Number(value).toFixed(2)}
        />
        <Area
          type="monotone"
          dataKey={field}
          stroke={color}
          strokeWidth={2}
          fill={"url(#fill-" + field + ")"}
          connectNulls={false}
          isAnimationActive={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}

export function VolumeChart({ rows }: { rows: Row[] }) {
  return (
    <ResponsiveContainer width="100%" height={120}>
      <BarChart
        data={rows}
        margin={{ top: 0, right: 10, left: -15, bottom: 0 }}
      >
        <XAxis dataKey="date" hide />
        <YAxis
          tick={axis}
          tickFormatter={(value) =>
            Number(value) >= 1e6
              ? (Number(value) / 1e6).toFixed(0) + "M"
              : (Number(value) / 1e3).toFixed(0) + "K"
          }
          axisLine={false}
          tickLine={false}
        />
        <Tooltip
          contentStyle={tooltip}
          formatter={(value: any) => Number(value).toLocaleString("en-IN")}
        />
        <Bar dataKey="volume" fill="#345b58" isAnimationActive={false} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function Candles({ rows }: { rows: Row[] }) {
  const data = rows.slice(-90);
  if (!data.length)
    return (
      <div className="chart-empty">
        No OHLCV observations for this selection.
      </div>
    );
  const low = Math.min(...data.map((row) => Number(row.low)));
  const high = Math.max(...data.map((row) => Number(row.high)));
  const range = high - low || high * 0.01 || 1;
  const y = (value: number) => 245 - ((value - low) / range) * 215;
  const step = 900 / data.length;
  return (
    <svg
      viewBox="0 0 1000 290"
      role="img"
      aria-label="Historical OHLC candlestick chart"
      className="candles"
    >
      {[0, 1, 2, 3, 4].map((index) => (
        <g key={index}>
          <line
            x1="8"
            y1={30 + index * 53.75}
            x2="918"
            y2={30 + index * 53.75}
            stroke="#20303a"
            strokeDasharray="3 5"
          />
          <text x="930" y={34 + index * 53.75} fill="#82939b" fontSize="11">
            {(high - (index * range) / 4).toLocaleString("en-IN", {
              maximumFractionDigits: 1,
            })}
          </text>
        </g>
      ))}
      {data.map((row, index) => {
        const color =
          Number(row.close) >= Number(row.open) ? "#70dfba" : "#e77e7e";
        const x = 12 + (index + 0.5) * step;
        return (
          <g key={row.date}>
            <title>
              {row.date +
                " | O " +
                row.open +
                " H " +
                row.high +
                " L " +
                row.low +
                " C " +
                row.close}
            </title>
            <line
              x1={x}
              x2={x}
              y1={y(Number(row.high))}
              y2={y(Number(row.low))}
              stroke={color}
            />
            <rect
              x={x - Math.max(1, step * 0.3)}
              y={Math.min(y(Number(row.open)), y(Number(row.close)))}
              width={Math.max(2, step * 0.6)}
              height={Math.max(
                1,
                Math.abs(y(Number(row.open)) - y(Number(row.close))),
              )}
              fill={color}
            />
          </g>
        );
      })}
      {[0, 0.25, 0.5, 0.75, 1].map((ratio) => {
        const index = Math.min(
          data.length - 1,
          Math.floor(ratio * (data.length - 1)),
        );
        return (
          <text
            key={ratio}
            x={12 + (index + 0.5) * step}
            y="275"
            textAnchor="middle"
            fill="#82939b"
            fontSize="11"
          >
            {String(data[index].date).slice(0, 10)}
          </text>
        );
      })}
    </svg>
  );
}
