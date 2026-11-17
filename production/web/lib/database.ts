import "server-only";
import { neon } from "@neondatabase/serverless";

const HOST = "ep-misty-shadow-azik30ky-pooler.c-3.ap-southeast-1.aws.neon.tech";
const TARGET = [
  "super-flower-30495255",
  "br-winter-thunder-azx88rt3",
  "ep-misty-shadow-azik30ky",
];
export type Statement = { text: string; values?: unknown[] };
export type Row = Record<string, any>;

export async function queryMany(
  statements: Statement[],
  expectedFeatureVersion?: string,
): Promise<Row[][]> {
  const value = process.env.LIQUIDITY_NEON_DATABASE_URL;
  if (!value) throw new Error("DATABASE_CONFIGURATION_MISSING");
  const uri = new URL(value);
  if (
    !["postgres:", "postgresql:"].includes(uri.protocol) ||
    uri.hostname !== HOST ||
    uri.pathname !== "/neondb" ||
    (uri.port && uri.port !== "5432")
  ) {
    throw new Error("DATABASE_IDENTITY_MISMATCH");
  }
  const sql = neon(value);
  const gate = sql.query(
    `SELECT 1 / coalesce((current_database()='neondb'
    AND current_user=$1 AND current_setting('neon.project_id',true)=$2
    AND current_setting('neon.branch_id',true)=$3
    AND current_setting('neon.endpoint_id',true)=$4),false)::int AS verified`,
    [decodeURIComponent(uri.username), ...TARGET],
  );
  const guards = [
    gate,
    sql.query(
      "SELECT set_config('search_path','pg_catalog',true),set_config('statement_timeout','10000',true)",
    ),
  ];
  if (expectedFeatureVersion) {
    guards.push(
      sql.query(
        `SELECT 1 / (EXISTS(SELECT 1 FROM liquidity.feature_versions WHERE feature_version=$1::uuid
      AND source_version='canonical-basis-'||(SELECT coalesce(max(generation_id),0)::text FROM liquidity.price_basis_generations WHERE state='ACTIVE')))::int AS vintage_verified`,
        [expectedFeatureVersion],
      ),
    );
  }
  for (let attempt = 0; ; attempt++) {
    try {
      const results = await sql.transaction(
        [
          ...guards,
          ...statements.map((statement) =>
            sql.query(statement.text, statement.values || []),
          ),
        ],
        {
          readOnly: true,
          isolationLevel: "RepeatableRead",
          fetchOptions: {
            signal: AbortSignal.timeout(12000),
            cache: "no-store",
          },
        },
      );
      return results.slice(guards.length) as Row[][];
    } catch (error) {
      const failure = error as {
        name?: string;
        message?: string;
        code?: string;
        cause?: { code?: string };
        sourceError?: {
          name?: string;
          message?: string;
          cause?: { code?: string };
        };
      };
      const source = failure.sourceError || failure;
      const code =
        failure.code || failure.cause?.code || source.cause?.code || "";
      const transient =
        source.name === "TimeoutError" ||
        [
          "08000",
          "08003",
          "08006",
          "57P01",
          "ECONNRESET",
          "ETIMEDOUT",
          "UND_ERR_CONNECT_TIMEOUT",
        ].includes(code) ||
        (source.name === "TypeError" && source.message === "fetch failed");
      if (attempt !== 0 || !transient) throw error;
      await new Promise((resolve) => setTimeout(resolve, 250));
    }
  }
}

export async function query(
  text: string,
  values: unknown[] = [],
  expectedFeatureVersion?: string,
): Promise<Row[]> {
  return (await queryMany([{ text, values }], expectedFeatureVersion))[0];
}
