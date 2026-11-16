// An in-memory PostgreSQL test backend. No network or Neon credentials are used.
import { createInterface } from 'node:readline';
import { pathToFileURL } from 'node:url';

const { PGlite } = await import(pathToFileURL(process.env.LIQUIDITY_TEST_PGLITE_MODULE).href);
const database = await PGlite.create('memory://');
const version = await database.query('SELECT version()');
console.log(JSON.stringify({ ready: true, version: version.rows[0].version }));

for await (const line of createInterface({ input: process.stdin })) {
  try {
    const request = JSON.parse(line);
    const result = request.multi
      ? await database.exec(request.sql)
      : await database.query(request.sql, request.params);
    console.log(JSON.stringify({ ok: true, result }, (_, value) =>
      typeof value === 'bigint' ? value.toString() : value));
  } catch (error) {
    console.log(JSON.stringify({ ok: false, code: error.code ?? 'TEST_SQL_ERROR',
      message: error.message, position: error.position }));
  }
}
await database.close();
