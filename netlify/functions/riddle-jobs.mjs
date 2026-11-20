// POST /api/riddle-jobs        {riddle, solution, handle?} -> {id}; queues a job and starts the background run.
//                              handle is how the submitter appears on the leaderboard; omit it to stay anonymous.
// GET  /api/riddle-jobs/:id    -> the job's row, for polling.
// POST /api/riddle-jobs/:id/retry -> re-queues a job that ended in 'error'; only failed stages run again.
//
// Both POSTs need an X-Browser-Id header: a random UUID the page keeps in localStorage. It is
// the only thing rate limits are keyed on; no IP address or hash of one is stored.
import { getDatabase } from '@netlify/database';
import { EXPLAIN_SCALE, QUALITY } from '../lib/riddle-pipeline.mjs';

const MAX_RIDDLE = 2000;
const MAX_SOLUTION = 3000;
const MAX_HANDLE = 40;
// Every run spends real OpenRouter credit, so cap how many can start. Submissions and retries
// both count.
const PER_BROWSER_PER_HOUR = 5;
const GLOBAL_PER_DAY = 60;

// Background functions are killed after 15 minutes, and every stage refreshes updated_at, so a
// job that has not moved for longer than this has died without recording why.
const STALE_MINUTES = 16;

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

const error = (status, message) => Response.json({ error: message }, { status });

// Records one run for this browser if it is under both limits; otherwise returns the error
// response to send. Clearing site data resets a browser's count, so the per-browser limit only
// keeps normal use fair; the daily cap is what bounds the total spend.
async function takeRun(req, sql) {
  const browser = req.headers.get('x-browser-id') || '';
  if (!UUID.test(browser)) return error(400, 'Missing browser ID. Reload the page and try again.');
  await sql`DELETE FROM riddle_runs WHERE created_at < now() - interval '1 day'`;
  const [{ mine, all }] = await sql`
    SELECT
      count(*) FILTER (WHERE browser_id = ${browser}::uuid AND created_at > now() - interval '1 hour') AS mine,
      count(*) AS all
    FROM riddle_runs`;
  if (Number(mine) >= PER_BROWSER_PER_HOUR) {
    return error(429, `You can run ${PER_BROWSER_PER_HOUR} riddles per hour (retries count). Try again later.`);
  }
  if (Number(all) >= GLOBAL_PER_DAY) return error(429, 'The lab has reached its daily limit. Try again tomorrow.');
  await sql`INSERT INTO riddle_runs (browser_id) VALUES (${browser}::uuid)`;
  return null;
}

async function create(req, sql) {
  let body;
  try {
    body = await req.json();
  } catch {
    return error(400, 'Send JSON with "riddle" and "solution".');
  }
  const riddle = typeof body?.riddle === 'string' ? body.riddle.trim() : '';
  const solution = typeof body?.solution === 'string' ? body.solution.trim() : '';
  if (!riddle || !solution) return error(400, 'Both a riddle and your solution are required.');
  if (riddle.length > MAX_RIDDLE) return error(400, `Keep the riddle under ${MAX_RIDDLE} characters.`);
  if (solution.length > MAX_SOLUTION) return error(400, `Keep the solution under ${MAX_SOLUTION} characters.`);
  if (body.handle != null && typeof body.handle !== 'string') return error(400, 'The handle must be text.');
  // Collapse whitespace and drop control characters; an empty handle means anonymous.
  const handle = (body.handle || '').replace(/\s+/g, ' ').replace(/\p{C}/gu, '').trim() || null;
  if (handle && handle.length > MAX_HANDLE) return error(400, `Keep the handle under ${MAX_HANDLE} characters.`);

  const limited = await takeRun(req, sql);
  if (limited) return limited;

  const [{ id }] = await sql`
    INSERT INTO riddle_jobs (riddle, solution_1, handle)
    VALUES (${riddle}, ${solution}, ${handle})
    RETURNING id`;

  if (!(await start(id, req, sql))) return error(502, 'The job was saved but could not be started.');
  return Response.json({ id }, { status: 201 });
}

// Background functions answer 202 at once and keep running for up to 15 minutes.
async function start(id, req, sql) {
  try {
    const res = await fetch(new URL('/.netlify/functions/riddle-run-background', req.url), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id }),
    });
    if (!res.ok) throw new Error(`background function answered ${res.status}`);
    return true;
  } catch (e) {
    await sql`UPDATE riddle_jobs SET status = 'error', error = ${`Could not start: ${e.message}`}, finished_at = now(), updated_at = now() WHERE id = ${id}::uuid`;
    return false;
  }
}

async function retry(id, req, sql) {
  if (!UUID.test(id)) return error(404, 'No such job.');
  // Check first so a retry that would be refused does not use up one of the browser's runs.
  const [job] = await sql`SELECT status FROM riddle_jobs WHERE id = ${id}::uuid`;
  if (!job) return error(404, 'No such job.');
  if (job.status !== 'error') return error(409, 'Only a job that ended in an error can be retried.');
  const limited = await takeRun(req, sql);
  if (limited) return limited;
  const rows = await sql`
    UPDATE riddle_jobs SET status = 'queued', error = NULL, finished_at = NULL, updated_at = now()
    WHERE id = ${id}::uuid AND status = 'error'
    RETURNING id`;
  if (!rows.length) return error(409, 'Only a job that ended in an error can be retried.');
  if (!(await start(id, req, sql))) return error(502, 'The job could not be restarted.');
  return Response.json({ id }, { status: 202 });
}

async function read(id, sql) {
  if (!UUID.test(id)) return error(404, 'No such job.');
  await sql`
    UPDATE riddle_jobs
    SET status = 'error', error = 'The job stopped responding and timed out. You can retry it.',
        finished_at = now(), updated_at = now()
    WHERE id = ${id}::uuid AND status IN ('queued', 'running')
      AND updated_at < now() - make_interval(mins => ${STALE_MINUTES})`;
  const rows = await sql`
    SELECT id, created_at, updated_at, finished_at, status, error, handle, riddle, solution_1,
           solver, solution_2, stumper_judges, stumper, quality, narrative, elegance, accuracy,
           total_cost
    FROM riddle_jobs WHERE id = ${id}::uuid`;
  if (!rows.length) return error(404, 'No such job.');
  return Response.json({ ...rows[0], explain_scale: EXPLAIN_SCALE, quality_criteria: QUALITY }, { headers: { 'Cache-Control': 'no-store' } });
}

export default async (req) => {
  const { sql } = getDatabase();
  // Read the id from the path itself; context.params is not filled for every route shape.
  const [id, action] = new URL(req.url).pathname.replace(/^\/api\/riddle-jobs\/?/, '').replace(/\/$/, '').split('/');
  if (req.method === 'POST' && !id) return create(req, sql);
  if (req.method === 'GET' && id && !action) return read(id, sql);
  if (req.method === 'POST' && id && action === 'retry') return retry(id, req, sql);
  return error(405, 'Method not allowed.');
};

export const config = { path: ['/api/riddle-jobs', '/api/riddle-jobs/:id', '/api/riddle-jobs/:id/retry'] };
