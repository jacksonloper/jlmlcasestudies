// POST /api/riddle-jobs        {riddle, solution, handle?} -> {id}; queues a job and starts the background run.
//                              handle is how the submitter appears on the leaderboard; omit it to stay anonymous.
// GET  /api/riddle-jobs/:id    -> the job's row (without the submitter hash), for polling.
import { createHash } from 'node:crypto';
import { getDatabase } from '@netlify/database';
import { EXPLAIN_SCALE, QUALITY } from '../lib/riddle-pipeline.mjs';

const MAX_RIDDLE = 2000;
const MAX_SOLUTION = 3000;
const MAX_HANDLE = 40;
// Every job spends real OpenRouter credit, so cap how many can start.
const PER_CLIENT_PER_HOUR = 5;
const GLOBAL_PER_DAY = 60;

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

const error = (status, message) => Response.json({ error: message }, { status });

async function create(req, context, sql) {
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

  const clientHash = createHash('sha256').update(String(context.ip || 'unknown')).digest('hex');
  const [{ mine, all }] = await sql`
    SELECT
      count(*) FILTER (WHERE client_hash = ${clientHash} AND created_at > now() - interval '1 hour') AS mine,
      count(*) FILTER (WHERE created_at > now() - interval '1 day') AS all
    FROM riddle_jobs`;
  if (Number(mine) >= PER_CLIENT_PER_HOUR) return error(429, `You can submit ${PER_CLIENT_PER_HOUR} riddles per hour. Try again later.`);
  if (Number(all) >= GLOBAL_PER_DAY) return error(429, 'The lab has reached its daily limit. Try again tomorrow.');

  const [{ id }] = await sql`
    INSERT INTO riddle_jobs (riddle, solution_1, handle, client_hash)
    VALUES (${riddle}, ${solution}, ${handle}, ${clientHash})
    RETURNING id`;

  // Background functions answer 202 at once and keep running for up to 15 minutes.
  try {
    const res = await fetch(new URL('/.netlify/functions/riddle-run-background', req.url), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id }),
    });
    if (!res.ok) throw new Error(`background function answered ${res.status}`);
  } catch (e) {
    await sql`UPDATE riddle_jobs SET status = 'error', error = ${`Could not start: ${e.message}`}, finished_at = now() WHERE id = ${id}::uuid`;
    return error(502, 'The job was saved but could not be started.');
  }
  return Response.json({ id }, { status: 201 });
}

async function read(id, sql) {
  if (!UUID.test(id)) return error(404, 'No such job.');
  const rows = await sql`
    SELECT id, created_at, updated_at, finished_at, status, error, handle, riddle, solution_1,
           fact_check, solver, solution_2, stumper_judges, stumper, quality, narrative, elegance, accuracy,
           total_cost
    FROM riddle_jobs WHERE id = ${id}::uuid`;
  if (!rows.length) return error(404, 'No such job.');
  return Response.json({ ...rows[0], explain_scale: EXPLAIN_SCALE, quality_criteria: QUALITY }, { headers: { 'Cache-Control': 'no-store' } });
}

export default async (req, context) => {
  const { sql } = getDatabase();
  // Read the id from the path itself; context.params is not filled for every route shape.
  const id = new URL(req.url).pathname.replace(/^\/api\/riddle-jobs\/?/, '').replace(/\/$/, '');
  if (req.method === 'POST' && !id) return create(req, context, sql);
  if (req.method === 'GET' && id) return read(id, sql);
  return error(405, 'Method not allowed.');
};

export const config = { path: ['/api/riddle-jobs', '/api/riddle-jobs/:id'] };
