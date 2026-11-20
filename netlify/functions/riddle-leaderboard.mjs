// GET /api/riddle-leaderboard -> the top finished riddle-lab jobs.
// Jobs whose solver failed are included with the worst stumper score (-3) and flagged.
// Ranked by stumper score, then by the quality grades (narrative + elegance + accuracy),
// then by who submitted first.
import { getDatabase } from '@netlify/database';

const LIMIT = 50;

export default async (req) => {
  if (req.method !== 'GET') return Response.json({ error: 'Method not allowed.' }, { status: 405 });
  const { sql } = getDatabase();
  const rows = await sql`
    SELECT id, created_at, handle, riddle, stumper, narrative, elegance, accuracy,
           (solver->>'error') IS NOT NULL AS solver_failed
    FROM riddle_jobs
    WHERE status IN ('done', 'error') AND stumper IS NOT NULL
    ORDER BY stumper DESC,
             coalesce(narrative, 0) + coalesce(elegance, 0) + coalesce(accuracy, 0) DESC,
             created_at ASC
    LIMIT ${LIMIT}`;
  const [{ total }] = await sql`SELECT count(*)::int AS total FROM riddle_jobs WHERE status IN ('done', 'error') AND stumper IS NOT NULL`;
  return Response.json(
    { total, entries: rows.map((r) => ({ ...r, stumper: Number(r.stumper) })) },
    { headers: { 'Cache-Control': 'no-store' } },
  );
};

export const config = { path: '/api/riddle-leaderboard' };
