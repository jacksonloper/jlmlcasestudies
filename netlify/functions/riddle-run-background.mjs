// Background function (up to 15 minutes): runs the riddle pipeline for one queued job and
// writes each stage's result into its riddle_jobs row as soon as that stage finishes.
// It only acts on a row it can move from 'queued' to 'running', so calling it again, or
// with someone else's id, never re-runs or re-bills a job.
import { getDatabase } from '@netlify/database';
import { factCheck, judge, solve } from '../lib/riddle-pipeline.mjs';

export default async (req) => {
  const { sql } = getDatabase();
  let id;
  try {
    ({ id } = await req.json());
  } catch {
    return;
  }
  if (typeof id !== 'string') return;

  const rows = await sql`
    UPDATE riddle_jobs SET status = 'running', updated_at = now()
    WHERE id = ${id}::uuid AND status = 'queued'
    RETURNING riddle, solution_1`;
  if (!rows.length) return;
  const { riddle, solution_1: solution1 } = rows[0];

  const apiKey = process.env.OPENROUTER_KEY;

  const save = async (column, result) => {
    const json = JSON.stringify(result);
    const cost = result.cost || 0;
    // Column names cannot be parameters, so each stage has its own statement.
    if (column === 'fact_check') {
      await sql`UPDATE riddle_jobs SET fact_check = ${json}::jsonb, total_cost = total_cost + ${cost}, updated_at = now() WHERE id = ${id}::uuid`;
    } else if (column === 'solver') {
      await sql`UPDATE riddle_jobs SET solver = ${json}::jsonb, solution_2 = ${result.raw}, total_cost = total_cost + ${cost}, updated_at = now() WHERE id = ${id}::uuid`;
    } else if (column === 'judge_1') {
      await sql`UPDATE riddle_jobs SET judge_1 = ${json}::jsonb, total_cost = total_cost + ${cost}, updated_at = now() WHERE id = ${id}::uuid`;
    } else if (column === 'judge_2') {
      await sql`UPDATE riddle_jobs SET judge_2 = ${json}::jsonb, total_cost = total_cost + ${cost}, updated_at = now() WHERE id = ${id}::uuid`;
    }
  };

  // A failed stage is recorded in its own column, so the other stages still finish.
  const stage = (column, run) =>
    run().then(
      (result) => save(column, result),
      async (e) => {
        await save(column, { error: String(e?.message || e) });
        throw e;
      },
    );

  try {
    if (!apiKey) throw new Error('OPENROUTER_KEY is not set for this deploy context');
    // Fact check, blind solve and judging solution I are independent; judging solution II
    // waits for the solver.
    const results = await Promise.allSettled([
      stage('fact_check', () => factCheck({ riddle, solution: solution1, apiKey })),
      stage('judge_1', () => judge({ riddle, solution: solution1, apiKey })),
      stage('solver', () => solve({ riddle, apiKey })).then(async () => {
        const [{ solution_2: solution2 }] = await sql`SELECT solution_2 FROM riddle_jobs WHERE id = ${id}::uuid`;
        return stage('judge_2', () => judge({ riddle, solution: solution2, apiKey }));
      }),
    ]);
    const failed = results.filter((r) => r.status === 'rejected').map((r) => String(r.reason?.message || r.reason));
    if (failed.length) throw new Error(failed.join('; '));
    await sql`UPDATE riddle_jobs SET status = 'done', finished_at = now(), updated_at = now() WHERE id = ${id}::uuid`;
  } catch (e) {
    const message = String(e?.message || e).slice(0, 2000);
    await sql`UPDATE riddle_jobs SET status = 'error', error = ${message}, finished_at = now(), updated_at = now() WHERE id = ${id}::uuid`;
  }
};
