// Background function (up to 15 minutes): runs the riddle pipeline for one queued job and
// writes each stage's result into its riddle_jobs row as soon as that stage finishes.
// It only acts on a row it can move from 'queued' to 'running', so calling it again, or
// with someone else's id, never re-runs or re-bills a job.
//
// Each model call is tried once. A job with a failed stage ends as 'error', and the user can
// retry it by hand (POST /api/riddle-jobs/:id/retry), which re-queues it. On a re-run, stages
// that already succeeded are kept and only the failed ones run again.
import { getDatabase } from '@netlify/database';
import { qualityJudge, solve, stumperJudge } from '../lib/riddle-pipeline.mjs';

const ORDERS = ['riddler_first', 'solver_first'];
// Grades run 1-4, so solution I minus solution II is at least 1 - 4.
const WORST_STUMPER = -3;

const ok = (stage) => stage && !stage.error;

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
    UPDATE riddle_jobs SET status = 'running', error = NULL, finished_at = NULL, updated_at = now()
    WHERE id = ${id}::uuid AND status = 'queued'
    RETURNING riddle, solution_1, solver, solution_2, stumper_judges, stumper, quality`;
  if (!rows.length) return;
  const job = rows[0];
  const { riddle, solution_1: riddler } = job;
  const apiKey = process.env.OPENROUTER_KEY;

  const solverOk = ok(job.solver);
  const stumperOk = solverOk && job.stumper != null && job.stumper_judges.length === ORDERS.length && job.stumper_judges.every(ok);
  const qualityOk = ok(job.quality);

  // Clear whatever is about to be redone, so the page never mixes old and new results.
  if (!solverOk) {
    await sql`UPDATE riddle_jobs SET solver = NULL, solution_2 = NULL WHERE id = ${id}::uuid`;
  }
  if (!stumperOk) {
    await sql`UPDATE riddle_jobs SET stumper_judges = '[]'::jsonb, stumper = NULL WHERE id = ${id}::uuid`;
  }
  if (!qualityOk) {
    await sql`UPDATE riddle_jobs SET quality = NULL, narrative = NULL, elegance = NULL, accuracy = NULL WHERE id = ${id}::uuid`;
  }

  // Column names cannot be parameters, so each stage has its own statement.
  const save = {
    solver: (r) =>
      sql`UPDATE riddle_jobs SET solver = ${JSON.stringify(r)}::jsonb, solution_2 = ${r.raw ?? null}, total_cost = total_cost + ${r.cost || 0}, updated_at = now() WHERE id = ${id}::uuid`,
    // Appending in one statement keeps the two parallel judges from overwriting each other.
    stumper: (r) =>
      sql`UPDATE riddle_jobs SET stumper_judges = stumper_judges || ${JSON.stringify([r])}::jsonb, total_cost = total_cost + ${r.cost || 0}, updated_at = now() WHERE id = ${id}::uuid`,
    quality: (r) =>
      sql`UPDATE riddle_jobs SET quality = ${JSON.stringify(r)}::jsonb,
            narrative = ${r.scores?.narrative?.grade ?? null}, elegance = ${r.scores?.elegance?.grade ?? null},
            accuracy = ${r.scores?.accuracy?.grade ?? null},
            total_cost = total_cost + ${r.cost || 0}, updated_at = now() WHERE id = ${id}::uuid`,
  };

  // A failed stage is recorded in its own column, so the other stages still finish.
  const stage = (name, run, extra = {}) =>
    run().then(
      async (result) => {
        await save[name](result);
        return result;
      },
      async (e) => {
        await save[name]({ ...extra, error: String(e?.message || e) });
        throw e;
      },
    );

  const judgeStumper = async (solution2) => {
    const judged = await Promise.all(
      ORDERS.map((order) =>
        stage('stumper', () => stumperJudge({ riddle, riddler, solver: solution2, order, apiKey }), { order }),
      ),
    );
    const score = judged.reduce((a, j) => a + (j.grade_riddler - j.grade_solver), 0) / judged.length;
    await sql`UPDATE riddle_jobs SET stumper = ${score}, updated_at = now() WHERE id = ${id}::uuid`;
  };

  try {
    if (!apiKey) throw new Error('OPENROUTER_KEY is not set for this deploy context');
    // The blind solve and the quality judge are independent. The two stumper judges need
    // solution II, so they start when the solver finishes, in parallel.
    let stumper;
    if (stumperOk) {
      stumper = Promise.resolve();
    } else if (solverOk) {
      stumper = judgeStumper(job.solution_2);
    } else {
      stumper = stage('solver', () => solve({ riddle, apiKey })).then(
        (sol) => judgeStumper(sol.raw),
        // A solver failure is not the riddle's achievement: it goes on the leaderboard with the
        // worst possible stumper score, -3. (A failing stumper judge leaves the score empty.)
        async (e) => {
          await sql`UPDATE riddle_jobs SET stumper = ${WORST_STUMPER}, updated_at = now() WHERE id = ${id}::uuid`;
          throw e;
        },
      );
    }
    const quality = qualityOk ? Promise.resolve() : stage('quality', () => qualityJudge({ riddle, solution: riddler, apiKey }));

    const results = await Promise.allSettled([quality, stumper]);
    const failed = results.filter((r) => r.status === 'rejected').map((r) => String(r.reason?.message || r.reason));
    if (failed.length) throw new Error(failed.join('; '));
    await sql`UPDATE riddle_jobs SET status = 'done', finished_at = now(), updated_at = now() WHERE id = ${id}::uuid`;
  } catch (e) {
    const message = String(e?.message || e).slice(0, 2000);
    await sql`UPDATE riddle_jobs SET status = 'error', error = ${message}, finished_at = now(), updated_at = now() WHERE id = ${id}::uuid`;
  }
};
