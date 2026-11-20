-- Riddle lab judging, version 2.
--   Stumper: two Opus judges, one per A/B order, each grade both solutions 1-4.
--            stumper = mean over the two judges of (grade for solution I - grade for solution II).
--   Quality: one Opus judge sees only solution I and grades narrative interest, elegance and accuracy 1-4.
-- The per-solution rubric columns from version 1 are dropped; nothing in production used them.
ALTER TABLE riddle_jobs
  DROP COLUMN judge_1,
  DROP COLUMN judge_2,
  -- How the submitter appears on the leaderboard; NULL means anonymous.
  ADD COLUMN handle text,
  -- Array of two judge results, each {order, grade_riddler, grade_solver, reason_riddler, reason_solver, ...}.
  ADD COLUMN stumper_judges jsonb NOT NULL DEFAULT '[]'::jsonb,
  ADD COLUMN stumper numeric,
  -- {scores: {narrative: {grade, reason}, elegance: ..., accuracy: ...}, ...}
  ADD COLUMN quality jsonb,
  ADD COLUMN narrative smallint,
  ADD COLUMN elegance smallint,
  ADD COLUMN accuracy smallint;

CREATE INDEX riddle_jobs_stumper_idx ON riddle_jobs (stumper DESC NULLS LAST) WHERE status = 'done';
