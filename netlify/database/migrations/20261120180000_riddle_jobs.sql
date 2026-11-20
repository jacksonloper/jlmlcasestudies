-- Case Study 5 riddle lab: one row per submitted riddle and everything the pipeline produced for it.
--   Solver:  Opus solves the riddle blind; its answer is solution II.
--   Stumper: two Opus judges, one per A/B order, each grade both solutions 1-4.
--            stumper = mean over the two judges of (grade for solution I - grade for solution II).
--   Quality: one Opus judge sees only solution I and grades narrative interest, elegance and accuracy 1-4.
CREATE TABLE riddle_jobs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  finished_at timestamptz,

  -- queued -> running -> done | error
  status text NOT NULL DEFAULT 'queued',
  error text,
  -- SHA-256 of the submitter's IP, used only for rate limiting.
  client_hash text,

  -- What the user submitted. handle is how they appear on the leaderboard; NULL means anonymous.
  handle text,
  riddle text NOT NULL,
  solution_1 text NOT NULL,

  -- Each model stage stores {model, effort, prompt, raw, cost, ...parsed fields}.
  solver jsonb,
  solution_2 text,
  -- Array of two judge results, each {order, grade_riddler, grade_solver, reason_riddler, reason_solver, ...}.
  stumper_judges jsonb NOT NULL DEFAULT '[]'::jsonb,
  stumper numeric,
  -- {scores: {narrative: {grade, reason}, elegance: ..., accuracy: ...}, ...}
  quality jsonb,
  narrative smallint,
  elegance smallint,
  accuracy smallint,

  total_cost numeric NOT NULL DEFAULT 0
);

CREATE INDEX riddle_jobs_created_at_idx ON riddle_jobs (created_at);
CREATE INDEX riddle_jobs_client_hash_idx ON riddle_jobs (client_hash, created_at);
CREATE INDEX riddle_jobs_stumper_idx ON riddle_jobs (stumper DESC NULLS LAST) WHERE status = 'done';
