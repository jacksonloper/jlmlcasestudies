-- Case Study 5 riddle lab: one row per submitted riddle and everything the pipeline produced for it.
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

  -- What the user submitted.
  riddle text NOT NULL,
  solution_1 text NOT NULL,

  -- Each stage stores {model, effort, prompt, raw, cost, ...parsed fields}.
  fact_check jsonb,  -- Sonnet checks the facts that solution I relies on.
  solver jsonb,      -- Opus solves the riddle blind; its answer is solution II.
  solution_2 text,
  judge_1 jsonb,     -- Opus scores solution I against the rubric.
  judge_2 jsonb,     -- Opus scores solution II against the same rubric, separately.

  total_cost numeric NOT NULL DEFAULT 0
);

CREATE INDEX riddle_jobs_created_at_idx ON riddle_jobs (created_at);
CREATE INDEX riddle_jobs_client_hash_idx ON riddle_jobs (client_hash, created_at);
