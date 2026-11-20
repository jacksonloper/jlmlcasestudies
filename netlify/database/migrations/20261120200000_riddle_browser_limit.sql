-- Rate-limit the riddle lab per browser instead of per IP, and keep no IP-derived data.
-- Dropping the column also drops its index and every stored IP hash.
ALTER TABLE riddle_jobs DROP COLUMN client_hash;

-- One row per submission or retry, keyed by a random ID the browser generates and keeps in
-- localStorage. It has no link to riddle_jobs, so it cannot tie a browser to its riddles, and
-- rows are deleted after a day.
CREATE TABLE riddle_runs (
  browser_id uuid NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX riddle_runs_browser_idx ON riddle_runs (browser_id, created_at);
CREATE INDEX riddle_runs_created_at_idx ON riddle_runs (created_at);
