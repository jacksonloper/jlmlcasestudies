import { useCallback, useEffect, useRef, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';

const API = '/api/riddle-jobs';
const POLL_MS = 4000;
const RECENT_KEY = 'case5-lab-recent';
const MAX_RIDDLE = 2000;
const MAX_SOLUTION = 3000;
const MAX_HANDLE = 40;
const HANDLE_KEY = 'case5-lab-handle';

const money = (x) => `$${Number(x || 0).toFixed(3)}`;

// localStorage can be missing or throw (private windows, blocked storage); the page works without it.
function loadRecent() {
  try {
    return JSON.parse(localStorage.getItem(RECENT_KEY) || '[]');
  } catch {
    return [];
  }
}
function saveRecent(list) {
  try {
    localStorage.setItem(RECENT_KEY, JSON.stringify(list.slice(0, 10)));
  } catch {
    // ignore
  }
}

function notify(title, body) {
  try {
    if ('Notification' in window && Notification.permission === 'granted') new Notification(title, { body });
  } catch {
    // ignore
  }
}

function StageBadge({ result, running }) {
  if (result?.error) return <span className="text-xs font-medium px-2 py-1 rounded bg-red-100 text-red-800">failed</span>;
  if (result) return <span className="text-xs font-medium px-2 py-1 rounded bg-green-100 text-green-800">done</span>;
  if (running) return <span className="text-xs font-medium px-2 py-1 rounded bg-yellow-100 text-yellow-800 animate-pulse">working…</span>;
  return <span className="text-xs font-medium px-2 py-1 rounded bg-gray-100 text-gray-600">waiting</span>;
}

function Stage({ title, model, result, running, children }) {
  return (
    <div className="border border-gray-200 rounded-lg p-4">
      <div className="flex flex-wrap justify-between items-center gap-2 mb-2">
        <h3 className="font-medium text-gray-900">{title}</h3>
        <span className="flex items-center gap-2">
          <span className="text-xs text-gray-500">{model}</span>
          <StageBadge result={result} running={running} />
        </span>
      </div>
      {result?.error ? <p className="text-sm text-red-700">{result.error}</p> : result ? children : null}
      {result?.cost ? <p className="text-xs text-gray-400 mt-2">Cost: {money(result.cost)}</p> : null}
    </div>
  );
}

const signed = (x) => (x > 0 ? `+${x}` : `${x}`);

function GradeRow({ label, grade, reason }) {
  return (
    <tr className="border-t border-gray-100 align-top">
      <td className="py-1 pr-3 font-medium whitespace-nowrap">{label}</td>
      <td className="py-1 pr-3 whitespace-nowrap">{grade ?? '–'} / 4</td>
      <td className="py-1 text-gray-600">{reason}</td>
    </tr>
  );
}

function StumperJudge({ j }) {
  const title = j.order === 'riddler_first' ? 'Judge 1 (saw solution I as A)' : 'Judge 2 (saw solution II as A)';
  if (j.error) {
    return (
      <div>
        <h4 className="font-medium text-gray-900 mb-1">{title}</h4>
        <p className="text-sm text-red-700">{j.error}</p>
      </div>
    );
  }
  return (
    <div>
      <h4 className="font-medium text-gray-900 mb-1">{title}</h4>
      <table className="w-full text-left text-sm text-gray-700">
        <tbody>
          <GradeRow label="Solution I" grade={j.grade_riddler} reason={j.reason_riddler} />
          <GradeRow label="Solution II" grade={j.grade_solver} reason={j.reason_solver} />
        </tbody>
      </table>
      <p className="text-xs text-gray-400 mt-1">Cost: {money(j.cost)}</p>
    </div>
  );
}

function Stumper({ job, running }) {
  const judges = [...(job.stumper_judges || [])].sort((a, b) => (a.order < b.order ? -1 : 1));
  const done = job.stumper != null;
  const failed = judges.some((j) => j.error) || job.solver?.error;
  return (
    <div className="border border-gray-200 rounded-lg p-4">
      <div className="flex flex-wrap justify-between items-center gap-2 mb-2">
        <h3 className="font-medium text-gray-900">2. Stumper: both solutions graded 1–4, two judges</h3>
        <span className="flex items-center gap-2">
          <span className="text-xs text-gray-500">Claude Opus 5.5, low effort, ×2</span>
          <StageBadge result={done ? {} : failed ? { error: true } : null} running={running} />
        </span>
      </div>
      <p className="text-xs text-gray-500 mb-3">
        {(job.explain_scale || []).map((s) => `${s.grade} = ${s.text}`).join(' ')} The judges see the two solutions
        in opposite orders to cancel position bias, and are not told which is yours.
      </p>
      {done && (
        <p className="text-lg font-medium text-gray-900 mb-3">
          Stumper score: {signed(Number(job.stumper))}{' '}
          <span className="text-sm font-normal text-gray-600">
            (your grade minus Opus’s, averaged over the two judges; range −3 to +3)
          </span>
        </p>
      )}
      <div className="grid md:grid-cols-2 gap-4">
        {judges.map((j) => (
          <StumperJudge key={j.order} j={j} />
        ))}
      </div>
    </div>
  );
}

function Quality({ job, running }) {
  const q = job.quality;
  return (
    <Stage title="3. Quality of solution I: narrative, elegance, accuracy" model="Claude Opus 5.5, low effort" result={q} running={running}>
      <table className="w-full text-left text-sm text-gray-700">
        <tbody>
          {(job.quality_criteria || []).map((c) => (
            <GradeRow key={c.key} label={c.label} grade={q?.scores?.[c.key]?.grade} reason={q?.scores?.[c.key]?.reason} />
          ))}
        </tbody>
      </table>
      <p className="text-xs text-gray-500 mt-2">
        This judge sees only the riddle and your solution.{' '}
        {(job.quality_criteria || []).map((c) => `${c.label}: ${c.help}`).join(' ')}
      </p>
    </Stage>
  );
}

function Job({ job }) {
  const running = job.status === 'running' || job.status === 'queued';
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3 text-sm text-gray-500">
        <span>
          Status: <strong className="text-gray-900">{job.status}</strong>
        </span>
        <span>
          By <strong className="text-gray-900">{job.handle || 'anonymous'}</strong>
        </span>
        <span>Submitted {new Date(job.created_at).toLocaleString()}</span>
        <span>Spent so far: {money(job.total_cost)}</span>
      </div>
      {job.error && <p className="text-sm text-red-700">{job.error}</p>}

      <div className="border border-gray-200 rounded-lg p-4 bg-gray-50">
        <h3 className="font-medium text-gray-900 mb-2">Riddle</h3>
        <p className="text-gray-900 whitespace-pre-line font-serif text-lg leading-relaxed">{job.riddle}</p>
        <h3 className="font-medium text-gray-900 mt-4 mb-1">Solution I (yours)</h3>
        <p className="text-sm text-gray-700 whitespace-pre-line">{job.solution_1}</p>
      </div>

      <Stage title="1. Blind solve → solution II" model="Claude Opus 5.5, low effort" result={job.solver} running={running}>
        <p className="text-sm text-gray-700 whitespace-pre-line">{(job.solution_2 || '').replaceAll('**', '')}</p>
      </Stage>
      <Stumper job={job} running={running} />
      <Quality job={job} running={running} />
    </div>
  );
}

export default function RiddleLab() {
  const [params, setParams] = useSearchParams();
  const jobId = params.get('job');
  const [riddle, setRiddle] = useState('');
  const [solution, setSolution] = useState('');
  const [handle, setHandle] = useState(() => {
    try {
      return localStorage.getItem(HANDLE_KEY) || '';
    } catch {
      return '';
    }
  });
  const [submitting, setSubmitting] = useState(false);
  const [formError, setFormError] = useState(null);
  const [job, setJob] = useState(null);
  const [loadError, setLoadError] = useState(null);
  const [recent, setRecent] = useState(loadRecent);
  const [notifyState, setNotifyState] = useState(() => ('Notification' in window ? Notification.permission : 'unsupported'));
  const lastStatus = useRef(null);

  const poll = useCallback(async (id) => {
    const res = await fetch(`${API}/${id}`);
    if (!res.ok) throw new Error(res.status === 404 ? 'No job with that id.' : `Server answered ${res.status}`);
    return res.json();
  }, []);

  useEffect(() => {
    if (!jobId) {
      setJob(null);
      return undefined;
    }
    let stopped = false;
    let timer;
    lastStatus.current = null;
    setLoadError(null);
    const tick = async () => {
      try {
        const j = await poll(jobId);
        if (stopped) return;
        setJob(j);
        const finished = j.status === 'done' || j.status === 'error';
        if (finished && lastStatus.current && lastStatus.current !== j.status) {
          notify(j.status === 'done' ? 'Your riddle is judged' : 'Your riddle job failed', j.riddle.slice(0, 80));
          document.title = j.status === 'done' ? '✓ Riddle judged' : '✗ Riddle failed';
        }
        lastStatus.current = j.status;
        setRecent((list) => {
          const next = list.map((x) => (x.id === j.id ? { ...x, status: j.status } : x));
          saveRecent(next);
          return next;
        });
        if (!finished) timer = setTimeout(tick, POLL_MS);
      } catch (e) {
        if (!stopped) {
          setLoadError(e.message);
          timer = setTimeout(tick, POLL_MS * 2);
        }
      }
    };
    tick();
    return () => {
      stopped = true;
      clearTimeout(timer);
    };
  }, [jobId, poll]);

  const askNotify = async () => {
    try {
      setNotifyState(await Notification.requestPermission());
    } catch {
      // ignore
    }
  };

  const submit = async (e) => {
    e.preventDefault();
    setSubmitting(true);
    setFormError(null);
    try {
      const res = await fetch(API, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ riddle, solution, handle: handle.trim() || undefined }),
      });
      const j = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(j.error || `Server answered ${res.status}`);
      try {
        localStorage.setItem(HANDLE_KEY, handle.trim());
      } catch {
        // ignore
      }
      const entry = { id: j.id, riddle: riddle.slice(0, 80), status: 'queued', at: Date.now() };
      const next = [entry, ...recent.filter((x) => x.id !== j.id)];
      setRecent(next);
      saveRecent(next);
      setRiddle('');
      setSolution('');
      setParams({ job: j.id });
    } catch (err) {
      setFormError(
        err instanceof TypeError || /Unexpected token/.test(err.message)
          ? 'Could not reach the lab’s server. It only runs on the Netlify deployment of this site.'
          : err.message,
      );
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-white">
      <div className="max-w-4xl mx-auto px-6 py-16">
        <Link to="/case5" className="text-blue-600 hover:text-blue-800 mb-8 inline-block">
          ← Back to Case Study 5
        </Link>
        <div className="flex flex-wrap items-baseline justify-between gap-2 mb-4">
          <h1 className="text-4xl font-light text-gray-900">Riddle Lab: Test Your Own Riddle</h1>
          <Link to="/case5/leaderboard" className="text-blue-600 hover:text-blue-800 font-medium">
            Leaderboard →
          </Link>
        </div>
        <div className="text-gray-700 space-y-3 mb-8">
          <p>
            Submit a riddle and your intended answer (solution I). Claude Opus 5.5 solves the riddle without seeing
            your answer (solution II). Two Opus judges then grade both
            solutions from 1 to 4 on how well they explain the riddle, seeing them in opposite orders and without
            being told which is yours. Your <strong>stumper score</strong> is how much higher your solution was
            graded, averaged over the two judges. A third Opus judge reads only your solution and grades its
            narrative interest, elegance and accuracy.
          </p>
          <p className="text-sm text-gray-500">
            A job usually takes one to three minutes. You can leave this page: the link to your job keeps working,
            and recent jobs are listed below on this device.
          </p>
        </div>

        {jobId ? (
          <section className="mb-10">
            <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
              <h2 className="text-2xl font-medium text-gray-900">Your job</h2>
              <button onClick={() => setParams({})} className="text-sm text-blue-600 hover:text-blue-800">
                Submit another riddle
              </button>
            </div>
            {notifyState === 'default' && job && (job.status === 'queued' || job.status === 'running') && (
              <p className="text-sm mb-4">
                <button onClick={askNotify} className="text-blue-600 hover:text-blue-800">
                  Notify me when it’s done
                </button>{' '}
                <span className="text-gray-500">(browser notification, while this tab is open)</span>
              </p>
            )}
            {loadError && !job && <p className="text-sm text-red-700">{loadError}</p>}
            {job ? <Job job={job} /> : !loadError && <p className="text-gray-500">Loading…</p>}
          </section>
        ) : (
          <form onSubmit={submit} className="space-y-4 mb-10">
            <label className="block">
              <span className="font-medium text-gray-900">Riddle</span>
              <textarea
                value={riddle}
                onChange={(e) => setRiddle(e.target.value)}
                maxLength={MAX_RIDDLE}
                rows={5}
                required
                className="mt-1 w-full border border-gray-300 rounded-lg p-3 font-serif"
              />
            </label>
            <label className="block">
              <span className="font-medium text-gray-900">Your solution</span>
              <textarea
                value={solution}
                onChange={(e) => setSolution(e.target.value)}
                maxLength={MAX_SOLUTION}
                rows={5}
                required
                className="mt-1 w-full border border-gray-300 rounded-lg p-3"
              />
            </label>
            <label className="block">
              <span className="font-medium text-gray-900">Leaderboard handle</span>{' '}
              <span className="text-sm text-gray-500">(optional; leave blank to stay anonymous)</span>
              <input
                type="text"
                value={handle}
                onChange={(e) => setHandle(e.target.value)}
                maxLength={MAX_HANDLE}
                className="mt-1 w-full sm:w-80 block border border-gray-300 rounded-lg p-2"
              />
            </label>
            {formError && <p className="text-sm text-red-700">{formError}</p>}
            <button
              type="submit"
              disabled={submitting}
              className="px-4 py-2 rounded-lg bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50"
            >
              {submitting ? 'Submitting…' : 'Submit riddle'}
            </button>
            <p className="text-xs text-gray-500">
              Submissions are stored, with every model answer, in this site’s database. Limit: 5 per hour.
            </p>
          </form>
        )}

        {recent.length > 0 && (
          <section>
            <h2 className="text-lg font-medium text-gray-900 mb-2">Your recent jobs</h2>
            <ul className="space-y-1 text-sm">
              {recent.map((x) => (
                <li key={x.id}>
                  <Link to={`?job=${x.id}`} className="text-blue-600 hover:text-blue-800">
                    {x.riddle || x.id}
                  </Link>{' '}
                  <span className="text-gray-500">({x.status})</span>
                </li>
              ))}
            </ul>
          </section>
        )}
      </div>
    </div>
  );
}
