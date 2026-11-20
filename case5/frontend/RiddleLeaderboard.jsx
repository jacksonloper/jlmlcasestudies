import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';

const signed = (x) => (x > 0 ? `+${x}` : `${x}`);

function stumperStyle(x) {
  if (x > 0) return 'bg-green-100 text-green-800';
  if (x < 0) return 'bg-red-100 text-red-800';
  return 'bg-gray-100 text-gray-700';
}

function Entry({ e, rank }) {
  return (
    <li className="border border-gray-200 rounded-lg p-4">
      <div className="flex items-start gap-3">
        <span className="text-lg font-medium text-gray-400 w-8 shrink-0 text-right">{rank}</span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span className="font-medium text-gray-900 break-words">{e.handle || 'anonymous'}</span>
            <span className="flex items-center gap-2">
              {e.out_of_budget && (
                <span
                  className="text-xs px-2 py-1 rounded bg-amber-50 text-amber-800"
                  title="Opus used up its thinking budget without answering; solution II is its forced best guess."
                >
                  Opus ran out of budget
                </span>
              )}
              <span className={`text-sm font-medium px-2 py-1 rounded ${stumperStyle(e.stumper)}`} title="Stumper score">
                Stumper {signed(e.stumper)}
              </span>
            </span>
          </div>
          <p className="mt-2 text-gray-800 font-serif whitespace-pre-line line-clamp-3">{e.riddle}</p>
          <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-sm text-gray-500">
            <span>Narrative {e.narrative ?? '–'}/4</span>
            <span>Elegance {e.elegance ?? '–'}/4</span>
            <span>Accuracy {e.accuracy ?? '–'}/4</span>
            <span>{new Date(e.created_at).toLocaleDateString()}</span>
            <Link to={`/case5/lab?job=${e.id}`} className="text-blue-600 hover:text-blue-800">
              Answers and judges’ reasons →
            </Link>
          </div>
        </div>
      </div>
    </li>
  );
}

export default function RiddleLeaderboard() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch('/api/riddle-leaderboard')
      .then(async (r) => {
        const j = await r.json().catch(() => null);
        if (!r.ok || !j) throw new Error('The leaderboard only runs on the Netlify deployment of this site.');
        return j;
      })
      .then(setData)
      .catch((e) => setError(e.message));
  }, []);

  return (
    <div className="min-h-screen bg-white">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 py-16">
        <Link to="/case5/lab" className="text-blue-600 hover:text-blue-800 mb-8 inline-block">
          ← Back to the Riddle Lab
        </Link>
        <h1 className="text-4xl font-light text-gray-900 mb-4">Riddle Lab Leaderboard</h1>
        <div className="text-gray-700 space-y-2 mb-8">
          <p>
            Riddles are ranked by <strong>stumper score</strong>: how much higher two Claude Opus 5.5 judges graded
            the author’s answer than Opus’s own blind attempt, on a 1–4 scale, averaged over the two judges (−3 to
            +3). If Opus runs out of thinking budget, its forced best guess is judged instead. Jobs that failed are
            not listed until they are retried successfully. Ties are
            broken by the narrative, elegance and accuracy grades, then by who submitted first.
          </p>
          <p className="text-sm text-gray-500">
            Scores come from single model runs and are noisy: the same riddle can score differently if submitted
            again.
          </p>
        </div>

        {error && <p className="text-sm text-red-700">{error}</p>}
        {!data && !error && <p className="text-gray-500">Loading…</p>}
        {data && data.entries.length === 0 && (
          <p className="text-gray-600">
            No riddles judged yet.{' '}
            <Link to="/case5/lab" className="text-blue-600 hover:text-blue-800">
              Submit the first one
            </Link>
            .
          </p>
        )}
        {data && data.entries.length > 0 && (
          <>
            <ol className="space-y-3">
              {data.entries.map((e, i) => (
                <Entry key={e.id} e={e} rank={i + 1} />
              ))}
            </ol>
            {data.total > data.entries.length && (
              <p className="text-sm text-gray-500 mt-4">
                Showing the top {data.entries.length} of {data.total} judged riddles.
              </p>
            )}
          </>
        )}
      </div>
    </div>
  );
}
