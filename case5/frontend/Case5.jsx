import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';

const EX = {
  blurb:
    'Every writer got the same 8 user-style themes: a wedding in Kerala, a submarine crew in 1943, a Tokyo sumo stable, ' +
    'a Scottish whisky distillery, a Kenyan marathon camp, a Paris bakery strike, a Texas oil rig and an Icelandic fishing trawler. ' +
    'Each riddle was fact-checked by Claude Sonnet 5.5 (is the real-world mechanism actually real?) and checked for theme fit ' +
    'by Claude Haiku 5.5.',
  factCheck: true,
  costLogged: true,
};

const shortModel = (m) => m.split('/')[1];
const pct = (x) => `${Math.round(100 * x)}%`;

function summarize(rounds) {
  const groups = {};
  for (const r of rounds) {
    const key = `${r.model}|${r.effort}`;
    groups[key] = groups[key] || { model: r.model, effort: r.effort, rounds: [] };
    groups[key].rounds.push(r);
  }
  return Object.values(groups)
    .map((g) => {
      const done = g.rounds.filter((r) => r.result === 'WIN' || r.result === 'LOSS');
      const wins = done.filter((r) => r.result === 'WIN').length;
      const real = done.filter((r) => r.real).length;
      const winReal = done.filter((r) => r.result === 'WIN' && r.real).length;
      const fit = done.filter((r) => r.fit).length;
      const genLogged = g.rounds.every((r) => typeof r.gen_cost === 'number');
      const genTotal = genLogged ? g.rounds.reduce((a, r) => a + r.gen_cost, 0) : null;
      return {
        ...g,
        key: `${g.model}|${g.effort}`,
        n: done.length,
        failed: g.rounds.length - done.length,
        wins,
        real,
        winReal,
        fit,
        winRate: done.length ? wins / done.length : 0,
        // Expected writer spend to obtain one riddle that wins: all writer spend
        // (including attempts that failed to produce a riddle) divided by wins.
        perWin: genTotal != null && wins ? genTotal / wins : null,
        perWinReal: genTotal != null && winReal ? genTotal / winReal : null,
      };
    })
    .sort((a, b) => b.winRate - a.winRate || b.winReal - a.winReal);
}

function Bar({ value, color = 'bg-blue-500' }) {
  return (
    <div className="w-24 h-2 bg-gray-200 rounded">
      <div className={`h-2 rounded ${color}`} style={{ width: `${Math.round(100 * value)}%` }} />
    </div>
  );
}

function ResultBadge({ result }) {
  const styles = {
    WIN: 'bg-green-100 text-green-800',
    LOSS: 'bg-red-100 text-red-800',
    FAILED: 'bg-gray-100 text-gray-600',
  };
  const label = { WIN: 'Riddler won', LOSS: 'Solver won', FAILED: 'No riddle produced' }[result];
  return <span className={`text-xs font-medium px-2 py-1 rounded ${styles[result]}`}>{label}</span>;
}

function RiddleCard({ r, factCheck }) {
  const [open, setOpen] = useState(false);
  if (r.result === 'FAILED') {
    return (
      <div className="border border-gray-200 rounded-lg p-4">
        <div className="flex justify-between items-center">
          <span className="text-sm text-gray-500">{r.prompt}</span>
          <ResultBadge result={r.result} />
        </div>
        <p className="text-sm text-gray-500 mt-2">The writer did not return a riddle in the required format.</p>
      </div>
    );
  }
  const judgePick = (r.judge || '').trim().slice(0, 1);
  return (
    <div className="border border-gray-200 rounded-lg p-4">
      <div className="flex justify-between items-center mb-3 gap-2">
        <span className="text-sm text-gray-500">{r.prompt}</span>
        <ResultBadge result={r.result} />
      </div>
      <p className="text-gray-900 whitespace-pre-line font-serif text-lg leading-relaxed">{r.riddle}</p>
      <button
        onClick={() => setOpen(!open)}
        className="mt-3 text-sm text-blue-600 hover:text-blue-800"
      >
        {open ? 'Hide answers ▲' : 'Show answers, solver and judge ▼'}
      </button>
      {open && (
        <div className="mt-4 space-y-4 text-sm text-gray-700">
          <div>
            <h4 className="font-medium text-gray-900 mb-1">Riddler&apos;s intended answer</h4>
            <p className="text-gray-600 mb-1">{r.chars}</p>
            <p className="whitespace-pre-line">{r.story}</p>
          </div>
          <div>
            <h4 className="font-medium text-gray-900 mb-1">Solver&apos;s answer (Claude Opus 5.5, low effort)</h4>
            <p className="whitespace-pre-line">{(r.solver || '').replaceAll('**', '')}</p>
          </div>
          <div>
            <h4 className="font-medium text-gray-900 mb-1">
              Blind judge (Claude Opus 5.5): picked {judgePick}; riddler&apos;s story was {r.riddler_pos}
            </h4>
            <p className="whitespace-pre-line">{(r.judge || '').trim().slice(1).trim()}</p>
          </div>
          {factCheck && r.fact && (
            <div>
              <h4 className="font-medium text-gray-900 mb-1">
                Fact check (Claude Sonnet 5.5):{' '}
                <span className={r.real ? 'text-green-700' : 'text-red-700'}>{r.real ? 'REAL' : 'INVENTED'}</span>
              </h4>
              <p className="whitespace-pre-line">{r.fact.trim().split('\n').slice(1).join(' ').trim()}</p>
            </div>
          )}
          <p className="text-xs text-gray-400">
            Total cost of this round: ${(r.cost || 0).toFixed(3)}
            {typeof r.gen_cost === 'number' && ` (writer: $${r.gen_cost.toFixed(3)})`}
          </p>
        </div>
      )}
    </div>
  );
}

export default function Case5() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [selected, setSelected] = useState(null);

  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}case5/data/riddles.json`)
      .then((r) => r.json())
      .then(setData)
      .catch((e) => setError(e.message));
  }, []);

  const rows = useMemo(() => (data ? summarize(data.rounds) : []), [data]);
  const sel = rows.find((r) => r.key === selected);
  const ex = EX;
  const total = useMemo(() => (data ? data.rounds.reduce((a, r) => a + (r.cost || 0), 0) : 0), [data]);

  return (
    <div className="min-h-screen bg-white">
      <div className="max-w-5xl mx-auto px-6 py-16">
        <Link to="/" className="text-blue-600 hover:text-blue-800 mb-8 inline-block">
          ← Back to all case studies
        </Link>

        <h1 className="text-4xl font-light text-gray-900 mb-8">Case Study 5: Riddles That Beat Opus</h1>

        <section className="mb-10">
          <h2 className="text-2xl font-medium text-gray-900 mb-4">The Game</h2>
          <div className="text-gray-700 space-y-3">
            <p>
              A <strong>riddler</strong> model writes a riddle of at most four lines. The answer is a tiny story:
              two characters (one with an occupation) and two sentences explaining what happened. The answer may
              rely on hidden facts that the riddle never states.
            </p>
            <p>
              A <strong>solver</strong> (Claude Opus 5.5 at low reasoning effort with a 1,500-token cap, standing in
              for &quot;10 seconds&quot;) reads the riddle and writes its own story, inventing hidden facts if it
              likes.
            </p>
            <p>
              A <strong>blind judge</strong> (also Claude Opus 5.5) sees the riddle and both stories as A and B, in
              random order, and picks the one that explains the riddle better. The riddler wins if the judge
              prefers the riddler&apos;s story.
            </p>
          </div>
        </section>

        <section className="mb-10">
          <h2 className="text-2xl font-medium text-gray-900 mb-4">The Recipe Every Writer Was Given</h2>
          <div className="text-gray-700 space-y-3">
            <p>
              Before these tournaments, about 35 hand-run rounds found that classic riddles, wordplay and letter
              puzzles almost never beat a strong solver: it decodes clues one at a time. What worked was a{' '}
              <strong>layered</strong> design:
            </p>
            <ul className="list-disc pl-6 space-y-1">
              <li>
                <strong>Hidden fact #1</strong> can be found from the riddle&apos;s odd details. Solvers usually
                find it and stop there.
              </li>
              <li>
                <strong>Hidden fact #2</strong> builds on #1 and rests on a real-world mechanism (a regulation,
                contract clause, technical property or procedure) that explains one specific odd detail: a timing,
                a number, a reaction or a choice.
              </li>
              <li>
                The mechanism may be well known, but one detail should only be explained by a lesser-known
                consequence of it, so a solver who knows the famous part still leaves something unexplained.
              </li>
            </ul>
          </div>
        </section>

        <section className="mb-10">
          <h2 className="text-2xl font-medium text-gray-900 mb-4">Key Findings</h2>
          <ul className="list-disc pl-6 space-y-2 text-gray-700">
            <li>
              <strong>Claude Opus 5.5 at medium effort</strong> was the best value: it won 6 of 8 themed rounds, 5 of
              which also passed the fact check, for an expected $0.054 of writer spend per winning, factually real
              riddle.
            </li>
            <li>
              <strong>Cheap open-weight writers win rarely but are close on cost per win.</strong> GLM-5.3
              ($0.059) and Qwen 3.7 Plus ($0.065) cost little per attempt, so their expected cost per winning, real
              riddle is near Opus&apos;s. Those figures rest on only one or two wins each, and Qwen often drifted off
              the requested theme.
            </li>
            <li>
              <strong>Claude Fable 5.2</strong> won the most rounds, but often by leaning on invented mechanisms,
              which the judge does not penalise. That makes it the most expensive per winning, real riddle among
              the strong writers ($0.16).
            </li>
            <li>
              Claude Sonnet 5.5 wrote riddles too convoluted to beat the solver&apos;s simpler story. Several
              open-weight models often failed to return a riddle in the required format.
            </li>
            <li>
              A fixed user theme roughly halved Gemini 3.8 Flash&apos;s win rate but did not hurt Fable 5.2: strong
              writers can find a fitting mechanism for almost any theme.
            </li>
          </ul>
        </section>

        <section className="mb-6">
          <h2 className="text-2xl font-medium text-gray-900 mb-4">Results</h2>
          {error && <p className="text-red-600">Could not load results: {error}</p>}
          {!data && !error && <p className="text-gray-500">Loading…</p>}
          {data && (
            <>
              <p className="text-gray-600 text-sm mb-4">{ex.blurb}</p>

              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-gray-500 border-b border-gray-200">
                      <th className="py-2 pr-4 font-medium">Riddle writer</th>
                      <th className="py-2 pr-4 font-medium">Effort</th>
                      <th className="py-2 pr-4 font-medium">Riddler wins</th>
                      {ex.factCheck && <th className="py-2 pr-4 font-medium">Mechanism real</th>}
                      {ex.factCheck && <th className="py-2 pr-4 font-medium">Won and real</th>}
                      {ex.factCheck && <th className="py-2 pr-4 font-medium">On theme</th>}
                      {ex.costLogged && <th className="py-2 pr-4 font-medium">$ per winning riddle</th>}
                      {ex.costLogged && ex.factCheck && (
                        <th className="py-2 font-medium">$ per winning, real riddle</th>
                      )}
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((row) => (
                      <tr
                        key={row.key}
                        onClick={() => setSelected(row.key === selected ? null : row.key)}
                        className={`border-b border-gray-100 cursor-pointer hover:bg-gray-50 ${
                          row.key === selected ? 'bg-blue-50' : ''
                        }`}
                      >
                        <td className="py-2 pr-4 text-blue-700">{shortModel(row.model)}</td>
                        <td className="py-2 pr-4 text-gray-600">{row.effort}</td>
                        <td className="py-2 pr-4">
                          <div className="flex items-center gap-2">
                            <Bar value={row.winRate} />
                            <span>
                              {row.wins}/{row.n}
                              {row.failed > 0 && <span className="text-gray-400"> ({row.failed} failed)</span>}
                            </span>
                          </div>
                        </td>
                        {ex.factCheck && <td className="py-2 pr-4">{row.n ? `${row.real}/${row.n}` : '–'}</td>}
                        {ex.factCheck && (
                          <td className="py-2 pr-4">
                            <div className="flex items-center gap-2">
                              <Bar value={row.n ? row.winReal / row.n : 0} color="bg-green-500" />
                              <span>{row.n ? `${row.winReal}/${row.n}` : '–'}</span>
                            </div>
                          </td>
                        )}
                        {ex.factCheck && <td className="py-2 pr-4">{row.n ? `${row.fit}/${row.n}` : '–'}</td>}
                        {ex.costLogged && (
                          <td className="py-2 pr-4">
                            {row.perWin != null ? `$${row.perWin.toFixed(3)}` : <span className="text-gray-400">no wins</span>}
                          </td>
                        )}
                        {ex.costLogged && ex.factCheck && (
                          <td className="py-2">
                            {row.perWinReal != null ? `$${row.perWinReal.toFixed(3)}` : <span className="text-gray-400">none</span>}
                          </td>
                        )}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="text-xs text-gray-500 mt-2">
                Click a row to see that writer&apos;s riddles.{' '}
                {ex.costLogged
                  ? 'Dollar columns are the expected writer spend to get one riddle that wins: total writer spend, including attempts that produced no riddle, divided by the number of wins. They exclude the solver, judge and checks, which are not needed to serve a riddle.'
                  : 'This run logged only the total cost of each round (writer, solver and judge together), not the writer on its own, so the cost per winning riddle cannot be computed for it.'}
              </p>
            </>
          )}
        </section>

        {sel && (
          <section className="mb-12">
            <h3 className="text-xl font-medium text-gray-900 mb-1">
              {shortModel(sel.model)} <span className="text-gray-500 font-normal">({sel.effort} effort)</span>
            </h3>
            <p className="text-sm text-gray-600 mb-4">
              {sel.wins} of {sel.n} riddles beat the solver ({pct(sel.winRate)}).
            </p>
            <div className="space-y-4">
              {sel.rounds.map((r, i) => (
                <RiddleCard key={i} r={r} factCheck={ex.factCheck} />
              ))}
            </div>
          </section>
        )}

        <section className="mb-12">
          <h2 className="text-2xl font-medium text-gray-900 mb-4">Caveats</h2>
          <ul className="list-disc pl-6 space-y-2 text-gray-700">
            <li>
              Samples are small: 8 rounds per writer, one per theme. A 6/8 win rate is consistent with anything from
              about 40% to 95%, and cost-per-win figures resting on one or two wins are very noisy.
            </li>
            <li>
              The judge does not check facts. Some winning riddles rest on invented regulations, which is why a
              separate fact check runs on every riddle and the table reports &quot;won and real&quot;.
            </li>
            <li>
              The fact checker is itself a model at low effort and can be wrong in either direction; expand a riddle
              to read its reasoning.
            </li>
            <li>
              The solver&apos;s &quot;10 seconds&quot; is approximated by low reasoning effort and a token cap, not a
              wall-clock limit.
            </li>
          </ul>
          {data && (
            <p className="text-sm text-gray-500 mt-4">
              {data.rounds.length} rounds in total, all run through OpenRouter, for ${total.toFixed(2)}.
            </p>
          )}
        </section>
      </div>
    </div>
  );
}
