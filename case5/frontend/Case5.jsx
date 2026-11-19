import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';

const THEMES_BLURB =
  'Every writer got the same 8 user-style themes: a wedding in Kerala, a submarine crew in 1943, a Tokyo sumo stable, ' +
  'a Scottish whisky distillery, a Kenyan marathon camp, a Paris bakery strike, a Texas oil rig and an Icelandic fishing ' +
  'trawler. Each riddle was fact-checked by Claude Sonnet 5.5 (is the real-world mechanism actually real?) and checked for ' +
  'theme fit by Claude Haiku 5.5.';

// Re-judging cost, from the rejudge.py run logs (not stored per round): $1.48 for the main
// tournament plus $0.27 for the Qwen4 Max rounds.
const REJUDGE_COST = 1.75;
// Total cost of the facts_sweep.py run (1, 2 and 3 hidden facts), from its run log.
const FACTS_RUN_COST = 2.99;

const JUDGES = [
  { key: 'opus', name: 'Claude Opus 5.5' },
  { key: 'sol', name: 'GPT-6.2 Sol' },
];

const shortModel = (m) => m.split('/')[1];
// Scores are multiples of 0.25 (two judges x two orders), e.g. 4, 4.5, 4.25.
const fmt = (x) => (Number.isInteger(x) ? `${x}` : x.toFixed(2).replace(/0$/, ''));
const money = (x) => `$${x.toFixed(3)}`;

function summarize(rounds) {
  const groups = {};
  for (const r of rounds) {
    const key = `${r.model}|${r.effort}|${r.variant || ''}`;
    groups[key] = groups[key] || { model: r.model, effort: r.effort, variant: r.variant, key, rounds: [] };
    groups[key].rounds.push(r);
  }
  return Object.values(groups)
    .map((g) => {
      const done = g.rounds.filter((r) => r.result !== 'FAILED');
      const sum = (f) => done.reduce((a, r) => a + f(r), 0);
      const opus = sum((r) => r.score_opus);
      const sol = sum((r) => r.score_sol);
      const avgWin = (opus + sol) / 2;
      const avgWinReal = sum((r) => (r.real ? (r.score_opus + r.score_sol) / 2 : 0));
      // Expected writer spend to obtain one riddle that wins: all writer spend
      // (including attempts that failed to produce a riddle) divided by expected wins.
      const spend = g.rounds.reduce((a, r) => a + (r.gen_cost || 0), 0);
      return {
        ...g,
        n: done.length,
        failed: g.rounds.length - done.length,
        opus,
        sol,
        avgWinReal,
        real: done.filter((r) => r.real).length,
        fit: done.filter((r) => r.fit).length,
        perWin: avgWin ? spend / avgWin : null,
        perWinReal: avgWinReal ? spend / avgWinReal : null,
      };
    })
    .sort((a, b) => b.avgWinReal - a.avgWinReal || b.opus + b.sol - (a.opus + a.sol));
}

function Bar({ value, color = 'bg-blue-500' }) {
  return (
    <div className="w-16 h-2 bg-gray-200 rounded shrink-0">
      <div className={`h-2 rounded ${color}`} style={{ width: `${Math.round(100 * value)}%` }} />
    </div>
  );
}

function Score({ label, score }) {
  const style = score === 1 ? 'bg-green-100 text-green-800' : score === 0 ? 'bg-red-100 text-red-800' : 'bg-yellow-100 text-yellow-800';
  const text = score === 1 ? 'riddler won' : score === 0 ? 'solver won' : 'split';
  return (
    <span className={`text-xs font-medium px-2 py-1 rounded ${style}`}>
      {label}: {text}
    </span>
  );
}

function Verdict({ v, order }) {
  if (!v) return null;
  const pick = (v.judge || '').trim().slice(0, 1);
  const won = v.result === 'WIN';
  return (
    <li>
      <span className="text-gray-500">
        Order {order}: riddler&apos;s story shown as {v.riddler_pos}; judge picked {pick}{' '}
      </span>
      <span className={won ? 'text-green-700' : 'text-red-700'}>({won ? 'riddler' : 'solver'})</span>.{' '}
      {(v.judge || '').trim().slice(1).trim()}
    </li>
  );
}

function RiddleCard({ r }) {
  const [open, setOpen] = useState(false);
  if (r.result === 'FAILED') {
    return (
      <div className="border border-gray-200 rounded-lg p-4">
        <div className="flex justify-between items-center gap-2">
          <span className="text-sm text-gray-500">{r.theme}</span>
          <span className="text-xs font-medium px-2 py-1 rounded bg-gray-100 text-gray-600">No riddle produced</span>
        </div>
        <p className="text-sm text-gray-500 mt-2">The writer did not return a riddle in the required format.</p>
      </div>
    );
  }
  return (
    <div className="border border-gray-200 rounded-lg p-4">
      <div className="flex flex-wrap justify-between items-center mb-3 gap-2">
        <span className="text-sm text-gray-500">{r.theme}</span>
        <span className="flex flex-wrap gap-1">
          {JUDGES.map((j) => (
            <Score key={j.key} label={j.name} score={r[`score_${j.key}`]} />
          ))}
        </span>
      </div>
      <p className="text-gray-900 whitespace-pre-line font-serif text-lg leading-relaxed">{r.riddle}</p>
      <button onClick={() => setOpen(!open)} className="mt-3 text-sm text-blue-600 hover:text-blue-800">
        {open ? 'Hide answers ▲' : 'Show answers, solver and judges ▼'}
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
          {JUDGES.map((j) => (
            <div key={j.key}>
              <h4 className="font-medium text-gray-900 mb-1">
                Blind judge: {j.name} (score {fmt(r[`score_${j.key}`])})
              </h4>
              <ul className="space-y-1">
                {(r.verdicts?.[j.key] || []).map((v, i) => (
                  <Verdict key={i} v={v} order={i + 1} />
                ))}
              </ul>
            </div>
          ))}
          {r.fact && (
            <div>
              <h4 className="font-medium text-gray-900 mb-1">
                Fact check (Claude Sonnet 5.5):{' '}
                <span className={r.real ? 'text-green-700' : 'text-red-700'}>{r.real ? 'REAL' : 'INVENTED'}</span>
              </h4>
              <p className="whitespace-pre-line">{r.fact.trim().split('\n').slice(1).join(' ').trim()}</p>
            </div>
          )}
          {typeof r.gen_cost === 'number' && (
            <p className="text-xs text-gray-400">Writer cost for this riddle: {money(r.gen_cost)}</p>
          )}
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
  const total = useMemo(() => (data ? data.rounds.reduce((a, r) => a + (r.cost || 0), 0) : 0), [data]);

  return (
    <div className="min-h-screen bg-white">
      <div className="max-w-6xl mx-auto px-6 py-16">
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
              Two <strong>blind judges</strong>, Claude Opus 5.5 and GPT-6.2 Sol (both at low effort), each see the
              riddle and both stories as A and B and pick the one that explains the riddle better. Both judges
              prefer whichever story is shown first, so each judges every round twice, once in each order. A
              judge scores a round 1 if it preferred the riddler&apos;s story both times, 0.5 if it split, and 0 if
              it preferred the solver&apos;s both times.
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
              <strong>Best value:</strong> Claude Opus 5.5 (about $0.05–0.06 of writer spend per riddle that wins and
              passes the fact check, at low or medium effort) and GLM-5.3 ($0.052). Gemini 4 Argon is nominally
              cheaper ($0.046), but that rests on half a win.
            </li>
            <li>
              <strong>A third hidden fact helps.</strong> With the recipe extended to three layered hidden facts,
              Claude Opus 5.5 (medium) won 13.5 of 16 rounds by the Opus judge and 11.5 by the Sol judge (0.78
              per round), against about 0.5–0.6 per round with the two-fact recipe. More of its riddles rest on invented mechanisms (10 of 16 passed
              the fact check), so per winning, real riddle it costs about the same as the two-fact recipe. In the
              same run, a one-fact recipe won 0.41 per round and two facts 0.47; the 1- and 2-fact riddles are in
              the raw data file only.
            </li>
            <li>
              <strong>Qwen4 Max improves with effort.</strong> Its riddles that won and passed the fact check rose
              from 3.5 of 8 at low effort to 4.75 of 8 at high, mostly because more of its mechanisms were real (4,
              then 5, then 6 of 8). At high effort it matches Claude Opus 5.5 (medium) for about $0.064 per winning,
              real riddle, the best of the open-weight writers. It is slow: about 3.5 minutes per riddle, and one
              medium-effort attempt used its whole 20,000-token allowance without producing a riddle.
            </li>
            <li>
              <strong>Most real wins:</strong> Gemini 4 Hydrogen (4.5 of 8, every one of them fact-checked as real),
              then Claude Fable 5.2 and Claude Opus 5.5 at medium effort (4.25 each). Hydrogen costs about $0.14 per
              such riddle.
            </li>
            <li>
              <strong>Winning by invention:</strong> Gemini 4 Neon won 7.75 of 8 and GPT-6.2 Sol 5 of 8, but few of
              their winning riddles rest on a real mechanism (3 and 0.25). The judges reward a story that explains
              every detail whether or not its rule exists.
            </li>
            <li>
              <strong>Judges mostly agree, with a hint of self-preference.</strong> Their scores match exactly on 106
              of 149 rounds. Opus scores Claude writers 0.53 per round against Sol&apos;s 0.45; Sol scores GPT-6.2
              writers 0.72 against Opus&apos;s 0.59. The samples are too small to call that a firm effect.
            </li>
            <li>
              <strong>Position bias is large.</strong> In the original single-order judging, Opus picked story A in
              60% of rounds and Sol in 64%, so a writer&apos;s luck in the coin flip mattered more than most
              differences between writers. Judging both orders removes it.
            </li>
            <li>
              Claude Sonnet 5.5 wrote riddles too convoluted to beat the solver&apos;s simpler story. Several
              open-weight models (Kimi K2.6, MiniMax M3, Mistral Large 4) and Gemini 4 Carbon often failed to
              return a riddle at all.
            </li>
          </ul>
        </section>

        <section className="mb-6">
          <h2 className="text-2xl font-medium text-gray-900 mb-4">Results</h2>
          {error && <p className="text-red-600">Could not load results: {error}</p>}
          {!data && !error && <p className="text-gray-500">Loading…</p>}
          {data && (
            <>
              <p className="text-gray-600 text-sm mb-4">{THEMES_BLURB}</p>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-gray-500 border-b border-gray-200 align-bottom">
                      <th className="py-2 pr-3 font-medium">Riddle writer</th>
                      <th className="py-2 pr-3 font-medium">Effort</th>
                      <th className="py-2 pr-3 font-medium">Wins, Opus judge</th>
                      <th className="py-2 pr-3 font-medium">Wins, Sol judge</th>
                      <th className="py-2 pr-3 font-medium">Mechanism real</th>
                      <th className="py-2 pr-3 font-medium">Won and real (both judges)</th>
                      <th className="py-2 pr-3 font-medium">On theme</th>
                      <th className="py-2 pr-3 font-medium">$ per winning riddle</th>
                      <th className="py-2 font-medium">$ per winning, real riddle</th>
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
                        <td className="py-2 pr-3 text-blue-700">{shortModel(row.model)}</td>
                        <td className="py-2 pr-3 text-gray-600">
                          {row.effort}
                          {row.variant && <div className="text-xs text-purple-700 whitespace-nowrap">{row.variant}</div>}
                        </td>
                        {[row.opus, row.sol].map((s, i) => (
                          <td key={i} className="py-2 pr-3">
                            <div className="flex items-center gap-2">
                              <Bar value={row.n ? s / row.n : 0} />
                              <span className="whitespace-nowrap">
                                {fmt(s)}/{row.n}
                              </span>
                            </div>
                          </td>
                        ))}
                        <td className="py-2 pr-3 whitespace-nowrap">
                          {row.real}/{row.n}
                          {row.failed > 0 && <span className="text-gray-400"> ({row.failed} failed)</span>}
                        </td>
                        <td className="py-2 pr-3">
                          <div className="flex items-center gap-2">
                            <Bar value={row.n ? row.avgWinReal / row.n : 0} color="bg-green-500" />
                            <span className="whitespace-nowrap">
                              {fmt(row.avgWinReal)}/{row.n}
                            </span>
                          </div>
                        </td>
                        <td className="py-2 pr-3">
                          {row.fit}/{row.n}
                        </td>
                        <td className="py-2 pr-3">
                          {row.perWin != null ? money(row.perWin) : <span className="text-gray-400">no wins</span>}
                        </td>
                        <td className="py-2">
                          {row.perWinReal != null ? money(row.perWinReal) : <span className="text-gray-400">none</span>}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="text-xs text-gray-500 mt-2">
                Click a row to see that writer&apos;s riddles. Win counts are the sum of each judge&apos;s per-round
                scores. &quot;Won and real&quot; averages the two judges over riddles that passed the fact check. The
                dollar columns are the expected writer spend to get one such riddle: total writer spend, including
                attempts that produced no riddle, divided by the expected wins (averaged over both judges). They
                exclude the solver, judges and checks, which are not needed to serve a riddle. Rows are sorted by
                &quot;won and real&quot;.
              </p>
            </>
          )}
        </section>

        {sel && (
          <section className="mb-12">
            <h3 className="text-xl font-medium text-gray-900 mb-1">
              {shortModel(sel.model)}{' '}
              <span className="text-gray-500 font-normal">
                ({sel.effort} effort{sel.variant ? `, ${sel.variant}` : ''})
              </span>
            </h3>
            <p className="text-sm text-gray-600 mb-4">
              Wins: {fmt(sel.opus)} of {sel.n} by the Opus judge, {fmt(sel.sol)} of {sel.n} by the Sol judge.
            </p>
            <div className="space-y-4">
              {sel.rounds.map((r, i) => (
                <RiddleCard key={i} r={r} />
              ))}
            </div>
          </section>
        )}

        <section className="mb-12">
          <h2 className="text-2xl font-medium text-gray-900 mb-4">Caveats</h2>
          <ul className="list-disc pl-6 space-y-2 text-gray-700">
            <li>
              Samples are small: 8 rounds per writer, one per theme (16, two per theme, for the three-fact row). A score of 6 of 8 is consistent with a true
              rate anywhere from about 40% to 95%, and cost-per-win figures resting on one or two wins are very
              noisy.
            </li>
            <li>
              The judges do not check facts. Some winning riddles rest on invented regulations, which is why a
              separate fact check runs on every riddle and the table reports &quot;won and real&quot;.
            </li>
            <li>
              The fact checker is itself a model at low effort and can be wrong in either direction; expand a riddle
              to read its reasoning.
            </li>
            <li>
              The solver is always Claude Opus 5.5, so the results say which writers beat that particular solver.
            </li>
            <li>
              The solver&apos;s &quot;10 seconds&quot; is approximated by low reasoning effort and a token cap, not a
              wall-clock limit.
            </li>
          </ul>
          {data && (
            <p className="text-sm text-gray-500 mt-4">
              {data.rounds.length} rounds in total, all run through OpenRouter: ${total.toFixed(2)} for writing,
              solving, checking and first-pass judging, plus ${REJUDGE_COST.toFixed(2)} for the extra judging passes
              and ${FACTS_RUN_COST.toFixed(2)} for the hidden-fact run (all three levels).
            </p>
          )}
        </section>
      </div>
    </div>
  );
}
