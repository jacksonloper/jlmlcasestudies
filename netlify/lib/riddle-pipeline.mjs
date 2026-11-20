// Case Study 5 riddle lab: prompts and OpenRouter calls for one submitted riddle.
//
// Pipeline for a (riddle, solution I) pair:
//   1. Claude Sonnet 5.5 fact-checks solution I.
//   2. Claude Opus 5.5 solves the riddle without seeing solution I; its answer is solution II.
//   3. Stumper: two Claude Opus 5.5 judges each see both solutions, in opposite A/B orders to
//      cancel position bias, and grade each one 1-4 on how well it explains the riddle.
//      Stumper = mean over the two judges of (grade for solution I - grade for solution II).
//   4. Quality: one Claude Opus 5.5 judge sees only solution I and grades its narrative interest,
//      elegance and accuracy 1-4.
// Unlike the case study's sweep.py, which asks for a single A/B pick, judges here give grades,
// and no judge is told which solution came from the riddle's author.

const URL = 'https://openrouter.ai/api/v1/chat/completions';
export const OPUS = 'anthropic/claude-opus-5.5';
export const SONNET = 'anthropic/claude-sonnet-5.5';

export const EXPLAIN_SCALE = [
  { grade: 1, text: "Contains fabricated details or doesn't explain the riddle at all." },
  { grade: 2, text: 'Explains part of the riddle but leaves important details unexplained or needs big stretches.' },
  { grade: 3, text: 'Explains nearly every detail, with small gaps or stretches.' },
  { grade: 4, text: 'Perfectly explains every detail of the riddle.' },
];

export const QUALITY = [
  {
    key: 'narrative',
    label: 'Narrative interest',
    help: '1 = flat, nothing worth telling; 4 = a vivid, intriguing story you would want to retell.',
  },
  {
    key: 'elegance',
    label: 'Elegance',
    help: '1 = forced or clunky; 4 = everything clicks into place with nothing wasted.',
  },
  {
    key: 'accuracy',
    label: 'Accuracy',
    help: '1 = rests on false or invented real-world facts; 4 = every claim about the real world is correct.',
  },
];

export const FACT_PROMPT = `Below is a riddle and a proposed solution. The solution may rely on real-world facts or mechanisms (laws, regulations, rules, technical properties, procedures, history).

RIDDLE:
{riddle}

PROPOSED SOLUTION:
{solution}

Check only whether the real-world facts the solution relies on are true. Do not judge how well it solves the riddle. Invented story details about fictional people are fine; claims about how the real world works are what you are checking.`;

export const SOLVE_PROMPT = `Solve this riddle. The intended answer may rely on hidden facts not stated in the riddle that explain every odd detail; you may supply such facts too. A judge will grade your answer on how completely and naturally it explains the riddle. Reply with your answer and a short explanation (at most four sentences).

{riddle}`;

export const STUMPER_PROMPT = `Below is a riddle and two proposed answers, A and B. An answer may add hidden facts that the riddle does not state; that is allowed. Grade EACH answer on this scale:

${EXPLAIN_SCALE.map((s) => `${s.grade} = ${s.text}`).join('\n')}

"Fabricated details" means claims about how the real world works (history, laws, technology, procedures) that are false or made up, or details that contradict the riddle. Inventing names, people or private story details is NOT fabrication: a riddle's answer is a story, and its characters may be fictional. Grade each answer on its own merits; both may get the same grade. Give a one- or two-sentence reason for each grade.

RIDDLE:
{riddle}

ANSWER A:
{a}

ANSWER B:
{b}`;

export const QUALITY_PROMPT = `Below is a riddle and its intended answer. Grade the answer from 1 to 4 on each criterion:

${QUALITY.map((c) => `- ${c.key} (${c.label}): ${c.help}`).join('\n')}

Give a one-sentence reason for each grade.

RIDDLE:
{riddle}

ANSWER:
{solution}`;

const FACT_SCHEMA = {
  type: 'object',
  properties: {
    verdict: { type: 'string', enum: ['ACCURATE', 'MINOR_ISSUES', 'WRONG', 'NO_FACTUAL_CLAIMS'] },
    issues: { type: 'array', items: { type: 'string' } },
    summary: { type: 'string' },
  },
  required: ['verdict', 'issues', 'summary'],
  additionalProperties: false,
};

const GRADE = { type: 'integer', enum: [1, 2, 3, 4] };

const STUMPER_SCHEMA = {
  type: 'object',
  properties: {
    a_reason: { type: 'string' },
    a_grade: GRADE,
    b_reason: { type: 'string' },
    b_grade: GRADE,
  },
  required: ['a_reason', 'a_grade', 'b_reason', 'b_grade'],
  additionalProperties: false,
};

const QUALITY_SCHEMA = {
  type: 'object',
  properties: Object.fromEntries(
    QUALITY.map((c) => [
      c.key,
      {
        type: 'object',
        properties: { reason: { type: 'string' }, grade: GRADE },
        required: ['reason', 'grade'],
        additionalProperties: false,
      },
    ]),
  ),
  required: QUALITY.map((c) => c.key),
  additionalProperties: false,
};

const fill = (template, vars) => template.replace(/\{(\w+)\}/g, (m, k) => (k in vars ? vars[k] : m));

async function chat({ model, prompt, maxTokens, effort, schema, apiKey }) {
  const body = {
    model,
    messages: [{ role: 'user', content: prompt }],
    max_tokens: maxTokens,
    reasoning: { effort },
  };
  if (schema) {
    body.response_format = { type: 'json_schema', json_schema: { name: 'result', strict: true, schema } };
  }
  let lastError;
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const res = await fetch(URL, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
          'X-Title': 'jlmlcasestudies riddle lab',
        },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(240_000),
      });
      const text = await res.text();
      if (!res.ok) throw new Error(`OpenRouter ${res.status}: ${text.slice(0, 300)}`);
      const j = JSON.parse(text);
      const raw = (j.choices?.[0]?.message?.content || '').trim();
      if (!raw) throw new Error('OpenRouter returned an empty answer');
      return { model, effort, prompt, raw, cost: Number(j.usage?.cost) || 0 };
    } catch (e) {
      lastError = e;
      await new Promise((r) => setTimeout(r, 3000 * (attempt + 1)));
    }
  }
  throw lastError;
}

// Structured outputs should give clean JSON; tolerate a fenced or prefixed answer anyway.
function parseJson(raw) {
  const start = raw.indexOf('{');
  const end = raw.lastIndexOf('}');
  if (start < 0 || end < start) throw new Error('Model answer was not JSON');
  return JSON.parse(raw.slice(start, end + 1));
}

const grade = (x) => {
  const g = Number(x);
  if (![1, 2, 3, 4].includes(g)) throw new Error(`Judge gave an invalid grade: ${x}`);
  return g;
};

export async function factCheck({ riddle, solution, apiKey }) {
  const r = await chat({
    model: SONNET,
    prompt: fill(FACT_PROMPT, { riddle, solution }),
    maxTokens: 4000,
    effort: 'low',
    schema: FACT_SCHEMA,
    apiKey,
  });
  return { ...r, ...parseJson(r.raw) };
}

export async function solve({ riddle, apiKey }) {
  return chat({ model: OPUS, prompt: fill(SOLVE_PROMPT, { riddle }), maxTokens: 4000, effort: 'low', apiKey });
}

// order 'riddler_first' shows solution I as A; 'solver_first' shows solution II as A.
export async function stumperJudge({ riddle, riddler, solver, order, apiKey }) {
  const riddlerFirst = order === 'riddler_first';
  const [a, b] = riddlerFirst ? [riddler, solver] : [solver, riddler];
  const r = await chat({
    model: OPUS,
    prompt: fill(STUMPER_PROMPT, { riddle, a, b }),
    maxTokens: 4000,
    effort: 'low',
    schema: STUMPER_SCHEMA,
    apiKey,
  });
  const p = parseJson(r.raw);
  const A = { grade: grade(p.a_grade), reason: p.a_reason };
  const B = { grade: grade(p.b_grade), reason: p.b_reason };
  const [mine, theirs] = riddlerFirst ? [A, B] : [B, A];
  return {
    ...r,
    order,
    grade_riddler: mine.grade,
    reason_riddler: mine.reason,
    grade_solver: theirs.grade,
    reason_solver: theirs.reason,
  };
}

export async function qualityJudge({ riddle, solution, apiKey }) {
  const r = await chat({
    model: OPUS,
    prompt: fill(QUALITY_PROMPT, { riddle, solution }),
    maxTokens: 4000,
    effort: 'low',
    schema: QUALITY_SCHEMA,
    apiKey,
  });
  const p = parseJson(r.raw);
  const scores = Object.fromEntries(QUALITY.map((c) => [c.key, { grade: grade(p[c.key]?.grade), reason: p[c.key]?.reason }]));
  return { ...r, scores };
}
