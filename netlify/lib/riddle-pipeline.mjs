// Case Study 5 riddle lab: prompts and OpenRouter calls for one submitted riddle.
//
// Pipeline for a (riddle, solution I) pair:
//   1. Claude Sonnet 5.5 fact-checks solution I.
//   2. Claude Opus 5.5 solves the riddle without seeing solution I; its answer is solution II.
//   3. Claude Opus 5.5 scores solution I against RUBRIC.
//   4. Claude Opus 5.5 scores solution II against the same RUBRIC, in a separate call.
// The judge is never told whether a solution came from the riddle's author or from a model.

const URL = 'https://openrouter.ai/api/v1/chat/completions';
export const OPUS = 'anthropic/claude-opus-5.5';
export const SONNET = 'anthropic/claude-sonnet-5.5';

// Unlike the case study's sweep.py, which asks a judge to pick the better of two stories,
// the lab scores each solution on its own. Every criterion is scored so that 5 is best.
export const RUBRIC = [
  {
    key: 'satisfying',
    label: 'Satisfying',
    help: 'Overall, how satisfying it is as the answer to this riddle: 5 = a clean "aha", 1 = leaves you cold.',
  },
  {
    key: 'not_convoluted',
    label: 'Not convoluted',
    help: '5 = simple and direct; 1 = excessively convoluted, piling on contrivances or hidden facts.',
  },
  {
    key: 'tight',
    label: 'Tight explanation',
    help: '5 = explains every detail of the riddle with nothing left over or unexplained; 1 = loose, leaves details unexplained or adds irrelevant ones.',
  },
];

export const FACT_PROMPT = `Below is a riddle and a proposed solution. The solution may rely on real-world facts or mechanisms (laws, regulations, rules, technical properties, procedures, history).

RIDDLE:
{riddle}

PROPOSED SOLUTION:
{solution}

Check only whether the real-world facts the solution relies on are true. Do not judge how well it solves the riddle. Invented story details about fictional people are fine; claims about how the real world works are what you are checking.`;

export const SOLVE_PROMPT = `Solve this riddle. The intended answer may rely on hidden facts not stated in the riddle that explain every odd detail; you may supply such facts too. A judge will score your answer on how completely and naturally it explains the riddle. Reply with your answer and a short explanation (at most four sentences).

{riddle}`;

export const JUDGE_PROMPT = `Below is a riddle and one proposed solution. Score the solution from 1 (poor) to 5 (excellent) on each criterion:

${RUBRIC.map((c) => `- ${c.key}: ${c.help}`).join('\n')}

Give a one-sentence reason for each score, then a one-sentence overall verdict.

RIDDLE:
{riddle}

PROPOSED SOLUTION:
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

const JUDGE_SCHEMA = {
  type: 'object',
  properties: {
    scores: {
      type: 'object',
      properties: Object.fromEntries(
        RUBRIC.map((c) => [
          c.key,
          {
            type: 'object',
            properties: { score: { type: 'integer', minimum: 1, maximum: 5 }, reason: { type: 'string' } },
            required: ['score', 'reason'],
            additionalProperties: false,
          },
        ]),
      ),
      required: RUBRIC.map((c) => c.key),
      additionalProperties: false,
    },
    verdict: { type: 'string' },
  },
  required: ['scores', 'verdict'],
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

export async function judge({ riddle, solution, apiKey }) {
  const r = await chat({
    model: OPUS,
    prompt: fill(JUDGE_PROMPT, { riddle, solution }),
    maxTokens: 4000,
    effort: 'low',
    schema: JUDGE_SCHEMA,
    apiKey,
  });
  const parsed = parseJson(r.raw);
  const total = RUBRIC.reduce((a, c) => a + (Number(parsed.scores?.[c.key]?.score) || 0), 0);
  return { ...r, ...parsed, total, max_total: 5 * RUBRIC.length };
}
