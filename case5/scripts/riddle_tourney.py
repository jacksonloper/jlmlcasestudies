import json, random, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "https://openrouter.ai/api/v1/chat/completions"
OPUS = "anthropic/claude-opus-5.5"

GEN_PROMPT = """Design ONE riddle round.

RULES: answer = two characters (one with an occupation) + a two-sentence story. Riddle max 4 lines. Hidden facts may appear only in your answer. A strong AI solver may also invent hidden facts. A blind judge picks whichever story explains the riddle BETTER (every odd detail, fewest stretches). You win if yours is preferred.

Proven recipe: LAYERED.
- Hidden fact #1 is findable from the riddle's odd details (solvers usually find it).
- Hidden fact #2 builds on #1 and rests on a REAL-WORLD mechanism (a regulation, contract clause, technical property, or procedure), stated explicitly in your story. It must explain one specific odd detail in the riddle (a timing, number, reaction, or choice).
- The mechanism may be known, but the riddle must contain one odd detail that only a LESSER-KNOWN consequence of it explains, so a solver who knows the famous part still leaves that detail unexplained.
- Your story must contradict no line of the riddle. Surprising, not sentimental. Never name the occupation in the riddle.

Setting seed: {seed}

Reply in exactly this format and nothing else:
RIDDLE: <text>
CHARACTERS: <char 1>; <char 2 (occupation)>
STORY: <two sentences with both hidden facts>"""

SOLVE_PROMPT = """You have 10 seconds. This riddle's answer is a tiny story: two characters (one with an occupation) and a two-sentence story. The intended answer may include hidden facts not stated in the riddle that explain every odd detail; you may invent such facts too. A judge will pick whichever story explains the riddle better. Reply with: Character 1, Character 2 (with occupation), and a two-sentence story.

{riddle}"""

JUDGE_PROMPT = """Below is a riddle and two candidate stories that each claim to explain it. Stories may include hidden facts not stated in the riddle. Pick the story that explains the riddle BETTER: accounts for every odd detail most completely and naturally, with the fewest stretches. Reply with only "A" or "B" on the first line, then a one-sentence reason.

RIDDLE:
{riddle}

STORY A: {a}

STORY B: {b}"""

SEEDS = ["a ferry or harbour", "a dental surgery", "a wedding venue", "a cemetery or crematorium"]


def chat(model, prompt, max_tokens, effort=None, retries=3):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}
    if effort:
        body["reasoning"] = {"effort": effort}
    data = json.dumps(body).encode()
    for i in range(retries):
        try:
            req = urllib.request.Request(URL, data=data, headers={
                "Authorization": "Bearer proxy-injected", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=240) as r:
                j = json.load(r)
            msg = j["choices"][0]["message"].get("content") or ""
            cost = (j.get("usage") or {}).get("cost", 0) or 0
            return msg.strip(), cost
        except Exception as e:
            err = e
            time.sleep(3 * (i + 1))
    raise err


def parse(gen):
    gen = gen.replace("**", "")
    m = re.search(r"RIDDLE:\s*(.*?)\s*CHARACTERS:\s*(.*?)\s*STORY:\s*(.*)", gen, re.S)
    if not m:
        return None
    return {"riddle": m.group(1).strip(), "chars": m.group(2).strip(), "story": m.group(3).strip()}


def run_round(model, seed, rng_seed):
    rec = {"model": model, "seed": seed, "cost": 0.0}
    try:
        gen, c = chat(model, GEN_PROMPT.format(seed=seed), 20000, effort="medium")
        rec["cost"] += c
        p = parse(gen)
        if not p:
            rec["result"] = "PARSE_FAIL"; rec["raw"] = gen[:500]; return rec
        rec.update(p)
        sol, c = chat(OPUS, SOLVE_PROMPT.format(riddle=p["riddle"]), 1500, effort="low")
        rec["cost"] += c; rec["solver"] = sol
        riddler_story = f"Characters: {p['chars']}. {p['story']}"
        flip = random.Random(rng_seed).random() < 0.5
        a, b = (riddler_story, sol) if flip else (sol, riddler_story)
        rec["riddler_pos"] = "A" if flip else "B"
        jd, c = chat(OPUS, JUDGE_PROMPT.format(riddle=p["riddle"], a=a, b=b), 1500, effort="low")
        rec["cost"] += c; rec["judge"] = jd
        pick = jd.strip()[:1].upper()
        rec["result"] = "WIN" if pick == rec["riddler_pos"] else ("LOSS" if pick in "AB" else "JUDGE_FAIL")
    except Exception as e:
        rec["result"] = f"ERROR {e}"
    return rec


if __name__ == "__main__":
    models = sys.argv[1].split(",")
    out = sys.argv[2]
    if len(sys.argv) > 3:
        SEEDS = sys.argv[3].split("|")
    jobs = [(m, s, hash((m, s)) & 0xffff) for m in models for s in SEEDS]
    with ThreadPoolExecutor(12) as ex:
        results = list(ex.map(lambda j: run_round(*j), jobs))
    json.dump(results, open(out, "w"), indent=1)
    summ = {}
    for r in results:
        s = summ.setdefault(r["model"], {"WIN": 0, "LOSS": 0, "other": 0, "cost": 0.0})
        k = r["result"] if r["result"] in ("WIN", "LOSS") else "other"
        s[k] += 1; s["cost"] += r["cost"]
    for m, s in summ.items():
        print(f"{m:45s} W{s['WIN']} L{s['LOSS']} other{s['other']}  ${s['cost']:.3f}")
