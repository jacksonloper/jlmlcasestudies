"""Case Study 5: riddle tournament with user themes and a fact check.

For each (writer model, reasoning effort) and each of 8 user-style themes:
  1. the writer produces a riddle plus its intended answer (GEN_PROMPT),
  2. Claude Sonnet 5.5 fact-checks the answer's mechanism (FACT_PROMPT),
  3. Claude Haiku 5.5 checks the riddle is on theme (FIT_PROMPT),
  4. Claude Opus 5.5 at low effort with a 1,500-token cap solves it (SOLVE_PROMPT),
  5. Claude Opus 5.5 at low effort judges both stories blind, in random order (JUDGE_PROMPT).
All calls go through OpenRouter. The key is injected by the network proxy, so the
Authorization header here carries a placeholder.

Usage: python sweep.py BUDGET_USD model:effort[,model:effort...] OUT.json
"""
import json
import random
import re
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "https://openrouter.ai/api/v1/chat/completions"
OPUS = "anthropic/claude-opus-5.5"
SONNET = "anthropic/claude-sonnet-5.5"
HAIKU = "anthropic/claude-haiku-5.5"

GEN_PROMPT = "Design ONE riddle round.\n\nRULES: answer = two characters (one with an occupation) + a two-sentence story. Riddle max 4 lines. Hidden facts may appear only in your answer. A strong AI solver may also invent hidden facts. A blind judge picks whichever story explains the riddle BETTER (every odd detail, fewest stretches). You win if yours is preferred.\n\nProven recipe: LAYERED.\n- Hidden fact #1 is findable from the riddle's odd details (solvers usually find it).\n- Hidden fact #2 builds on #1 and rests on a REAL-WORLD mechanism (a regulation, contract clause, technical property, or procedure), stated explicitly in your story. It must explain one specific odd detail in the riddle (a timing, number, reaction, or choice).\n- The mechanism may be known, but the riddle must contain one odd detail that only a LESSER-KNOWN consequence of it explains, so a solver who knows the famous part still leaves that detail unexplained.\n- Your story must contradict no line of the riddle. Surprising, not sentimental. Never name the occupation in the riddle.\n\nTHEME (requested by a user; the riddle and story must clearly be about this, using its specific people, place and era where given): {theme}\n\nReply in exactly this format and nothing else:\nRIDDLE: <text>\nCHARACTERS: <char 1>; <char 2 (occupation)>\nSTORY: <two sentences with both hidden facts>"

SOLVE_PROMPT = "You have 10 seconds. This riddle's answer is a tiny story: two characters (one with an occupation) and a two-sentence story. The intended answer may include hidden facts not stated in the riddle that explain every odd detail; you may invent such facts too. A judge will pick whichever story explains the riddle better. Reply with: Character 1, Character 2 (with occupation), and a two-sentence story.\n\n{riddle}"

JUDGE_PROMPT = 'Below is a riddle and two candidate stories that each claim to explain it. Stories may include hidden facts not stated in the riddle. Pick the story that explains the riddle BETTER: accounts for every odd detail most completely and naturally, with the fewest stretches. Reply with only "A" or "B" on the first line, then a one-sentence reason.\n\nRIDDLE:\n{riddle}\n\nSTORY A: {a}\n\nSTORY B: {b}'

FACT_PROMPT = 'Here is a riddle answer that relies on a real-world mechanism (a law, regulation, rule, technical property, or procedure).\nANSWER: {chars}. {story}\nIs the central mechanism REAL (it exists and works roughly as described), or INVENTED/materially wrong? Minor embellishments are fine. Reply with only REAL or INVENTED on the first line, then one short sentence.'

FIT_PROMPT = 'A user asked for a riddle about: "{theme}".\nHere is the riddle and its intended answer.\nRIDDLE: {riddle}\nANSWER: {chars}. {story}\nIgnoring whether the facts are accurate, is this riddle clearly set in / about what the user asked for? Reply with only YES or NO on the first line, then a short reason.'

THEMES = ['a wedding in Kerala', 'a submarine crew in 1943', 'a Tokyo sumo stable', 'a Scottish whisky distillery', 'a Kenyan marathon camp', 'a Paris bakery strike', 'a Texas oil rig', 'an Icelandic fishing trawler']

budget = 0.0
spent = 0.0
lock = threading.Lock()


def chat(model, prompt, max_tokens, effort=None, retries=3):
    global spent
    with lock:
        if spent > budget:
            raise RuntimeError("budget")
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}
    if effort:
        body["reasoning"] = {"effort": effort}
    data = json.dumps(body).encode()
    err = None
    for i in range(retries):
        try:
            req = urllib.request.Request(URL, data=data, headers={
                "Authorization": "Bearer proxy-injected", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=240) as r:
                j = json.load(r)
            msg = j["choices"][0]["message"].get("content") or ""
            cost = (j.get("usage") or {}).get("cost", 0) or 0
            with lock:
                spent += cost
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


def run(model, effort, theme, seed):
    rec = {"model": model, "effort": effort, "theme": theme, "cost": 0.0, "gen_cost": 0.0}
    try:
        t0 = time.time()
        gen, c = chat(model, GEN_PROMPT.format(theme=theme), 20000, effort=effort)
        rec["gen_cost"] = c; rec["cost"] += c; rec["gen_secs"] = round(time.time() - t0)
        p = parse(gen)
        if not p:
            rec["result"] = "PARSE_FAIL"; return rec
        rec.update(p)
        f, c = chat(SONNET, FACT_PROMPT.format(**p), 3000, effort="low")
        rec["cost"] += c; rec["real"] = f.strip().upper().startswith("REAL"); rec["fact"] = f
        fit, c = chat(HAIKU, FIT_PROMPT.format(theme=theme, **p), 3000, effort="low")
        rec["cost"] += c; rec["fit"] = fit.strip().upper().startswith("YES"); rec["fit_reason"] = fit
        sol, c = chat(OPUS, SOLVE_PROMPT.format(riddle=p["riddle"]), 1500, effort="low")
        rec["cost"] += c; rec["solver"] = sol
        mine = f"Characters: {p['chars']}. {p['story']}"
        flip = random.Random(seed).random() < 0.5
        a, b = (mine, sol) if flip else (sol, mine)
        rec["riddler_pos"] = "A" if flip else "B"
        jd, c = chat(OPUS, JUDGE_PROMPT.format(riddle=p["riddle"], a=a, b=b), 1500, effort="low")
        rec["cost"] += c; rec["judge"] = jd
        rec["result"] = "WIN" if jd.strip()[:1].upper() == rec["riddler_pos"] else "LOSS"
    except Exception as e:
        rec["result"] = f"ERROR {e}"
    return rec


def main():
    global budget
    budget = float(sys.argv[1])
    configs = [tuple(x.split(":")) for x in sys.argv[2].split(",")]
    out = sys.argv[3]
    jobs = [(m, e, t, random.randrange(1 << 16)) for (m, e) in configs for t in THEMES]
    random.shuffle(jobs)
    with ThreadPoolExecutor(14) as ex:
        res = list(ex.map(lambda j: run(*j), jobs))
    json.dump(res, open(out, "w"), indent=1)
    print(f"TOTAL SPENT ${spent:.2f}")
    for m, e in configs:
        rs = [r for r in res if r["model"] == m and r["effort"] == e]
        ok = [r for r in rs if r["result"] in ("WIN", "LOSS")]
        w = sum(r["result"] == "WIN" for r in ok)
        wr = sum(r["result"] == "WIN" and bool(r.get("real")) for r in ok)
        g = sum(r["gen_cost"] for r in rs)
        print(f"{m:32s} {e:6s} win {w}/{len(ok)}  win&real {wr}/{len(ok)}  "
              f"$/win {g / w if w else float('nan'):.3f}  $/win&real {g / wr if wr else float('nan'):.3f}  fails {len(rs) - len(ok)}")


if __name__ == "__main__":
    main()
