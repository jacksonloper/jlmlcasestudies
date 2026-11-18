import json, random, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
import riddle_tourney as rt
import themed_tourney as tt

BUDGET = 5.0
spent = 0.0
lock = threading.Lock()

FACT_PROMPT = """Here is a riddle answer that relies on a real-world mechanism (a law, regulation, rule, technical property, or procedure).
ANSWER: {chars}. {story}
Is the central mechanism REAL (it exists and works roughly as described), or INVENTED/materially wrong? Minor embellishments are fine. Reply with only REAL or INVENTED on the first line, then one short sentence."""

THEMES = ["a wedding in Kerala", "a submarine crew in 1943", "a Tokyo sumo stable", "a Scottish whisky distillery",
          "a Kenyan marathon camp", "a Paris bakery strike", "a Texas oil rig", "an Icelandic fishing trawler"]

CONFIGS = [("anthropic/claude-fable-5.2", "low"), ("anthropic/claude-fable-5.2", "medium"),
           ("anthropic/claude-opus-5.5", "low"), ("anthropic/claude-opus-5.5", "medium"),
           ("anthropic/claude-sonnet-5.5", "low"), ("anthropic/claude-sonnet-5.5", "medium"),
           ("google/gemini-3.8-flash", "high")]

SONNET = "anthropic/claude-sonnet-5.5"


def call(*a, **k):
    global spent, BUDGET
    with lock:
        if spent > BUDGET:
            raise RuntimeError("budget")
    out, c = rt.chat(*a, **k)
    with lock:
        spent += c
    return out, c


def run(model, effort, theme, seed):
    rec = {"model": model, "effort": effort, "theme": theme, "cost": 0.0, "gen_cost": 0.0}
    try:
        t0 = time.time()
        gen, c = call(model, tt.GEN.format(theme=theme), 20000, effort=effort)
        rec["gen_cost"] = c; rec["cost"] += c; rec["gen_secs"] = round(time.time() - t0)
        p = rt.parse(gen)
        if not p:
            rec["result"] = "PARSE_FAIL"; return rec
        rec.update(p)
        f, c = call(SONNET, FACT_PROMPT.format(**p), 3000, effort="low")
        rec["cost"] += c; rec["real"] = f.strip().upper().startswith("REAL"); rec["fact"] = f
        fit, c = call(tt.HAIKU, tt.FIT_PROMPT.format(theme=theme, **p).replace(
            "Is this clearly a riddle about what the user asked for?",
            "Ignoring whether the facts are accurate, is this riddle clearly set in / about what the user asked for?"), 3000, effort="low")
        rec["cost"] += c; rec["fit"] = fit.strip().upper().startswith("YES")
        sol, c = call(rt.OPUS, rt.SOLVE_PROMPT.format(riddle=p["riddle"]), 1500, effort="low")
        rec["cost"] += c; rec["solver"] = sol
        mine = f"Characters: {p['chars']}. {p['story']}"
        flip = random.Random(seed).random() < 0.5
        a, b = (mine, sol) if flip else (sol, mine)
        rec["riddler_pos"] = "A" if flip else "B"
        jd, c = call(rt.OPUS, rt.JUDGE_PROMPT.format(riddle=p["riddle"], a=a, b=b), 1500, effort="low")
        rec["cost"] += c; rec["judge"] = jd
        rec["result"] = "WIN" if jd.strip()[:1].upper() == rec["riddler_pos"] else "LOSS"
    except Exception as e:
        rec["result"] = f"ERROR {e}"
    return rec


if __name__ == "__main__":
    if len(sys.argv) > 1:
        BUDGET = float(sys.argv[1]); CONFIGS = [tuple(x.split(":")) for x in sys.argv[2].split(",")]; OUT = sys.argv[3]
    else:
        OUT = "sweep.json"
    jobs = [(m, e, t, hash((m, e, t)) & 0xffff) for (m, e) in CONFIGS for t in THEMES]
    random.Random(1).shuffle(jobs)
    with ThreadPoolExecutor(14) as ex:
        res = list(ex.map(lambda j: run(*j), jobs))
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"TOTAL SPENT ${spent:.2f}")
    for m, e in CONFIGS:
        rs = [r for r in res if r["model"] == m and r["effort"] == e]
        ok = [r for r in rs if r["result"] in ("WIN", "LOSS")]
        w = sum(r["result"] == "WIN" for r in ok)
        real = sum(bool(r.get("real")) for r in ok); fit = sum(bool(r.get("fit")) for r in ok)
        wr = sum(r["result"] == "WIN" and r.get("real") for r in ok)
        g = sum(r["gen_cost"] for r in rs) / max(1, len(rs))
        secs = sum(r.get("gen_secs", 0) for r in rs) / max(1, len(rs))
        print(f"{m.split('/')[1]:18s} {e:6s} win {w}/{len(ok)}  real {real}/{len(ok)}  win&real {wr}/{len(ok)}  "
              f"on-theme {fit}/{len(ok)}  gen ${g:.3f}/riddle  {secs:.0f}s  fails {len(rs)-len(ok)}")
