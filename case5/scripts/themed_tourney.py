import json, random, sys
from concurrent.futures import ThreadPoolExecutor
import riddle_tourney as rt

THEME_LINE = ("THEME (requested by a user; the riddle and story must clearly be about this, "
              "using its specific people, place and era where given): {theme}")
GEN = rt.GEN_PROMPT.replace("Setting seed: {seed}", THEME_LINE)

FIT_PROMPT = """A user asked for a riddle about: "{theme}".
Here is the riddle and its intended answer.
RIDDLE: {riddle}
ANSWER: {chars}. {story}
Is this clearly a riddle about what the user asked for? Reply with only YES or NO on the first line, then a short reason."""

HAIKU = "anthropic/claude-haiku-5.5"


def run(model, theme, seed):
    rec = {"model": model, "theme": theme, "cost": 0.0}
    try:
        gen, c = rt.chat(model, GEN.format(theme=theme), 20000, effort="medium")
        rec["cost"] += c
        p = rt.parse(gen)
        if not p:
            rec["result"] = "PARSE_FAIL"; return rec
        rec.update(p)
        fit, c = rt.chat(HAIKU, FIT_PROMPT.format(theme=theme, **p), 400)
        rec["cost"] += c; rec["fit"] = fit.strip()[:3].upper().startswith("YES"); rec["fit_reason"] = fit
        sol, c = rt.chat(rt.OPUS, rt.SOLVE_PROMPT.format(riddle=p["riddle"]), 1500, effort="low")
        rec["cost"] += c; rec["solver"] = sol
        mine = f"Characters: {p['chars']}. {p['story']}"
        flip = random.Random(seed).random() < 0.5
        a, b = (mine, sol) if flip else (sol, mine)
        rec["riddler_pos"] = "A" if flip else "B"
        jd, c = rt.chat(rt.OPUS, rt.JUDGE_PROMPT.format(riddle=p["riddle"], a=a, b=b), 1500, effort="low")
        rec["cost"] += c; rec["judge"] = jd
        pick = jd.strip()[:1].upper()
        rec["result"] = "WIN" if pick == rec["riddler_pos"] else "LOSS"
    except Exception as e:
        rec["result"] = f"ERROR {e}"
    return rec


if __name__ == "__main__":
    models = sys.argv[1].split(",")
    themes = sys.argv[2].split("|")
    out = sys.argv[3]
    jobs = [(m, t, i * 7919 + j) for i, m in enumerate(models) for j, t in enumerate(themes)]
    with ThreadPoolExecutor(12) as ex:
        res = list(ex.map(lambda x: run(*x), jobs))
    json.dump(res, open(out, "w"), indent=1)
    for m in models:
        rs = [r for r in res if r["model"] == m]
        w = sum(r["result"] == "WIN" for r in rs); n = sum(r["result"] in ("WIN", "LOSS") for r in rs)
        f = sum(bool(r.get("fit")) for r in rs)
        print(f"{m:35s} wins {w}/{n}  on-theme {f}/{len(rs)}  ${sum(r['cost'] for r in rs):.2f}")
