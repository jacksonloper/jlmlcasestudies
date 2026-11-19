"""Hidden-fact count sweep: 1, 2 (the main recipe) or 3 layered hidden facts.

Same 8 themes as sweep.py, REPS riddles per theme per level, one writer at one effort. Only the
recipe block of sweep.GEN_PROMPT changes (level 2 is exactly sweep.GEN_PROMPT). Each riddle is
fact-checked, theme-checked, solved by Opus 5.5 (low), then judged by Claude Opus 5.5 and GPT-6.2 Sol,
each in both story orders, and scored 1 / 0.5 / 0 per judge.

Usage: python facts_sweep.py BUDGET_USD WRITER EFFORT REPS OUT.json
Writes raw/facts_<...>.json records with: writer, effort, level, theme, rep, riddle, chars, story,
gen_cost, real, fact, fit, solver, verdicts {opus: [2], sol: [2]}, score_opus, score_sol, result.
"""
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sweep  # noqa: E402

SOL = "openai/gpt-6.2-sol"

BASE = sweep.GEN_PROMPT
START = BASE.index("Proven recipe: LAYERED.")
END = BASE.index("- Your story must contradict no line")
MECH = ("rests on a REAL-WORLD mechanism (a regulation, contract clause, technical property, or procedure), "
        "stated explicitly in your story")
LESSER = ("- The mechanism may be known, but the riddle must contain one odd detail that only a LESSER-KNOWN "
          "consequence of it explains, so a solver who knows the famous part still leaves that detail unexplained.\n")
RECIPES = {
    1: ("Recipe: ONE hidden fact.\n"
        f"- A single hidden fact {MECH}. It must explain the riddle's odd details (a timing, number, reaction, or choice).\n"
        + LESSER, "the hidden fact"),
    2: (BASE[START:END], "both hidden facts"),
    3: ("Recipe: LAYERED, THREE hidden facts.\n"
        "- Hidden fact #1 is findable from the riddle's odd details (solvers usually find it).\n"
        f"- Hidden fact #2 builds on #1 and {MECH}. It must explain one specific odd detail in the riddle (a timing, number, reaction, or choice).\n"
        "- Hidden fact #3 builds on #2 (another real-world mechanism or a consequence of it) and must explain a "
        "DIFFERENT odd detail that #1 and #2 leave unexplained.\n"
        + LESSER, "all three hidden facts"),
}


def gen_prompt(theme, level):
    recipe, phrase = RECIPES[level]
    p = BASE[:START] + recipe + BASE[END:]
    p = p.replace("<two sentences with both hidden facts>", f"<two sentences with {phrase}>")
    return p.format(theme=theme)


def judge(model, r, pos):
    mine = f"Characters: {r['chars']}. {r['story']}"
    a, b = (mine, r["solver"]) if pos == "A" else (r["solver"], mine)
    jd, _ = sweep.chat(model, sweep.JUDGE_PROMPT.format(riddle=r["riddle"], a=a, b=b), 4000, effort="low")
    pick = sweep.judge_pick(jd)
    return {"judge": jd, "riddler_pos": pos, "result": "WIN" if pick == pos else ("LOSS" if pick else "FAILED")}


def run(writer, effort, level, theme, rep, seed):
    rec = {"writer": writer, "effort": effort, "level": level, "theme": theme, "rep": rep, "gen_cost": 0.0}
    try:
        gen, c = sweep.chat(writer, gen_prompt(theme, level), 20000, effort=effort)
        rec["gen_cost"] = c
        p = sweep.parse(gen)
        if not p:
            rec["result"] = "PARSE_FAIL"; return rec
        rec.update(p)
        f, _ = sweep.chat(sweep.SONNET, sweep.FACT_PROMPT.format(**p), 3000, effort="low")
        rec["real"] = f.strip().upper().startswith("REAL"); rec["fact"] = f
        fit, _ = sweep.chat(sweep.HAIKU, sweep.FIT_PROMPT.format(theme=theme, **p), 3000, effort="low")
        rec["fit"] = fit.strip().upper().startswith("YES")
        sol, _ = sweep.chat(sweep.OPUS, sweep.SOLVE_PROMPT.format(riddle=p["riddle"]), 1500, effort="low")
        rec["solver"] = sol
        first = "A" if random.Random(seed).random() < 0.5 else "B"
        second = "B" if first == "A" else "A"
        rec["verdicts"] = {}
        for key, model in (("opus", sweep.OPUS), ("sol", SOL)):
            vs = [judge(model, rec, first), judge(model, rec, second)]
            rec["verdicts"][key] = vs
            if all(v["result"] != "FAILED" for v in vs):
                rec[f"score_{key}"] = sum(v["result"] == "WIN" for v in vs) / 2
        rec["result"] = "OK"
    except Exception as e:
        rec["result"] = f"ERROR {e}"
    return rec


def main():
    sweep.budget = float(sys.argv[1])
    writer, effort, reps, out = sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
    jobs = [(writer, effort, lv, t, rep, random.randrange(1 << 16))
            for lv in RECIPES for t in sweep.THEMES for rep in range(reps)]
    random.shuffle(jobs)
    with ThreadPoolExecutor(14) as ex:
        res = list(ex.map(lambda j: run(*j), jobs))
    json.dump(res, open(out, "w"), indent=1)
    print(f"TOTAL SPENT ${sweep.spent:.2f}")
    for lv in RECIPES:
        ok = [r for r in res if r["level"] == lv and "score_opus" in r and "score_sol" in r]
        w = sum((r["score_opus"] + r["score_sol"]) / 2 for r in ok)
        print(f"{lv} hidden fact(s): n={len(ok)} avg win rate {w / max(1, len(ok)):.2f}")


if __name__ == "__main__":
    main()
