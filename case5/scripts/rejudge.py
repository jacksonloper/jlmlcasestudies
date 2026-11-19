"""Re-judge every completed round with a second, non-Anthropic judge.

The riddle, both stories and the A/B order are exactly those the Opus judge saw, and the
prompt is sweep.JUDGE_PROMPT, so only the judge model changes.

Usage: python rejudge.py BUDGET_USD JUDGE_MODEL EFFORT
Writes case5/data/raw/rejudge_<model-slug>.json: {raw file name: [{"i", "judge", "result"}, ...]}.
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sweep  # noqa: E402
RAW = os.path.join(HERE, "..", "data", "raw")
FILES = ["sweep.json", "sweep_open.json", "sweep_frontier.json"]


def rejudge(r, model, effort):
    mine = f"Characters: {r['chars']}. {r['story']}"
    a, b = (mine, r["solver"]) if r["riddler_pos"] == "A" else (r["solver"], mine)
    jd, _ = sweep.chat(model, sweep.JUDGE_PROMPT.format(riddle=r["riddle"], a=a, b=b), 4000, effort=effort)
    pick = jd.strip()[:1].upper()
    result = "WIN" if pick == r["riddler_pos"] else ("LOSS" if pick in ("A", "B") else "FAILED")
    return jd, result


def main():
    sweep.budget = float(sys.argv[1])
    model, effort = sys.argv[2], sys.argv[3]
    jobs = []
    for name in FILES:
        path = os.path.join(RAW, name)
        if os.path.exists(path):
            for i, r in enumerate(json.load(open(path))):
                if r.get("result") in ("WIN", "LOSS"):
                    jobs.append((name, i, r))

    def work(job):
        name, i, r = job
        try:
            jd, res = rejudge(r, model, effort)
        except Exception as e:
            jd, res = f"ERROR {e}", "FAILED"
        return name, {"i": i, "judge": jd, "result": res}

    out = {}
    with ThreadPoolExecutor(14) as ex:
        for name, rec in ex.map(work, jobs):
            out.setdefault(name, []).append(rec)
    slug = model.replace("/", "_")
    json.dump({"judge_model": model, "effort": effort, "rounds": out},
              open(os.path.join(RAW, f"rejudge_{slug}.json"), "w"), indent=1)
    n = sum(len(v) for v in out.values())
    print(f"re-judged {n} rounds with {model} for ${sweep.spent:.2f}")


if __name__ == "__main__":
    main()
