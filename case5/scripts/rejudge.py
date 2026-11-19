"""Re-judge every completed round, optionally with the two stories in swapped order.

Uses the riddle and both stories from the sweep and the exact sweep.JUDGE_PROMPT. Without
"swap", the A/B order is the one the sweep's Opus judge saw, so only the judge model changes.
With "swap", the order is reversed. Judging each round in both orders cancels the judges'
preference for whichever story is shown first.

Usage: python rejudge.py BUDGET_USD JUDGE_MODEL EFFORT [swap]
Writes case5/data/raw/rejudge_<model-slug>[_swapped].json:
  {"judge_model", "effort", "swapped", "rounds": {raw file name: [{"i", "judge", "result", "riddler_pos"}]}}
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


def rejudge(r, model, effort, swap):
    """Judge one round. With swap, the two stories are shown in the opposite order from the sweep."""
    pos = r["riddler_pos"]
    if swap:
        pos = "B" if pos == "A" else "A"
    mine = f"Characters: {r['chars']}. {r['story']}"
    a, b = (mine, r["solver"]) if pos == "A" else (r["solver"], mine)
    jd, _ = sweep.chat(model, sweep.JUDGE_PROMPT.format(riddle=r["riddle"], a=a, b=b), 4000, effort=effort)
    pick = jd.strip()[:1].upper()
    result = "WIN" if pick == pos else ("LOSS" if pick in ("A", "B") else "FAILED")
    return jd, result, pos


def main():
    sweep.budget = float(sys.argv[1])
    model, effort = sys.argv[2], sys.argv[3]
    swap = len(sys.argv) > 4 and sys.argv[4] == "swap"
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
            jd, res, pos = rejudge(r, model, effort, swap)
        except Exception as e:
            jd, res, pos = f"ERROR {e}", "FAILED", None
        return name, {"i": i, "judge": jd, "result": res, "riddler_pos": pos}

    out = {}
    with ThreadPoolExecutor(14) as ex:
        for name, rec in ex.map(work, jobs):
            out.setdefault(name, []).append(rec)
    slug = model.replace("/", "_") + ("_swapped" if swap else "")
    json.dump({"judge_model": model, "effort": effort, "swapped": swap, "rounds": out},
              open(os.path.join(RAW, f"rejudge_{slug}.json"), "w"), indent=1)
    n = sum(len(v) for v in out.values())
    print(f"re-judged {n} rounds with {model} for ${sweep.spent:.2f}")


if __name__ == "__main__":
    main()
