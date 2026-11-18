"""Merge the raw tournament outputs in case5/data/raw/ into case5/data/riddles.json.

Every record comes from a real OpenRouter run of the scripts in this folder:
  riddle_tourney.py  -> raw/results1.json, results2.json, results3.json   (Experiment 1: setting seeds, no theme)
  themed_tourney.py  -> raw/themed1.json                                  (Experiment 2: pilot with user themes)
  sweep.py           -> raw/sweep.json, raw/sweep_open.json               (Experiment 3: user themes + fact check)

results1.json rounds for models that were re-run in results2.json are dropped: in that first
run the reasoning models exhausted their token budget before writing a riddle (a harness bug,
fixed for results2.json by raising max_tokens and setting reasoning effort to medium).
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "data", "raw")
OUT = os.path.join(HERE, "..", "data", "riddles.json")

RERUN = {"deepseek/deepseek-v4.1-flash-1017", "qwen/qwen3.8-flash", "moonshotai/kimi-k2.6", "anthropic/claude-fable-5.2"}

KEEP = ["model", "effort", "theme", "seed", "riddle", "chars", "story", "solver", "judge", "riddler_pos",
        "result", "real", "fact", "fit", "fit_reason", "cost", "gen_cost", "gen_secs"]


def load(name):
    p = os.path.join(RAW, name)
    return json.load(open(p)) if os.path.exists(p) else []


def norm(r, experiment, default_effort):
    out = {k: r.get(k) for k in KEEP if r.get(k) is not None}
    out["experiment"] = experiment
    out.setdefault("effort", default_effort)
    out["prompt"] = r.get("theme") or r.get("seed")
    if not str(out.get("result", "")).startswith(("WIN", "LOSS")):
        out["result"] = "FAILED"
    return out


def main():
    rounds = []
    for r in load("results1.json"):
        if r["model"] not in RERUN:
            rounds.append(norm(r, "exp1", "default"))
    for r in load("results2.json"):
        rounds.append(norm(r, "exp1", "medium"))
    for r in load("results3.json"):
        rounds.append(norm(r, "exp1", "medium"))
    for r in load("themed1.json"):
        rounds.append(norm(r, "exp2", "medium"))
    for r in load("sweep.json"):
        rounds.append(norm(r, "exp3", r.get("effort")))
    for r in load("sweep_open.json"):
        rounds.append(norm(r, "exp3", r.get("effort")))
    json.dump({"rounds": rounds}, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(f"wrote {len(rounds)} rounds to {OUT}")


if __name__ == "__main__":
    main()
