"""Merge the raw tournament outputs in case5/data/raw/ into case5/data/riddles.json.

Every record comes from a real OpenRouter run of case5/scripts/sweep.py on the same
8 user-style themes:
  raw/sweep.json           Anthropic writers at low/medium effort, Gemini 3.8 Flash at high
  raw/sweep_open.json      open-weight writers at high effort
  raw/sweep_frontier.json  GPT-6.2 Astra/Sol and Gemini 4 writers at medium effort
  raw/sweep_qwen4.json     Qwen4 Max at low, medium and high effort

Each completed round is judged by two judges, each in both story orders (rejudge.py):
  Claude Opus 5.5: original order is the sweep's own verdict; raw/rejudge_anthropic_claude-opus-5.5_swapped.json
  GPT-6.2 Sol:     raw/rejudge_openai_gpt-6.2-sol.json and raw/rejudge_openai_gpt-6.2-sol_swapped.json
A judge's score for a round is 1 if it preferred the riddler's story in both orders, 0.5 if it
split, and 0 if it preferred the solver's both times.

raw/facts_*.json (facts_sweep.py) adds Claude Opus 5.5 riddles written with a three-hidden-fact
recipe, two per theme, scored the same way; they get variant "3 hidden facts".
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sweep import judge_pick  # noqa: E402
RAW = os.path.join(HERE, "..", "data", "raw")
OUT = os.path.join(HERE, "..", "data", "riddles.json")
FILES = ["sweep.json", "sweep_open.json", "sweep_frontier.json", "sweep_qwen4.json"]
KEEP = ["model", "effort", "theme", "riddle", "chars", "story", "solver",
        "result", "real", "fact", "fit", "fit_reason", "cost", "gen_cost", "gen_secs"]
OPUS = "anthropic/claude-opus-5.5"
SOL = "openai/gpt-6.2-sol"
JUDGES = {OPUS: "opus", SOL: "sol"}


def load_rejudges():
    """-> {(judge key, swapped, raw file, index): {"judge", "result", "riddler_pos"}}."""
    out = {}
    for fn in sorted(os.listdir(RAW)):
        if fn.startswith("rejudge_") and fn.endswith(".json"):
            d = json.load(open(os.path.join(RAW, fn)))
            jk = JUDGES[d["judge_model"]]
            for name, recs in d["rounds"].items():
                for rec in recs:
                    out[(jk, bool(d.get("swapped")), name, rec["i"])] = rec
    return out


def rederive(v, pos):
    """Recompute a verdict's result from its text, so a pick wrapped in markdown (**A**) counts.

    The original scripts read only the first character; two of 880 stored verdicts began with
    '**' and were misread (one as a riddler loss, one as a failure). pos is where the riddler's
    story was shown in this pass; the earliest rejudge file does not store it per record.
    """
    if not v:
        return v
    pos = v.get("riddler_pos") or pos
    pick = judge_pick(v.get("judge"))
    result = "WIN" if pick == pos else ("LOSS" if pick else "FAILED")
    return {**v, "riddler_pos": pos, "result": result}


def scores(out, verdicts):
    out["verdicts"] = verdicts
    for jk, vs in verdicts.items():
        if all(v and v["result"] in ("WIN", "LOSS") for v in vs):
            out[f"score_{jk}"] = sum(v["result"] == "WIN" for v in vs) / 2


def main():
    rej = load_rejudges()
    rounds = []
    for name in FILES:
        path = os.path.join(RAW, name)
        if not os.path.exists(path):
            print(f"skipping missing {name}")
            continue
        for i, r in enumerate(json.load(open(path))):
            out = {k: r[k] for k in KEEP if r.get(k) is not None}
            if "judge" not in r:  # no riddle was produced (PARSE_FAIL) or the round errored
                out["result"] = "FAILED"
                rounds.append(out)
                continue
            pos = r["riddler_pos"]
            flipped = "B" if pos == "A" else "A"
            first = rederive({"judge": r["judge"]}, pos)
            out["result"] = first["result"]
            scores(out, {
                "opus": [first, rederive(rej.get(("opus", True, name, i)), flipped)],
                "sol": [rederive(rej.get(("sol", False, name, i)), pos),
                        rederive(rej.get(("sol", True, name, i)), flipped)],
            })
            rounds.append(out)
    # Three-hidden-fact riddles from facts_sweep.py (same themes, 2 per theme, already judged by both
    # judges in both orders). The 1- and 2-fact levels of that run stay in the raw file only.
    for fn in sorted(os.listdir(RAW)):
        if not fn.startswith("facts_"):
            continue
        for r in json.load(open(os.path.join(RAW, fn))):
            if r.get("level") != 3:
                continue
            out = {k: r[k] for k in KEEP if r.get(k) is not None}
            out["model"] = r["writer"]
            out["variant"] = "3 hidden facts"
            if r.get("result") != "OK" or "verdicts" not in r:
                out["result"] = "FAILED"
            else:
                scores(out, {k: [rederive(v, v["riddler_pos"]) for v in vs] for k, vs in r["verdicts"].items()})
                both = [out.get("score_opus"), out.get("score_sol")]
                out["result"] = ("WIN" if sum(both) / 2 >= 0.5 else "LOSS") if None not in both else "FAILED"
            rounds.append(out)
    json.dump({"rounds": rounds}, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(f"wrote {len(rounds)} rounds to {OUT}")


if __name__ == "__main__":
    main()
