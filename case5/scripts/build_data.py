"""Merge the raw tournament outputs in case5/data/raw/ into case5/data/riddles.json.

Every record comes from a real OpenRouter run of case5/scripts/sweep.py on the same
8 user-style themes:
  raw/sweep.json           Anthropic writers at low/medium effort, Gemini 3.8 Flash at high
  raw/sweep_open.json      open-weight writers at high effort
  raw/sweep_frontier.json  GPT-6.2 Astra/Sol and Gemini 4 writers at medium effort
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "data", "raw")
OUT = os.path.join(HERE, "..", "data", "riddles.json")
FILES = ["sweep.json", "sweep_open.json", "sweep_frontier.json"]
KEEP = ["model", "effort", "theme", "riddle", "chars", "story", "solver", "judge", "riddler_pos",
        "result", "real", "fact", "fit", "fit_reason", "cost", "gen_cost", "gen_secs"]


def load_rejudges():
    """raw/rejudge_*.json (from rejudge.py) -> {(raw file, index): (judge model, verdict, result)}."""
    alt = {}
    for fn in sorted(os.listdir(RAW)):
        if fn.startswith("rejudge_") and fn.endswith(".json"):
            d = json.load(open(os.path.join(RAW, fn)))
            for name, recs in d["rounds"].items():
                for rec in recs:
                    alt[(name, rec["i"])] = (d["judge_model"], rec["judge"], rec["result"])
    return alt


def main():
    alt = load_rejudges()
    rounds = []
    for name in FILES:
        path = os.path.join(RAW, name)
        if not os.path.exists(path):
            print(f"skipping missing {name}")
            continue
        for i, r in enumerate(json.load(open(path))):
            out = {k: r[k] for k in KEEP if r.get(k) is not None}
            if out.get("result") not in ("WIN", "LOSS"):
                out["result"] = "FAILED"
            if (name, i) in alt:
                out["alt_judge_model"], out["alt_judge"], out["alt_result"] = alt[(name, i)]
            rounds.append(out)
    json.dump({"rounds": rounds}, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(f"wrote {len(rounds)} rounds to {OUT}")


if __name__ == "__main__":
    main()
