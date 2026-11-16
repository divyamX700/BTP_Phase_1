"""
eoh_auto/replay.py - re-evaluate the heuristics of a logged EoH run WITHOUT calling any LLM, and check that every fitness
equals the logged one (the evaluation is deterministic). Optionally re-derive the population of every generation.
Usage: python -m eoh_auto.replay <run name> [max candidates]
"""
import json, os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from eoh_auto.evaluate import Evaluator

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    name = sys.argv[1]; cap = int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 9
    d = os.path.join(ROOT, "eoh_auto", "runs", name)
    cands = [json.loads(l) for l in open(os.path.join(d, "candidates.jsonl"), encoding="utf-8")]
    cands = [c for c in cands if c.get("code") and not c.get("duplicate")][:cap]
    ev = Evaluator(); ok = bad = 0
    for c in cands:
        fit, err, _ = ev(c["code"])
        same = (fit is None and c["fitness"] is None) or (fit is not None and c["fitness"] is not None and abs(fit - c["fitness"]) < 1e-9)
        ok += same; bad += not same
        if not same:
            print(f"MISMATCH #{c['id']}: logged {c['fitness']} replayed {fit} ({err})")
    ev.close()
    print(f"run {name}: re-evaluated {len(cands)} logged heuristics without any LLM call: {ok} identical, {bad} different")
