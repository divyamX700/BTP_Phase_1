"""
eoh_auto/test_best.py - evaluate the best heuristics of a finished EoH run on the TEST set (653 instances never used in the run).
Compares with HEFT, DLS (mean, insertion) and the seven-rule portfolio; paired 95% bootstrap interval of the improvement over HEFT.
Usage: python -m eoh_auto.test_best <run name> [k=3]
"""
import json, os, sys
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from eoh_auto.evaluate import Evaluator

BEST7 = ["HEFT", "rank_median", "rank_min", "rank_max", "DLS_paper", "DLS_mean_ins", "DLS_min_ins"]


def ci(d, B=4000, seed=0):
    rng = np.random.default_rng(seed); n = len(d)
    bs = [d[rng.integers(0, n, n)].mean() for _ in range(B)]
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    name = sys.argv[1]; k = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    d = os.path.join(ROOT, "eoh_auto", "runs", name)
    last = sorted(f for f in os.listdir(d) if f.startswith("population_g"))[-1]
    pop = json.load(open(os.path.join(d, last)))["population"][:k]
    ev = Evaluator(split="test")
    bk = np.array(ev.best_known); proven = np.array([x["proven"] for x in ev.insts]) == 1
    base = {m: np.array([x["base"][m] for x in ev.insts]) for m in BEST7}
    cols = {"HEFT": base["HEFT"], "DLS (mean, ins.)": base["DLS_mean_ins"], "Best of 7": np.min([base[m] for m in BEST7], axis=0)}
    for c in pop:
        fit, err, ms = ev(c["code"])
        if err:
            print(f"#{c['id']} failed on test: {err}"); continue
        cols[f"EoH #{c['id']} ({c['strategy']}, gen {c['gen']})"] = np.array(ms)
    ev.close()
    evolved = [v for k, v in cols.items() if k.startswith("EoH")]
    if evolved:
        cols["Best of 7 + best evolved"] = np.minimum(cols["Best of 7"], evolved[0])
        cols[f"Best of 7 + top-{len(evolved)} evolved"] = np.min([cols["Best of 7"]] + evolved, axis=0)
    gh = (cols["HEFT"] / bk - 1) * 100
    rows = []
    print(f"run {name}, test set: {len(bk)} instances ({proven.sum()} with proven optimum)\n")
    print(f"{'method':38s} {'mean gap':>9s} {'proven':>8s}   improvement over HEFT [95% CI]   better/worse than HEFT")
    for nm, ms in cols.items():
        g = (ms / bk - 1) * 100; mu, lo, hi = ci(gh - g)
        b = int((ms < cols['HEFT'] - 1e-9).sum()); w = int((ms > cols['HEFT'] + 1e-9).sum())
        print(f"{nm:38s} {g.mean():8.3f}% {g[proven].mean():7.3f}%   {mu:+.3f} [{lo:+.3f}, {hi:+.3f}]          {b}/{w}")
        rows.append(dict(method=nm, mean_gap=g.mean(), proven_gap=g[proven].mean(), imp=mu, lo=lo, hi=hi, better=b, worse=w))
    json.dump(rows, open(os.path.join(d, "test_results.json"), "w"), indent=1)
    for c in pop:
        print(f"\n#{c['id']} (train gap {c['fitness']:.3f}%): {{{c['thought']}}}\n{c['code']}")
