"""
Step 3 (first pass): why is HEFT worse than the best schedule on REAL instances?

For every instance (Pegasus workflows with <= 100 tasks, machines m in {4, 8}; plus TGFF graphs,
m = 4) it measures:
  heft         makespan of HEFT
  ref          best known schedule: exact solver (10 s cap), UB;  ref_proven = solver proved it optimal
  replay       HEFT's machine rule applied to the task ORDER of the best known schedule (tasks sorted by
               start time). If replay is ~ref, the machine rule is fine and only the order was wrong.
  reorder      best makespan found by searching over task orders (local search on priority keys, starting
               from HEFT's own order) with HEFT's machine rule fixed. "How much can ordering alone fix?"
  structure    n, depth, widest level, widest level / m, heterogeneity of the p_ij
Output: results/step3_failure_analysis.csv   (one row per instance, written as it goes)

Run from the project root:  python -u experiments/step3_failure_analysis.py
"""
import csv, glob, os, random, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import networkx as nx
import numpy as np
from data.pegasus_parser import parse_pegasus_dax
from data.tgff_pipeline import generate_tgff_dag
from algorithms.heft import heft, _compute_avg_exec, _compute_upward_rank
from algorithms.exact import solve_exact
from algorithms.list_scheduling import schedule_from_order, order_from_keys

EXACT_CAP = 10.0       # seconds per instance for the exact solver
SEARCH_EVALS = 3000    # max schedule evaluations in the order search
SEARCH_SECONDS = 20.0  # and a time cap per instance
OUT = os.path.join("results", "step3_failure_analysis.csv")
FIELDS = ["family", "instance", "m", "n", "edges", "depth", "max_width", "width_over_m", "hetero_ratio",
          "hetero_cv", "heft", "ref", "ref_proven", "heft_over_ref", "replay", "replay_over_ref",
          "reorder", "reorder_over_ref", "search_evals"]


def instances():
    for f in sorted(glob.glob("WF/WorkFlowBenchmark/*/*.xml")):
        if f.endswith("~"):
            continue
        if open(f, errors="ignore").read().count("<job ") <= 100:
            for m in (4, 8):
                yield "Pegasus", os.path.basename(f).replace(".xml", ""), m, parse_pegasus_dax(f, m, seed=42)
    for size in (10, 20):
        for seed in (1, 2, 3, 4):
            pre = f"data/_fa_{size}_{seed}"
            G = generate_tgff_dag(seed=seed, num_tasks=size, num_machines=4, out_prefix=pre)
            for ext in ("tgff", "tgffopt", "vcg", "eps"):
                if os.path.exists(f"{pre}.{ext}"):
                    os.remove(f"{pre}.{ext}")
            yield "TGFF", f"TGFF_{size}_seed{seed}", 4, G


def structure(G, m):
    gens = list(nx.topological_generations(G))
    width = max(len(g) for g in gens)
    P = np.array([d["p_ij"] for _, d in G.nodes(data=True)], dtype=float)
    pos = P[P.min(axis=1) > 0]
    ratio = float(np.mean(pos.max(axis=1) / pos.min(axis=1))) if len(pos) else 1.0
    cv = float(np.mean(pos.std(axis=1) / pos.mean(axis=1))) if len(pos) else 0.0
    return len(gens), width, width / m, ratio, cv


def reorder_search(G, m, start_keys, rng, ref):
    """Local search over priority keys; decoder = HEFT's machine rule. Returns (best makespan, evals)."""
    keys = dict(start_keys)
    cur = schedule_from_order(G, m, order_from_keys(G, keys))[1]
    best = cur
    scale = float(np.std(list(keys.values()))) or 1.0
    nodes = list(G.nodes())
    t0 = time.time(); evals = 1
    while evals < SEARCH_EVALS and time.time() - t0 < SEARCH_SECONDS:
        new = dict(keys)
        for t in rng.sample(nodes, k=min(len(nodes), rng.choice([1, 2, 3]))):
            new[t] = keys[t] + rng.gauss(0, scale)
        val = schedule_from_order(G, m, order_from_keys(G, new))[1]
        evals += 1
        if val <= cur + 1e-12:
            keys, cur = new, val
            best = min(best, cur)
            if best <= ref + 1e-9:
                break
    return best, evals


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    rng = random.Random(0)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for family, name, m, G in instances():
            depth, width, wm, ratio, cv = structure(G, m)
            _, h = heft(G, m)
            r = solve_exact(G, m, time_limit=EXACT_CAP, num_workers=8, seed=0)
            ref = r["ub"]
            topo = {n: i for i, n in enumerate(nx.topological_sort(G))}
            best_sched = {t: s for lst in r["schedule"].values() for (t, s, _) in lst}
            replay_order = sorted(G.nodes(), key=lambda t: (best_sched[t], topo[t]))
            replay = schedule_from_order(G, m, replay_order, validate=True)[1]
            rank = _compute_upward_rank(G, _compute_avg_exec(G, m))
            reorder, evals = reorder_search(G, m, {t: -rank[t] for t in G}, rng, ref)
            row = dict(family=family, instance=name, m=m, n=G.number_of_nodes(), edges=G.number_of_edges(),
                       depth=depth, max_width=width, width_over_m=round(wm, 3), hetero_ratio=round(ratio, 3),
                       hetero_cv=round(cv, 3), heft=round(h, 4), ref=round(ref, 4),
                       ref_proven=int(r["status"] == "OPTIMAL"), heft_over_ref=round(h / ref, 4),
                       replay=round(replay, 4), replay_over_ref=round(replay / ref, 4),
                       reorder=round(reorder, 4), reorder_over_ref=round(reorder / ref, 4), search_evals=evals)
            w.writerow(row); fh.flush()
            print(f"{name:20s} m={m} n={row['n']:3d} HEFT/ref={row['heft_over_ref']:.3f} "
                  f"replay/ref={row['replay_over_ref']:.3f} reorder/ref={row['reorder_over_ref']:.3f} "
                  f"proven={row['ref_proven']}", flush=True)
