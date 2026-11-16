"""
Reproducibility check: compare a fresh rerun of the first suite batch (results/_repro_suite.csv) with the
stored one (results/step7_suite.csv), and re-evaluate the stored adversarial instances from their JSON files.
Run: python experiments/check_reproducibility.py
"""
import glob, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
import networkx as nx

a = pd.read_csv("results/step7_suite.csv")
b = pd.read_csv("results/_repro_suite.csv")
# Graph + settings alone are not unique (a workflow can receive the same machine setting twice with different random
# machine times), so the computed properties of the machine-time table are part of the key.
for x in (a, b):
    x["hr"] = x.hetero_ratio.round(9); x["cv"] = x.cv_task.round(9)
KEY = ["family", "graph", "tag", "m", "r_mach", "consistency", "n", "edges", "hr", "cv"]
assert a[KEY].drop_duplicates().shape[0] == len(a), "instance key not unique"
m = a.merge(b, on=KEY, suffixes=("_old", "_new"))
print(f"instances: stored {len(a)}, rerun {len(b)}, matched on graph+settings {len(m)}")
meth = [c[3:] for c in a.columns if c.startswith("ms_")]
for k in meth:
    d = (m[f"ms_{k}_old"] - m[f"ms_{k}_new"]).abs()
    print(f"  {k:15s} identical makespan on {int((d < 1e-6).sum())}/{len(m)}   max abs diff {d.max():.3g}")
# p_ij identical?  -> the heuristic makespans being identical already implies it; also check features
for f in ["hetero_ratio", "cv_task", "top1_share", "work_over_chain"]:
    print(f"  feature {f:16s} max abs diff {(m[f + '_old'] - m[f + '_new']).abs().max():.3g}")
pv = (m.ref_proven_old == 1) & (m.ref_proven_new == 1)
print(f"solver: proven in both runs {int(pv.sum())}, proven in only one {int((m.ref_proven_old != m.ref_proven_new).sum())}")
print(f"  proven-in-both: max |ref_old - ref_new| = {(m.ref_old - m.ref_new)[pv].abs().max():.3g}")
up = ~pv
rel = ((m.ref_old - m.ref_new) / m.ref_old * 100)[up]
print(f"  not proven in both: {int(up.sum())} instances, solver upper bound differs by {rel.abs().mean():.2f}% on average, max {rel.abs().max():.2f}%")

# adversarial instances: re-evaluate from the saved JSON (instance, not the search)
from algorithms.heft import heft
from algorithms.exact import solve_exact
from algorithms.priority_variants import dls_variant, heft_variant, best_of_ranks, sufferage_dag
T = {"HEFT": heft, "DLS": lambda G, m: dls_variant(G, m, "mean", True, True), "RANKMIN": lambda G, m: heft_variant(G, m, "min"),
     "BESTOF4": best_of_ranks, "SUFFERAGE": sufferage_dag, "DLSPAPER": lambda G, m: dls_variant(G, m, "median", True, False)}
print("\nadversarial instances re-evaluated from saved files:")
for f in sorted(glob.glob("results/step5_adv_*.json")):
    j = json.load(open(f))
    if j["target"] not in T:
        continue
    G = nx.DiGraph()
    for i, row in enumerate(j["p_ij"]):
        G.add_node(i, p_ij=row)
    G.add_edges_from(map(tuple, j["edges"]))
    _, ms = T[j["target"]](G, j["m"])
    r = solve_exact(G, j["m"], time_limit=30, num_workers=8)
    print(f"  {os.path.basename(f):38s} stored ratio {j['ratio']:.3f}  recomputed {ms / r['ub']:.3f}  optimum proven again: {r['status'] == 'OPTIMAL'}")
