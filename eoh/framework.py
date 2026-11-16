"""
eoh/framework.py - harness for evolving scheduling heuristics (Evolution of Heuristics style).

A heuristic is a function  score(f) -> float  where f is a dict of features of one (ready task, machine) pair.
The scheduler repeatedly evaluates every ready task on every machine, places the pair with the HIGHEST score
(ties: lower task index, then lower machine index) at its earliest start on that machine (idle gaps allowed),
and repeats until all tasks are placed. Every schedule is checked by the strict validator.

Features in f (all floats):
  task, static:   p_mean, p_min, p_med, p_max, p_std   (task's times over machines)
                  rank_mean, rank_min, rank_med          (longest chain to the exit, own time included, using that statistic)
                  desc (number of descendants), nsucc, npred, depth (edges from the start)
  pair (t, k):    p (time of t on k), est (earliest start on k), eft (= est + p), ready (time all parents are done),
                  mfree (time machine k finishes its last placed task), eft_best, eft_2nd (best / second-best eft of t
                  over all machines), p_rel (= p / p_min)
  global:         n, m, n_ready, done (tasks already placed)
Instances: the main suite of the progress report. TRAIN = first batch (161 instances, seeds 1-2), TEST = the other three
batches (653 instances), regenerated deterministically and matched to the stored CSV rows to get the best-known makespans.
"""
import glob, importlib, math, os, pickle, statistics, sys
from multiprocessing import Pool
import numpy as np, pandas as pd, networkx as nx

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "experiments"))
CACHE = os.path.join(ROOT, "eoh", "_cache"); os.makedirs(CACHE, exist_ok=True)
BASE_METHODS = ["HEFT", "rank_median", "rank_min", "rank_max", "best_of_4", "Sufferage", "DLS_paper", "DLS_mean_ins",
                "DLS_mean_noins", "DLS_min_ins", "MinMin", "MaxMin"]
BATCHES = {"train": [("step7_suite.csv", [1, 2], 2026)],
           "test": [("step7_suite_b.csv", [3, 4, 5], 11), ("step7_suite_c.csv", [6, 7, 8], 12), ("step7_suite_d.csv", [9, 10, 11], 13)]}


# ----------------------------------------------------------------------------- instances
def _build(seeds, rng_seed):
    old = sys.argv; sys.argv = ["x"]
    import step7_variety_suite as s7
    sys.argv = old
    s7.SEEDS, s7.RNG_SEED = seeds, rng_seed
    return s7.build_tasks()


def _key(t):
    P = np.array(t["p_ij"], dtype=float); pos = P[P.min(axis=1) > 0]
    hr = float(np.mean(pos.max(axis=1) / pos.min(axis=1))) if len(pos) else 1.0
    cv = float(P.mean(axis=1).std() / max(P.mean(), 1e-9))
    mt = t["meta"]
    return (mt["family"], mt["graph"], str(mt["tag"]), int(mt["m"]), int(mt["r_mach"]), mt["consistency"], len(t["p_ij"]), len(t["edges"]), round(hr, 9), round(cv, 9))


def load(split):
    path = os.path.join(CACHE, f"{split}.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    out = []
    for csv, seeds, rs in BATCHES[split]:
        df = pd.read_csv(os.path.join(ROOT, "results", csv))
        df["key"] = list(zip(df.family, df.graph, df.tag.astype(str), df.m.astype(int), df.r_mach.astype(int), df.consistency,
                             df.n.astype(int), df.edges.astype(int), df.hetero_ratio.round(9), df.cv_task.round(9)))
        rows = {k: r for k, r in zip(df.key, df.to_dict("records"))}
        if len(rows) != len(df):
            raise RuntimeError(f"{csv}: instance keys are not unique")
        for t in _build(seeds, rs):
            r = rows[_key(t)]          # KeyError here would mean the regenerated instance differs from the stored one
            bk = min([r["ref"]] + [r[f"ms_{k}"] for k in BASE_METHODS])
            out.append(dict(m=t["m"], p=t["p_ij"], edges=t["edges"], meta=t["meta"], ref_solver=r["ref"], proven=int(r["ref_proven"]),
                            best_known=bk, base={k: r[f"ms_{k}"] for k in BASE_METHODS}))
    for inst in out:
        _prepare(inst)
    pickle.dump(out, open(path, "wb"))
    return out


def _prepare(inst):
    n = len(inst["p"]); G = nx.DiGraph(); G.add_nodes_from(range(n)); G.add_edges_from(inst["edges"])
    topo = list(nx.topological_sort(G))
    preds = [list(G.predecessors(i)) for i in range(n)]; succs = [list(G.successors(i)) for i in range(n)]
    P = inst["p"]; st = {}
    st["p_mean"] = [statistics.fmean(r) for r in P]; st["p_min"] = [min(r) for r in P]; st["p_max"] = [max(r) for r in P]
    st["p_med"] = [statistics.median(r) for r in P]; st["p_std"] = [statistics.pstdev(r) for r in P]
    for w, name in (("p_mean", "rank_mean"), ("p_min", "rank_min"), ("p_med", "rank_med")):
        rk = [0.0] * n
        for v in reversed(topo):
            rk[v] = st[w][v] + max((rk[s] for s in succs[v]), default=0.0)
        st[name] = rk
    depth = [0] * n
    for v in topo:
        depth[v] = max((depth[u] + 1 for u in preds[v]), default=0)
    st["depth"] = depth; st["nsucc"] = [len(s) for s in succs]; st["npred"] = [len(p) for p in preds]
    st["desc"] = [len(nx.descendants(G, v)) for v in range(n)]
    inst.update(n=n, preds=preds, succs=succs, static=st)


# ----------------------------------------------------------------------------- scheduler
def _slot(intervals, ready, dur):
    """earliest start >= ready in a machine timeline (sorted list of (s, f)) with an idle gap of length dur."""
    cand = ready
    for (s, f) in intervals:
        if cand + dur <= s + 1e-12:
            return cand
        if f > cand:
            cand = f
    return cand


def schedule(inst, score):
    n, m, P = inst["n"], inst["m"], inst["p"]; preds, succs, st = inst["preds"], inst["succs"], inst["static"]
    timeline = [[] for _ in range(m)]; mfree = [0.0] * m; finish = [None] * n; mach = [None] * n; start = [None] * n
    indeg = [len(p) for p in preds]; ready = [i for i in range(n) if indeg[i] == 0]; done = 0
    keys = list(st.keys())
    while ready:
        best = None
        nr = len(ready)
        for t in ready:
            rt = max((finish[u] for u in preds[t]), default=0.0)
            opts = []
            for k in range(m):
                s = _slot(timeline[k], rt, P[t][k]); opts.append((s + P[t][k], s, k))
            efts = sorted(o[0] for o in opts)
            e1 = efts[0]; e2 = efts[1] if m > 1 else efts[0]
            base = {key: st[key][t] for key in keys}
            base.update(n=n, m=m, n_ready=nr, done=done, ready=rt, eft_best=e1, eft_2nd=e2)
            for (e, s, k) in opts:
                f = dict(base); f.update(p=P[t][k], est=s, eft=e, mfree=mfree[k], p_rel=P[t][k] / st["p_min"][t] if st["p_min"][t] > 0 else 1.0)
                sc = score(f)
                if sc != sc:  # NaN
                    sc = -math.inf
                key = (sc, -t, -k)
                if best is None or key > best[0]:
                    best = (key, t, k, s, e)
        _, t, k, s, e = best
        tl = timeline[k]; tl.append((s, e)); tl.sort()
        mfree[k] = max(mfree[k], e); finish[t] = e; mach[t] = k; start[t] = s; done += 1
        ready.remove(t)
        for v in succs[t]:
            indeg[v] -= 1
            if indeg[v] == 0:
                ready.append(v)
    _validate(inst, mach, start, finish)
    return max(finish)


def _validate(inst, mach, start, finish):
    P = inst["p"]; n = inst["n"]
    assert all(f is not None for f in finish), "unscheduled task"
    for i in range(n):
        assert abs(finish[i] - start[i] - P[i][mach[i]]) < 1e-6 and start[i] >= -1e-9
    for (u, v) in inst["edges"]:
        assert start[v] >= finish[u] - 1e-9, "precedence violated"
    by = {}
    for i in range(n):
        by.setdefault(mach[i], []).append((start[i], finish[i]))
    for iv in by.values():
        iv.sort()
        for a, b in zip(iv, iv[1:]):
            assert b[0] >= a[1] - 1e-9, "overlap"


# ----------------------------------------------------------------------------- evaluation
_INST = None
def _init(split):
    global _INST
    _INST = load(split)


def _work(args):
    module, name, idx = args
    fn = getattr(importlib.import_module(module), name)
    return schedule(_INST[idx], fn)


def evaluate(module, names, split="train", procs=8):
    """Returns DataFrame: one row per instance, one makespan column per heuristic name, plus reference columns."""
    insts = load(split)
    with Pool(procs, initializer=_init, initargs=(split,)) as pool:
        res = {nm: pool.map(_work, [(module, nm, i) for i in range(len(insts))], chunksize=4) for nm in names}
    df = pd.DataFrame({nm: res[nm] for nm in names})
    df["best_known"] = [x["best_known"] for x in insts]; df["proven"] = [x["proven"] for x in insts]
    for k in BASE_METHODS:
        df[f"base_{k}"] = [x["base"][k] for x in insts]
    for c in ("n", "m"):
        df[c] = [x[c] for x in insts]
    df["consistency"] = [x["meta"]["consistency"] for x in insts]; df["r_mach"] = [x["meta"]["r_mach"] for x in insts]
    df["family"] = [x["meta"]["family"] for x in insts]
    return df


def gap(df, col):
    return (df[col] / df.best_known - 1) * 100
