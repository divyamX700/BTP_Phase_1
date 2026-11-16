"""
Check the CP-SAT model against an independent brute force on tiny random instances.

Brute force: an optimal schedule is "semi-active" (every task starts as early as its
machine predecessor and its DAG predecessors allow), so it is fully described by
(machine assignment, global processing order). We try ALL assignments and ALL
topological orders. Only feasible for n <= ~6, m = 2-3.

Run from the project root:  python tests/test_exact_vs_bruteforce.py
"""
import itertools, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import networkx as nx
from algorithms.exact import solve_exact, SCALE


def brute_force_opt(G, m):
    nodes = list(G.nodes())
    best = float("inf")
    orders = list(nx.all_topological_sorts(G))
    for assign in itertools.product(range(m), repeat=len(nodes)):
        a = dict(zip(nodes, assign))
        for order in orders:
            free = [0.0] * m
            fin = {}
            for t in order:
                k = a[t]
                ready = max((fin[p] for p in G.predecessors(t)), default=0.0)
                st = max(free[k], ready)
                fin[t] = st + G.nodes[t]["p_ij"][k]
                free[k] = fin[t]
            best = min(best, max(fin.values()))
    return best


def random_instance(rng, n, m, edge_prob, allow_zero=False):
    G = nx.DiGraph()
    for i in range(n):
        p = [round(rng.uniform(1, 50), 4) for _ in range(m)]
        if allow_zero and rng.random() < 0.15:
            p[rng.randrange(m)] = 0.0
        G.add_node(i, p_ij=p)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < edge_prob:
                G.add_edge(i, j)
    return G


if __name__ == "__main__":
    rng = random.Random(12345)
    tol = n_max = 0
    checked = 0
    worst_rel = 0.0
    for trial in range(60):
        n = rng.choice([3, 4, 5, 6])
        m = rng.choice([2, 2, 3])
        G = random_instance(rng, n, m, rng.choice([0.0, 0.2, 0.4, 0.7]), allow_zero=(trial % 5 == 0))
        opt = brute_force_opt(G, m)
        r = solve_exact(G, m, time_limit=20, num_workers=4, seed=trial)
        assert r["status"] == "OPTIMAL", f"trial {trial}: status {r['status']}"
        slack = (n + 1) / SCALE  # rounding allowance (floor to 3 decimals, <=1/SCALE per task)
        assert r["ub"] >= opt - 1e-9, f"trial {trial}: UB {r['ub']} below brute-force optimum {opt}!"
        assert r["lb"] <= opt + 1e-9, f"trial {trial}: LB {r['lb']} above brute-force optimum {opt}!"
        assert r["ub"] - opt <= slack, f"trial {trial}: UB {r['ub']} too far above optimum {opt}"
        worst_rel = max(worst_rel, (r["ub"] - opt) / opt if opt > 0 else 0)
        checked += 1
    print(f"PASS: {checked} random tiny instances; CP-SAT interval [LB,UB] always contains the brute-force "
          f"optimum; worst relative UB excess = {worst_rel:.2e}")
