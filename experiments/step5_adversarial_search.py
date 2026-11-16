"""
Step 5: adversarial test cases inside the ORIGINAL problem (Rm | prec | Cmax, zero communication).

Search for an instance (DAG + p_ij matrix) that maximises   ratio = makespan(target) / OPT,
where OPT comes from the exact solver (proven optimal; if not proven within 2 s the solver's best
schedule is used, which makes the ratio a lower bound). Hill climbing with random restarts,
fixed seeds, best-ever instance kept.
Instance space (stated bounds): n tasks, m machines, edges only i -> j with i < j,
every p_ij in [1, 100] (so within the whole matrix the spread is at most 100x).
Mutations: resample one p_ij, rescale one task's whole row, toggle one edge.

Usage: step5_adversarial_search.py TARGET N M SECONDS SEED      TARGET in {HEFT, DLS, RANKMIN, BESTOF4, SUFFERAGE, DLSPAPER, PORTFOLIO7}
Output: results/step5_adv_{TARGET}_n{N}_m{M}_s{SEED}.json  (best instance + trajectory)
"""
import json, math, os, random, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import networkx as nx
from algorithms.heft import heft
from algorithms.exact import solve_exact
from algorithms.priority_variants import dls_variant, heft_variant, best_of_ranks, sufferage_dag

TARGETS = {"HEFT": heft, "DLS": lambda G, m: dls_variant(G, m, "mean", True, True),
           "RANKMIN": lambda G, m: heft_variant(G, m, "min"), "BESTOF4": best_of_ranks, "SUFFERAGE": sufferage_dag,
           "DLSPAPER": lambda G, m: dls_variant(G, m, "median", True, False),
           "PORTFOLIO7": lambda G, m: min((f(G, m) for f in (
               heft, lambda G, m: heft_variant(G, m, "median"), lambda G, m: heft_variant(G, m, "min"),
               lambda G, m: heft_variant(G, m, "max"), lambda G, m: dls_variant(G, m, "median", True, False),
               lambda G, m: dls_variant(G, m, "mean", True, True), lambda G, m: dls_variant(G, m, "min", True, True))),
               key=lambda r: r[1])}
LO, HI = 1.0, 100.0


def build(n, P, edges):
    G = nx.DiGraph()
    for i in range(n):
        G.add_node(i, p_ij=[float(x) for x in P[i]])
    G.add_edges_from(edges)
    return G


def score(target, n, m, P, edges):
    G = build(n, P, edges)
    _, ms = TARGETS[target](G, m)
    r = solve_exact(G, m, time_limit=2.0, num_workers=2, seed=0)
    return ms / r["ub"], ms, r["ub"], r["status"] == "OPTIMAL"


def random_instance(rng, n, m):
    """Random start graph from a random family: random edges, layered, or fork-join blocks (edges i -> j, i < j)."""
    P = [[math.exp(rng.uniform(0, math.log(HI))) for _ in range(m)] for _ in range(n)]
    fam = rng.choice(["random", "layered", "forkjoin"])
    edges = set()
    if fam == "random":
        edges = {(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < rng.choice([0.08, 0.15, 0.3])}
    elif fam == "layered":
        i = 0; layers = []
        while i < n:
            w = rng.randint(1, max(2, n // 3)); layers.append(list(range(i, min(n, i + w)))); i += w
        for a, b in zip(layers, layers[1:]):
            for v in b:
                for u in rng.sample(a, k=min(len(a), rng.randint(1, 2))):
                    edges.add((u, v))
    else:
        i = 0
        while i + 3 <= n:
            k = rng.randint(2, max(2, min(6, n - i - 2)))
            src, mids, snk = i, list(range(i + 1, min(n - 1, i + 1 + k))), min(n - 1, i + 1 + k)
            for v in mids:
                edges.add((src, v)); edges.add((v, snk))
            i = snk
    return P, edges


def mutate(rng, n, m, P, edges):
    P = [row[:] for row in P]; edges = set(edges)
    c = rng.random()
    if c < 0.5:
        P[rng.randrange(n)][rng.randrange(m)] = math.exp(rng.uniform(0, math.log(HI)))
    elif c < 0.7:
        i = rng.randrange(n); f = math.exp(rng.uniform(-1, 1))
        P[i] = [min(HI, max(LO, x * f)) for x in P[i]]
    else:
        i, j = sorted(rng.sample(range(n), 2))
        edges ^= {(i, j)}
    return P, edges


if __name__ == "__main__":
    target, n, m, seconds, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
    rng = random.Random(seed); t0 = time.time()
    best = None; evals = 0; traj = []
    while time.time() - t0 < seconds:
        P, E = random_instance(rng, n, m)                     # restart
        cur = score(target, n, m, P, E); evals += 1
        stall = 0
        while time.time() - t0 < seconds and stall < 150:
            P2, E2 = mutate(rng, n, m, P, E)
            s = score(target, n, m, P2, E2); evals += 1
            if s[0] >= cur[0] - 1e-12:
                stall = 0 if s[0] > cur[0] + 1e-9 else stall + 1
                P, E, cur = P2, E2, s
            else:
                stall += 1
            if best is None or cur[0] > best[0][0]:
                best = (cur, [r[:] for r in P], set(E)); traj.append((round(time.time() - t0, 1), round(cur[0], 4)))
    (ratio, ms, opt, proven), P, E = best
    out = {"target": target, "n": n, "m": m, "seed": seed, "evals": evals, "ratio": ratio, "makespan": ms, "opt": opt,
           "opt_proven": proven, "p_ij": P, "edges": sorted(E), "trajectory": traj}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open(f"results/step5_adv_{target}_n{n}_m{m}_s{seed}.json", "w"))
    print(f"{target} n={n} m={m} seed={seed}: best ratio {ratio:.3f} (makespan {ms:.1f} vs OPT {opt:.1f}, proven={proven}) after {evals} evals")
