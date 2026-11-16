"""
Step 7: a varied instance suite and all methods on it.
Graphs: 7 TGFF shapes x 4 sizes x 2 seeds (<= 100 tasks), plus every Pegasus workflow with 20-100 tasks.
Machine times: made with data/heterogeneity.py with randomly chosen m in {4, 8, 16}, r_mach in {3, 30},
consistency in {inconsistent, semi, consistent}; TGFF task sizes base_i ~ U(1, 20), Pegasus uses its runtimes.
Optional arguments: output csv name, comma-separated TGFF seeds, rng seed (default: the original suite).
Each graph gets 2 (TGFF) or 3 (Pegasus) different machine settings.
Reference = exact solver (2 s cap). All methods + instance features are written to results/step7_suite.csv.
Run: python -u experiments/step7_variety_suite.py   (the caller enforces a hard time limit)
"""
import csv, glob, os, random, sys, time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import networkx as nx

OUT = os.path.join("results", sys.argv[1] if len(sys.argv) > 1 else "step7_suite.csv")
SEEDS = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1, 2]
RNG_SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 2026
SHAPES = {  # name: (style, kwargs)
    "deg22": ("degree", dict(max_in=2, max_out=2)),
    "deg33": ("degree", dict(max_in=3, max_out=3)),
    "fork14": ("degree", dict(max_in=1, max_out=4)),
    "deg24": ("degree", dict(max_in=2, max_out=4)),
    "sp_w3": ("sp", dict(width=3, length=4, rejoin=1)),
    "sp_w8": ("sp", dict(width=8, length=2, rejoin=1)),
    "sp_x": ("sp", dict(width=6, length=3, rejoin=0, xover=20)),
}
SIZES = (15, 30, 55, 85)
HET = [(m, r, c) for m in (4, 8, 16) for r in (3, 30) for c in ("inconsistent", "semi", "consistent")]


def _evaluate(task):
    """Worker: build the graph, features, reference and all methods. task = dict."""
    from algorithms.exact import solve_exact, simple_lower_bounds
    from algorithms.heft import heft
    from algorithms.min_max_min import min_min, max_min
    from algorithms.priority_variants import heft_variant, best_of_ranks, sufferage_dag, dls_variant
    G = nx.DiGraph(); m = task["m"]
    for i, p in enumerate(task["p_ij"]):
        G.add_node(i, p_ij=p)
    G.add_edges_from(task["edges"])
    methods = {
        "HEFT": heft, "rank_median": lambda G, m: heft_variant(G, m, "median"),
        "rank_min": lambda G, m: heft_variant(G, m, "min"), "rank_max": lambda G, m: heft_variant(G, m, "max"),
        "best_of_4": best_of_ranks, "Sufferage": sufferage_dag,
        "DLS_paper": lambda G, m: dls_variant(G, m, "median", True, False),
        "DLS_mean_ins": lambda G, m: dls_variant(G, m, "mean", True, True),
        "DLS_mean_noins": lambda G, m: dls_variant(G, m, "mean", True, False),
        "DLS_min_ins": lambda G, m: dls_variant(G, m, "min", True, True),
        "MinMin": min_min, "MaxMin": max_min}
    row = dict(task["meta"])
    gens = list(nx.topological_generations(G)); P = np.array(task["p_ij"], dtype=float)
    pos = P[P.min(axis=1) > 0]
    lb = simple_lower_bounds(G, m)
    row.update(n=G.number_of_nodes(), edges=G.number_of_edges(), depth=len(gens), max_width=max(len(g) for g in gens),
               width_over_m=max(len(g) for g in gens) / m, density=G.number_of_edges() / max(1, G.number_of_nodes()),
               max_in=max(d for _, d in G.in_degree()), max_out=max(d for _, d in G.out_degree()),
               hetero_ratio=float(np.mean(pos.max(axis=1) / pos.min(axis=1))) if len(pos) else 1.0,
               cv_task=float(P.mean(axis=1).std() / max(P.mean(), 1e-9)),
               top1_share=float(np.bincount(P.argmin(axis=1), minlength=m).max() / len(P)),
               work_over_chain=float(lb["work_lb"] * m / max(lb["chain_lb"], 1e-9)))
    ref = solve_exact(G, m, time_limit=2.0, num_workers=2, seed=0)
    row.update(ref=ref["ub"], ref_proven=int(ref["status"] == "OPTIMAL"))
    for name, fn in methods.items():
        t0 = time.perf_counter(); _, ms = fn(G, m); dt = time.perf_counter() - t0
        row[f"ms_{name}"] = ms; row[f"t_{name}"] = dt
    return row


def build_tasks():
    from data.tgff_graphs import tgff_graph
    from data.pegasus_parser import parse_pegasus_dax
    from data.heterogeneity import make_p_ij
    rng = np.random.default_rng(RNG_SEED); r = random.Random(RNG_SEED)
    jobs = []  # (family, shape_or_workflow, size_tag, graph)
    specs = [(sh, tc, seed) for sh in SHAPES for tc in SIZES for seed in SEEDS]
    with ThreadPoolExecutor(8) as ex:
        futs = {ex.submit(tgff_graph, 100 + seed, tc, 1, SHAPES[sh][0], **SHAPES[sh][1]): (sh, tc, seed) for sh, tc, seed in specs}
        for f in as_completed(futs):
            sh, tc, seed = futs[f]
            G = f.result()
            if G.number_of_nodes() <= 100:
                jobs.append(("TGFF", sh, f"{tc}_s{seed}", G, None))
    for fpath in sorted(glob.glob("WF/WorkFlowBenchmark/*/*.xml")):
        if fpath.endswith("~"):
            continue
        nj = open(fpath, errors="ignore").read().count("<job ")
        if 20 <= nj <= 100:
            G = parse_pegasus_dax(fpath, 4, seed=0)
            base = [G.nodes[n]["runtime"] for n in G.nodes()]
            jobs.append(("Pegasus", os.path.basename(fpath).replace(".xml", ""), "real", G, base))
    tasks = []
    for fam, name, tag, G, base in sorted(jobs, key=lambda j: (j[0], j[1], j[2])):
        G = nx.convert_node_labels_to_integers(G, ordering="sorted") if fam == "TGFF" else nx.convert_node_labels_to_integers(G)
        n = G.number_of_nodes()
        b = base if base is not None else rng.uniform(1, 20, size=n).tolist()
        for k in range(2 if fam == "TGFF" else 3):
            m, rm, cons = r.choice(HET)
            tasks.append(dict(meta=dict(family=fam, graph=name, tag=tag, m=m, r_mach=rm, consistency=cons),
                              m=m, edges=list(G.edges()), p_ij=make_p_ij(b, m, rm, cons, rng)))
    return tasks


if __name__ == "__main__":
    t0 = time.time()
    tasks = build_tasks()
    print(f"{len(tasks)} instances built in {time.time()-t0:.0f}s", flush=True)
    os.makedirs("results", exist_ok=True)
    writer = None
    with open(OUT, "w", newline="") as fh, ProcessPoolExecutor(6) as ex:
        futs = [ex.submit(_evaluate, t) for t in tasks]
        done = 0
        for f in as_completed(futs):
            row = f.result()
            if writer is None:
                writer = csv.DictWriter(fh, fieldnames=list(row.keys())); writer.writeheader()
            writer.writerow(row); fh.flush(); done += 1
            if done % 20 == 0:
                print(f"{done}/{len(tasks)} done ({time.time()-t0:.0f}s)", flush=True)
    print(f"finished in {time.time()-t0:.0f}s", flush=True)
