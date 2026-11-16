"""
data/tgff_graphs.py - TGFF graph-only generator with shape controls (no machine times).

Shapes (style):
  "degree": TGFF's classic generator; max_in / max_out limit the number of parents / children per task.
  "sp"    : series-parallel generator; width / length set the number and length of parallel chains,
            rejoin (0/1) forces chains to rejoin, xover adds extra edges between chains.
The task count `task_cnt` is only a rough target (TGFF returns about 1-2x of it); read the real size from the graph.
Machine times are NOT taken from TGFF here; see data/heterogeneity.py.
"""
import os, re, subprocess, uuid
import networkx as nx

_ROOT = os.path.join(os.path.dirname(__file__), "..")
_TMP = os.path.join(os.path.dirname(__file__), "_tgff_tmp")


def tgff_graph(seed: int, task_cnt: int, mult: int = 1, style: str = "degree", max_in: int = 2, max_out: int = 2,
               width: int = 4, length: int = 4, rejoin: int = 1, xover: int = 0) -> nx.DiGraph:
    os.makedirs(_TMP, exist_ok=True)
    name = f"g_{uuid.uuid4().hex[:10]}"
    rel_prefix = f"data/_tgff_tmp/{name}"
    lines = [f"seed {seed}", "tg_cnt 1", f"task_cnt {task_cnt} {mult}", "period_laxity 1", "period_mul 1"]
    if style == "degree":
        lines.append(f"task_degree {max_in} {max_out}")
    elif style == "sp":
        lines += ["gen_series_parallel", f"series_wid {width} {max(1, width // 2)}",
                  f"series_len {length} {max(1, length // 2)}", f"series_must_rejoin {rejoin}",
                  f"series_local_xover {xover}"]
    else:
        raise ValueError(style)
    lines.append("tg_write")
    with open(os.path.join(_ROOT, rel_prefix + ".tgffopt"), "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    subprocess.run(["wsl", "-d", "Ubuntu", "-e", "./tgff-3.5/tgff", rel_prefix], cwd=_ROOT, check=True, capture_output=True)
    G = nx.DiGraph()
    out = os.path.join(_ROOT, rel_prefix + ".tgff")
    with open(out) as f:
        for line in f:
            p = line.split()
            if len(p) >= 2 and p[0] == "TASK":
                G.add_node(p[1])
            elif len(p) >= 6 and p[0] == "ARC":
                G.add_edge(p[3], p[5])
    for ext in ("tgff", "tgffopt", "vcg", "eps"):
        fp = os.path.join(_ROOT, rel_prefix + "." + ext)
        if os.path.exists(fp):
            os.remove(fp)
    if G.number_of_nodes() == 0 or not nx.is_directed_acyclic_graph(G):
        raise RuntimeError(f"TGFF produced an empty or cyclic graph for seed={seed}, style={style}")
    return G
