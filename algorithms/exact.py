"""
algorithms/exact.py - Exact / reference solver for Rm|prec|Cmax  (the "measuring stick")

PURPOSE
  For an instance (DAG G, p_ij matrix, m machines) compute a certified interval
        LB  <=  OPT  <=  UB
  so that any heuristic's makespan C can be judged by  C/UB  and  C/LB.
  If the solver proves optimality then LB == UB (up to rounding, see below).

MODEL (Google OR-Tools CP-SAT)
  - s_i  : start time of task i (integer variable).
  - x_ij : boolean, task i runs on machine j.  Exactly one j per task.
  - For every (i, j) an OPTIONAL interval [s_i, s_i + d_ij) that exists only if x_ij.
  - NoOverlap on each machine over its optional intervals (one task at a time).
  - Precedence: for every edge u -> v:  s_v >= s_u + sum_j d_uj * x_uj.
  - Cmax >= every task end;  minimise Cmax.      (zero communication cost)

INTEGER ROUNDING (CP-SAT only handles integers)
  Durations are scaled by SCALE and ROUNDED DOWN (floor). The solved model is
  therefore a *relaxation* of the real problem, so its proven bound / SCALE is a valid
  LOWER bound. The solver's machine assignment and order are then RE-TIMED with the
  exact floating-point p_ij, giving a real schedule that passes the strict validator;
  its makespan is the UPPER bound.

WARM START
  HEFT's schedule is given to the solver as a hint, and the final UB is
  min(re-timed CP-SAT schedule, HEFT schedule), so UB is never worse than HEFT.
"""

import math
import networkx as nx
from ortools.sat.python import cp_model

try:
    from .base import compute_makespan, validate_schedule
    from .heft import heft
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from algorithms.base import compute_makespan, validate_schedule
    from algorithms.heft import heft

SCALE = 1000  # 3 decimal places


def simple_lower_bounds(G: nx.DiGraph, num_machines: int) -> dict:
    """
    Cheap, always-valid lower bounds on the optimal makespan.
      chain_lb : longest chain of tasks when every task takes its MINIMUM time
                 (nothing can finish before a chain of dependent tasks has run).
      work_lb  : (sum over tasks of min_j p_ij) / m
                 (total work is at least that, and m machines share it).
    """
    min_p = {n: min(d["p_ij"]) for n, d in G.nodes(data=True)}
    longest = {}
    for n in nx.topological_sort(G):
        longest[n] = min_p[n] + max((longest[p] for p in G.predecessors(n)), default=0.0)
    chain_lb = max(longest.values())
    work_lb = sum(min_p.values()) / num_machines
    return {"chain_lb": chain_lb, "work_lb": work_lb, "simple_lb": max(chain_lb, work_lb)}


def _retime(G, num_machines, assignment, int_start):
    """
    Turn (machine assignment, integer start times from CP-SAT) into a real schedule
    with exact float durations: tasks are processed in order of integer start time
    (ties: topological order) and each starts as early as machine and predecessors allow.
    """
    topo_index = {n: i for i, n in enumerate(nx.topological_sort(G))}
    order = sorted(G.nodes(), key=lambda n: (int_start[n], topo_index[n]))
    machine_free = [0.0] * num_machines
    finish = {}
    schedule = {m: [] for m in range(num_machines)}
    for t in order:
        k = assignment[t]
        ready = max((finish[p] for p in G.predecessors(t)), default=0.0)
        start = max(machine_free[k], ready)
        end = start + G.nodes[t]["p_ij"][k]
        schedule[k].append((t, start, end))
        finish[t] = end
        machine_free[k] = end
    return schedule


def solve_exact(G: nx.DiGraph, num_machines: int, time_limit: float = 60.0,
                num_workers: int = 8, seed: int = 0, use_heft_hint: bool = True) -> dict:
    """
    Returns a dict with:
      status      : 'OPTIMAL' (proved) | 'FEASIBLE' (time limit hit, gap remains) | other
      ub          : makespan of the best real, validated schedule found
      lb          : best proven lower bound (max of solver bound and the simple bounds)
      gap         : (ub - lb) / ub
      schedule    : the schedule achieving ub
      source      : 'cpsat' or 'heft' (which schedule gave ub)
      wall_time   : seconds spent in the solver
      simple_lb   : chain/work bounds (see simple_lower_bounds)
    """
    nodes = list(G.nodes())
    m = num_machines
    d = {n: [int(math.floor(p * SCALE + 1e-9)) for p in G.nodes[n]["p_ij"]] for n in nodes}
    sb = simple_lower_bounds(G, m)

    heft_sched, heft_ms = (heft(G, m) if use_heft_hint else (None, None))

    # Horizon: any schedule finishing by it is enough. Running everything on the best
    # single machine is always feasible; HEFT's makespan is usually much tighter.
    seq = min(sum(d[n][j] for n in nodes) for j in range(m))
    horizon = seq if heft_ms is None else min(seq, int(math.ceil(heft_ms * SCALE)) + len(nodes))
    horizon = max(horizon, 1)

    model = cp_model.CpModel()
    s = {n: model.new_int_var(0, horizon, f"s_{n}") for n in nodes}
    x = {(n, j): model.new_bool_var(f"x_{n}_{j}") for n in nodes for j in range(m)}
    per_machine = {j: [] for j in range(m)}
    end = {}
    for n in nodes:
        model.add_exactly_one(x[n, j] for j in range(m))
        for j in range(m):
            per_machine[j].append(
                model.new_optional_fixed_size_interval_var(s[n], d[n][j], x[n, j], f"i_{n}_{j}")
            )
        end[n] = s[n] + sum(d[n][j] * x[n, j] for j in range(m))
    for j in range(m):
        model.add_no_overlap(per_machine[j])
    for u, v in G.edges():
        model.add(s[v] >= end[u])
    cmax = model.new_int_var(0, horizon, "cmax")
    for n in nodes:
        model.add(cmax >= end[n])
    model.minimize(cmax)

    if heft_sched is not None:  # warm start
        for k, lst in heft_sched.items():
            for (t, st, _) in lst:
                model.add_hint(s[t], min(horizon, int(math.floor(st * SCALE))))
                for j in range(m):
                    model.add_hint(x[t, j], 1 if j == k else 0)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = num_workers
    solver.parameters.random_seed = seed
    status = solver.solve(model)
    status_name = solver.status_name(status)

    result = {"status": status_name, "wall_time": solver.wall_time, "simple_lb": sb,
              "schedule": heft_sched, "ub": heft_ms, "source": "heft"}
    solver_lb = 0.0
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        solver_lb = solver.best_objective_bound / SCALE
        assignment = {n: next(j for j in range(m) if solver.value(x[n, j])) for n in nodes}
        int_start = {n: solver.value(s[n]) for n in nodes}
        sched = _retime(G, m, assignment, int_start)
        validate_schedule(G, sched)          # real schedule, strict check
        ms = compute_makespan(sched)
        if result["ub"] is None or ms < result["ub"] - 1e-9:
            result.update(schedule=sched, ub=ms, source="cpsat")
    elif result["ub"] is None:
        raise RuntimeError(f"CP-SAT returned {status_name} and no HEFT fallback schedule exists.")

    lb = max(solver_lb, sb["simple_lb"])
    if lb > result["ub"] + 1e-6:  # impossible unless there is a bug: LB must never exceed a real schedule
        raise RuntimeError(f"Lower bound {lb} exceeds a real schedule's makespan {result['ub']}.")
    lb = min(lb, result["ub"])    # only removes float noise (<= 1e-6)
    result.update(lb=lb, solver_lb=solver_lb, gap=(result["ub"] - lb) / result["ub"] if result["ub"] > 0 else 0.0)
    return result
