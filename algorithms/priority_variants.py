"""
algorithms/priority_variants.py - Candidate alternatives to HEFT's average-time priority list.
All for Rm|prec|Cmax (zero communication). Every function takes (G, num_machines) and returns
(schedule, makespan), and the returned schedule is checked by the strict validator.

STATIC LIST + HEFT PLACEMENT (placement = earliest finish time, with idle-gap insertion)
  heft_variant(G, m, weight)   priority = longest chain to the exit, where each task's length is
                               its mean / median / min / max time over machines. "mean" is HEFT.
  best_of_ranks(G, m)          runs the four weights above and keeps the shortest schedule
                               (the idea suggested by Zhao & Sakellariou 2003; costs 4 runs).

DYNAMIC (decide the next task AND its machine at every step, among the tasks whose parents are done)
  sufferage_dag(G, m)          Sufferage (Maheswaran et al. 1999) adapted to a DAG: for each ready task,
                               sufferage = (2nd-best earliest finish time) - (best earliest finish time)
                               over machines. Schedule the ready task with the LARGEST sufferage on its
                               best machine (placement with insertion). Ties: larger min-weight rank.
  dls_zero_comm(G, m)          Dynamic Level Scheduling (Sih & Lee 1991), heterogeneous form, with zero
                               communication: DL(t, k) = SL*(t) - start(t, k) + (median(t) - p_tk), where
                               SL* is the longest chain to the exit using median times, start = max(parents
                               done, machine k free). Pick the (ready task, machine) pair with the largest DL.
                               As published, no idle-gap insertion.
"""

import statistics
import networkx as nx

try:
    from .base import (earliest_predecessor_finish, earliest_free_slot_on_machine,
                       compute_makespan, validate_schedule)
    from .list_scheduling import schedule_from_order, order_from_keys
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from algorithms.base import (earliest_predecessor_finish, earliest_free_slot_on_machine,
                                 compute_makespan, validate_schedule)
    from algorithms.list_scheduling import schedule_from_order, order_from_keys

WEIGHTS = {"mean": statistics.fmean, "median": statistics.median, "min": min, "max": max}


def _task_weights(G, weight):
    f = WEIGHTS[weight]
    return {n: f(G.nodes[n]["p_ij"]) for n in G}


def _upward_rank(G, w):
    """rank(t) = w(t) + max over successors of rank(successor); the exit tasks have rank w(t)."""
    rank = {}
    for n in reversed(list(nx.topological_sort(G))):
        rank[n] = w[n] + max((rank[s] for s in G.successors(n)), default=0.0)
    return rank


def heft_variant(G, num_machines, weight="mean"):
    rank = _upward_rank(G, _task_weights(G, weight))
    order = order_from_keys(G, {n: -rank[n] for n in G})
    return schedule_from_order(G, num_machines, order, validate=True)


def best_of_ranks(G, num_machines):
    best = None
    for weight in ("mean", "median", "min", "max"):
        sched, ms = heft_variant(G, num_machines, weight)
        if best is None or ms < best[1] - 1e-12:
            best = (sched, ms)
    return best


def _options(G, sched, finish, task, m):
    """(finish time, start time, machine) for the task on every machine, with idle-gap insertion."""
    ready = earliest_predecessor_finish(task, G, finish)
    p = G.nodes[task]["p_ij"]
    out = []
    for k in range(m):
        s = earliest_free_slot_on_machine(sched[k], ready, p[k])
        out.append((s + p[k], s, k))
    out.sort()
    return out


def sufferage_dag(G, num_machines):
    m = num_machines
    idx = {n: i for i, n in enumerate(G.nodes())}
    tie_rank = _upward_rank(G, _task_weights(G, "min"))
    sched = {k: [] for k in range(m)}
    finish = {}
    remaining = set(G.nodes())
    while remaining:
        ready = [t for t in remaining if all(p in finish for p in G.predecessors(t))]
        if not ready:
            raise RuntimeError("No ready task although tasks remain: the graph is not a DAG.")
        chosen = None
        for t in ready:
            opts = _options(G, sched, finish, t, m)
            suff = (opts[1][0] - opts[0][0]) if m > 1 else 0.0
            key = (suff, tie_rank[t], -idx[t])
            if chosen is None or key > chosen[0]:
                chosen = (key, t, opts[0])
        _, t, (eft, start, k) = chosen
        sched[k].append((t, start, eft))
        finish[t] = eft
        remaining.remove(t)
    validate_schedule(G, sched)
    return sched, compute_makespan(sched)


def dls_variant(G, num_machines, sl_weight="median", use_delta=True, insertion=False):
    """
    Dynamic-level scheduling with switches, for ablation.
      sl_weight : statistic of a task's machine times used for the static level SL* and for the
                  speed term (paper: "median").
      use_delta : include the speed term (stat(t) - p_tk). Paper: True.
      insertion : allow placing a task in an idle gap (start via earliest_free_slot_on_machine);
                  paper: False (a task starts after the machine's last task).
    DL(t, k) = SL*(t) - start(t, k) + [use_delta] * (stat(t) - p_tk); each step takes the largest DL.
    """
    m = num_machines
    idx = {n: i for i, n in enumerate(G.nodes())}
    stat = _task_weights(G, sl_weight)
    sl = _upward_rank(G, stat)
    sched = {k: [] for k in range(m)}
    free = [0.0] * m
    finish = {}
    remaining = set(G.nodes())
    while remaining:
        ready = [t for t in remaining if all(p in finish for p in G.predecessors(t))]
        if not ready:
            raise RuntimeError("No ready task although tasks remain: the graph is not a DAG.")
        best = None
        for t in ready:
            rt = earliest_predecessor_finish(t, G, finish)
            p = G.nodes[t]["p_ij"]
            for k in range(m):
                start = earliest_free_slot_on_machine(sched[k], rt, p[k]) if insertion else max(rt, free[k])
                dl = sl[t] - start + ((stat[t] - p[k]) if use_delta else 0.0)
                key = (dl, -idx[t], -k)
                if best is None or key > best[0]:
                    best = (key, t, k, start)
        _, t, k, start = best
        end = start + G.nodes[t]["p_ij"][k]
        sched[k].append((t, start, end))
        finish[t] = end
        free[k] = max(free[k], end)
        remaining.remove(t)
    validate_schedule(G, sched)
    return sched, compute_makespan(sched)


def dls_zero_comm(G, num_machines):
    """DLS as published (median weights, speed term, no insertion), zero communication."""
    return dls_variant(G, num_machines, "median", True, False)
