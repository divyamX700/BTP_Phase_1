"""
algorithms/list_scheduling.py - HEFT's machine-selection phase as a reusable function.

schedule_from_order(G, m, order) takes a ready-made priority list of tasks (every task
after all its predecessors) and places the tasks one by one, each on the machine where
it would FINISH earliest (wall-clock finish = start + running time), using idle-gap
insertion. This is exactly HEFT's second phase; HEFT itself is this function applied to
the list sorted by upward rank. It lets us ask: "given a different order, how good is
HEFT's machine rule?"
"""

import networkx as nx

try:
    from .base import (earliest_predecessor_finish, earliest_free_slot_on_machine,
                       compute_makespan, validate_schedule)
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from algorithms.base import (earliest_predecessor_finish, earliest_free_slot_on_machine,
                                 compute_makespan, validate_schedule)


def schedule_from_order(G: nx.DiGraph, num_machines: int, order: list, validate: bool = False):
    schedule = {k: [] for k in range(num_machines)}
    finish = {}
    for task in order:
        p_ij = G.nodes[task]["p_ij"]
        ready = earliest_predecessor_finish(task, G, finish)   # raises if a parent is not yet placed
        best = None
        for k in range(num_machines):
            start = earliest_free_slot_on_machine(schedule[k], ready, p_ij[k])
            eft = start + p_ij[k]
            if best is None or eft < best[0]:
                best = (eft, start, k)
        eft, start, k = best
        schedule[k].append((task, start, eft))
        finish[task] = eft
    if validate:
        validate_schedule(G, schedule)
    return schedule, compute_makespan(schedule)


def order_from_keys(G: nx.DiGraph, keys: dict) -> list:
    """Valid priority list: smallest key first, but never a task before its predecessors."""
    idx = {n: i for i, n in enumerate(G.nodes())}
    return list(nx.lexicographical_topological_sort(G, key=lambda n: (keys[n], idx[n])))
