"""
algorithms/heft.py - Heterogeneous Earliest Finish Time (HEFT)

Reference: Topcuoglu, Hariri & Wu (2002). "Performance-Effective and 
Low-Complexity Task Scheduling for Heterogeneous Computing."

HEFT is a two-phase algorithm:

PHASE 1 - RANKING (Upward Rank):
  For every task n, the upward rank is defined recursively as:
    rank_u(n) = avg_exec(n) + max over all successors s of:
                  (avg_comm(n, s) + rank_u(s))
  For exit task:
    rank_u(exit) = avg_exec(exit)

  Intuition: rank_u measures the "weight" of the critical path from 
  task n all the way to the exit node. Tasks with high rank sit on the 
  longest path, so they must be scheduled first.

PHASE 2 - PROCESSOR SELECTION (Insertion-based):
  Tasks are sorted in descending order of rank_u.
  For each task (in rank order), it is assigned to the machine that 
  gives the EARLIEST FINISH TIME (EFT), considering idle slot insertion.

This is the gold-standard baseline for Rm|prec|Cmax scheduling.
"""

import networkx as nx
try:
    from .base import (
        earliest_predecessor_finish,
        earliest_free_slot_on_machine,
        compute_makespan,
        validate_schedule,
    )
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from algorithms.base import (
        earliest_predecessor_finish,
        earliest_free_slot_on_machine,
        compute_makespan,
        validate_schedule,
    )


def _compute_avg_exec(G: nx.DiGraph, num_machines: int) -> dict:
    """Average execution time of each task across all machines."""
    avg = {}
    for node, data in G.nodes(data=True):
        p_ij = data.get("p_ij", [1.0] * num_machines)
        avg[node] = sum(p_ij) / len(p_ij)
    return avg


def _compute_upward_rank(G: nx.DiGraph, avg_exec: dict) -> dict:
    """
    Computes the upward rank for every task via reverse topological traversal.
    We ignore communication costs (set comm=0) since our Rm model focuses on 
    computation only.
    """
    rank = {}
    # Process nodes in REVERSE topological order (from exit node back to entry)
    for node in reversed(list(nx.topological_sort(G))):
        if G.out_degree(node) == 0:
            # Exit node: rank = just its own average execution time
            rank[node] = avg_exec[node]
        else:
            # rank_u(n) = avg_exec(n) + max(rank_u(successor)) over all successors
            rank[node] = avg_exec[node] + max(rank[s] for s in G.successors(node))
    return rank


def heft(G: nx.DiGraph, num_machines: int) -> tuple[dict, float]:
    """
    Runs the HEFT algorithm on DAG G with the given number of machines.

    Args:
        G:            NetworkX DiGraph. Each node must have a 'p_ij' attribute
                      (list of exec times per machine, length == num_machines).
        num_machines: Number of unrelated machines (m).

    Returns:
        schedule:  dict {machine_id: [(task_id, start_time, finish_time), ...]}
        makespan:  float - the minimized Cmax.
    """
    # -------------------------------------------------------------------------
    # PHASE 1: Compute upward rank for every task
    # -------------------------------------------------------------------------
    avg_exec = _compute_avg_exec(G, num_machines)
    rank_u   = _compute_upward_rank(G, avg_exec)

    # Sort tasks by descending upward rank - highest rank gets scheduled first.
    # Equal ranks are common (e.g. TGFF tasks of the same type), so ties are broken
    # deterministically by the node's position in G. A lexicographical topological
    # sort is used so that, even when a zero-time task gives a parent and its child
    # the SAME rank, the parent is still scheduled before its children.
    node_index = {n: i for i, n in enumerate(G.nodes())}
    task_priority = list(nx.lexicographical_topological_sort(
        G, key=lambda n: (-rank_u[n], node_index[n])))

    # -------------------------------------------------------------------------
    # PHASE 2: Assign each task (in priority order) to the machine that gives
    #          the Earliest Finish Time (EFT)
    # -------------------------------------------------------------------------
    schedule    = {m: [] for m in range(num_machines)}
    task_finish = {}   # task_id -> finish_time (for precedence checks)

    for task in task_priority:
        p_ij = G.nodes[task]["p_ij"]

        # Earliest time this task is "allowed" to start (all predecessors done)
        ready_time = earliest_predecessor_finish(task, G, task_finish)

        best_machine = None
        best_eft     = float("inf")
        best_start   = None

        for machine in range(num_machines):
            duration = p_ij[machine]
            # Find the earliest gap on this machine that can fit the task
            start = earliest_free_slot_on_machine(
                schedule[machine], ready_time, duration
            )
            eft = start + duration

            if eft < best_eft:
                best_eft     = eft
                best_machine = machine
                best_start   = start

        # Commit the assignment
        schedule[best_machine].append((task, best_start, best_eft))
        task_finish[task] = best_eft

    # Validate that no precedence constraint was violated
    validate_schedule(G, schedule)

    return schedule, compute_makespan(schedule)


# =============================================================================
# Self-test
# =============================================================================
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from data.tgff_pipeline import generate_tgff_dag

    print("=== HEFT Self-Test ===")
    G = generate_tgff_dag(seed=42, num_tasks=10, num_machines=4, out_prefix="data/heft_test")
    schedule, cmax = heft(G, num_machines=4)

    print(f"Makespan (Cmax): {cmax:.3f}")
    print("\nSchedule:")
    for machine, tasks in schedule.items():
        tasks_sorted = sorted(tasks, key=lambda x: x[1])
        for task_id, s, f in tasks_sorted:
            print(f"  Machine {machine}: Task {task_id:8s} | Start={s:7.3f}  Finish={f:7.3f}")
