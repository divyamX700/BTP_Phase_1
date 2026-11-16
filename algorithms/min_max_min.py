"""
algorithms/min_max_min.py - Min-Min and Max-Min Heuristics

These two classic greedy algorithms are often studied together because
they are mirror images of each other.

SETUP (same for both):
  At each step, for every ready task i and every machine j, compute
  the minimum completion time (MCT):
    MCT(i, j) = max(machine_free[j], ready_time(i)) + p_ij[j]
  
  Then for each ready task i, find its minimum MCT across all machines:
    min_completion(i) = min over j of MCT(i, j)

MIN-MIN:
  Among all ready tasks, pick the task with the MINIMUM min_completion.
  Assign it to the machine that gives that minimum completion time.
  
  Philosophy: "Do the easiest/quickest task first."
  Benefit: Frees up machines quickly. Works well when tasks are small and
           numerous - it keeps machines busy constantly.
  Risk: May leave large tasks stuck at the end, causing a long tail.

MAX-MIN:
  Among all ready tasks, pick the task with the MAXIMUM min_completion.
  Assign it to the machine that gives its minimum completion time.
  
  Philosophy: "Get the hardest/longest task started early."
  Benefit: Long tasks get scheduled on their fastest machine immediately.
           Short tasks are scheduled concurrently around them.
  Risk: Can starve quick tasks if there are many slow tasks.

Both algorithms are greedy and make ONE scheduling decision per iteration.
"""

import networkx as nx

try:
    from .base import (
        earliest_predecessor_finish,
        compute_makespan,
        validate_schedule,
    )
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from algorithms.base import (
        earliest_predecessor_finish,
        compute_makespan,
        validate_schedule,
    )


def _min_max_min_core(G: nx.DiGraph, num_machines: int, use_max: bool) -> tuple[dict, float]:
    """
    Core implementation shared by both Min-Min and Max-Min.
    use_max=False => Min-Min (pick task with minimum min_completion)
    use_max=True  => Max-Min (pick task with maximum min_completion)
    """
    schedule      = {m: [] for m in range(num_machines)}
    task_finish   = {}
    machine_free  = {m: 0.0 for m in range(num_machines)}
    scheduled     = set()
    total_tasks   = G.number_of_nodes()

    while len(scheduled) < total_tasks:
        # Build the ready queue
        ready = [
            n for n in G.nodes()
            if n not in scheduled
            and all(pred in scheduled for pred in G.predecessors(n))
        ]

        if not ready:
            raise RuntimeError("No ready task although tasks remain: the graph is not a DAG.")

        # For each ready task, compute its best (minimum) completion time
        # across all machines, and record which machine gives that best time.
        task_best = {}   # task -> (best_machine, best_start, best_eft)
        for task in ready:
            p_ij       = G.nodes[task]["p_ij"]
            ready_time = earliest_predecessor_finish(task, G, task_finish)

            best_m     = None
            best_eft   = float("inf")
            best_start = None

            for machine in range(num_machines):
                start = max(machine_free[machine], ready_time)
                eft   = start + p_ij[machine]
                if eft < best_eft:
                    best_eft   = eft
                    best_m     = machine
                    best_start = start

            task_best[task] = (best_m, best_start, best_eft)

        # Select the task to schedule based on Min-Min or Max-Min criterion
        if use_max:
            # MAX-MIN: pick the task whose minimum completion time is the LARGEST
            chosen = max(ready, key=lambda t: task_best[t][2])
        else:
            # MIN-MIN: pick the task whose minimum completion time is the SMALLEST
            chosen = min(ready, key=lambda t: task_best[t][2])

        best_m, best_start, best_eft = task_best[chosen]

        schedule[best_m].append((chosen, best_start, best_eft))
        task_finish[chosen]   = best_eft
        machine_free[best_m]  = best_eft
        scheduled.add(chosen)

    validate_schedule(G, schedule)
    return schedule, compute_makespan(schedule)


def min_min(G: nx.DiGraph, num_machines: int) -> tuple[dict, float]:
    """Min-Min: always schedule the task with the minimum earliest completion time."""
    return _min_max_min_core(G, num_machines, use_max=False)


def max_min(G: nx.DiGraph, num_machines: int) -> tuple[dict, float]:
    """Max-Min: always schedule the task with the maximum (of minimum) completion times."""
    return _min_max_min_core(G, num_machines, use_max=True)


# =============================================================================
# Self-test
# =============================================================================
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from data.tgff_pipeline import generate_tgff_dag

    G = generate_tgff_dag(seed=42, num_tasks=10, num_machines=4, out_prefix="data/minmax_test")

    print("=== Min-Min Self-Test ===")
    schedule, cmax = min_min(G, num_machines=4)
    print(f"Makespan (Cmax): {cmax:.3f}")
    for machine, tasks in schedule.items():
        for task_id, s, f in sorted(tasks, key=lambda x: x[1]):
            print(f"  Machine {machine}: Task {task_id:8s} | Start={s:7.3f}  Finish={f:7.3f}")

    print("\n=== Max-Min Self-Test ===")
    schedule, cmax = max_min(G, num_machines=4)
    print(f"Makespan (Cmax): {cmax:.3f}")
    for machine, tasks in schedule.items():
        for task_id, s, f in sorted(tasks, key=lambda x: x[1]):
            print(f"  Machine {machine}: Task {task_id:8s} | Start={s:7.3f}  Finish={f:7.3f}")
