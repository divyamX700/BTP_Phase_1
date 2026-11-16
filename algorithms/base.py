"""
algorithms/base.py - Shared Utilities for all scheduling algorithms.

A Schedule is a dictionary:
  schedule[machine_id] = list of (task_id, start_time, finish_time)

All algorithms must:
  1. Accept a NetworkX DiGraph (G) and num_machines (int).
  2. Return a Schedule dict and the makespan (float).
  3. Respect DAG precedence: a task can only start after ALL its predecessors finish.
"""

import networkx as nx


def compute_makespan(schedule: dict) -> float:
    """
    Given a schedule dict, computes Cmax (the maximum finish time across all machines).
    This is the metric we are minimizing throughout this entire project.
    """
    cmax = 0.0
    for machine_tasks in schedule.values():
        for _, _, finish_time in machine_tasks:
            if finish_time > cmax:
                cmax = finish_time
    return cmax


def earliest_predecessor_finish(task: str, G: nx.DiGraph, task_finish: dict) -> float:
    """
    Computes the earliest time a task is "ready" to start.
    A task is ready only when ALL its predecessor (parent) tasks have finished.
    This directly enforces the 'prec' (precedence) constraint in Rm|prec|Cmax.
    
    task_finish: dict mapping task_id -> finish_time of that task.
    """
    if G.in_degree(task) == 0:
        return 0.0  # Entry node — no predecessors, ready at time 0
    for pred in G.predecessors(task):
        if pred not in task_finish:
            raise RuntimeError(
                f"Task {task} was considered before its predecessor {pred} was scheduled."
            )
    return max(task_finish[pred] for pred in G.predecessors(task))


def earliest_free_slot_on_machine(machine_tasks: list, ready_time: float, duration: float) -> float:
    """
    Finds the earliest time slot on a given machine where a task of 'duration'
    can fit, given the machine's currently scheduled tasks and that the task
    cannot start before 'ready_time' (the precedence constraint).
    
    This implements insertion-based scheduling (looking for gaps).
    """
    # Build a sorted list of intervals already assigned on this machine
    intervals = sorted((s, f) for _, s, f in machine_tasks)
    
    # Try to fit before the first task or between consecutive tasks
    candidate_start = ready_time
    for (s, f) in intervals:
        if candidate_start + duration <= s:
            # Task fits in the gap before this interval starts
            return candidate_start
        # Otherwise push start time to after this interval
        candidate_start = max(candidate_start, f)
    
    # No gap found — schedule after the last task
    return candidate_start


def validate_schedule(G: nx.DiGraph, schedule: dict) -> bool:
    """
    Strict correctness checker for a schedule of the Rm|prec|Cmax problem.
    Returns True if valid; raises AssertionError on the first violated rule:
      1. every task of G is scheduled exactly once (and nothing else is);
      2. each task runs for exactly p_ij on the machine it was given, starting at t >= 0;
      3. no two tasks overlap on the same machine;
      4. precedence: a task starts only after ALL its predecessors have finished.
    """
    tol = 1e-9
    task_times = {}   # task_id -> (start_time, finish_time)

    for machine, machine_tasks in schedule.items():
        for task_id, start, finish in machine_tasks:
            assert task_id in G, f"UNKNOWN TASK {task_id} in schedule!"
            assert task_id not in task_times, f"TASK {task_id} is scheduled more than once!"
            task_times[task_id] = (start, finish)

            expected = G.nodes[task_id]["p_ij"][machine]
            assert start >= -tol, f"Task {task_id} starts at negative time {start:.3f}!"
            assert abs((finish - start) - expected) <= tol * max(1.0, expected) + tol, (
                f"DURATION MISMATCH: Task {task_id} on machine {machine} runs "
                f"{finish - start:.6f} but p_ij = {expected:.6f}!"
            )

        intervals = sorted((s, f, t) for t, s, f in machine_tasks)
        for (s1, f1, t1), (s2, f2, t2) in zip(intervals, intervals[1:]):
            assert s2 >= f1 - tol, (
                f"MACHINE OVERLAP on machine {machine}: task {t1} [{s1:.3f}, {f1:.3f}] "
                f"and task {t2} [{s2:.3f}, {f2:.3f}]!"
            )

    missing = set(G.nodes()) - set(task_times)
    assert not missing, f"UNSCHEDULED TASKS: {sorted(map(str, missing))[:5]}..."

    for u, v in G.edges():
        u_finish = task_times[u][1]
        v_start  = task_times[v][0]
        assert v_start >= u_finish - tol, (
            f"PRECEDENCE VIOLATION: Task {v} starts at {v_start:.3f} "
            f"but predecessor {u} finishes at {u_finish:.3f}!"
        )
    return True
