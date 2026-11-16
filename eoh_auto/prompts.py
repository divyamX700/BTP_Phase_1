"""
eoh_auto/prompts.py - prompts of Evolution of Heuristics (Liu et al., ICML 2024), adapted to our scoring function.
The wording of the five strategies follows the prompt templates of the official EoH code (github.com/FeiLiu36/EoH);
only the task description is ours. Response format, as in EoH: the idea ("thought") in braces, then the code.
"""

TASK = """Task: schedule a directed acyclic graph of tasks on m unrelated machines to minimise the makespan (the finishing time of the last task). Task t takes time p[t][k] on machine k; a task may start only after all its predecessors have finished; a machine runs one task at a time; there are no communication costs.
A fixed list scheduler builds the schedule: at every step, for every ready task t (all predecessors placed) and every machine k it calls score(f) with the features of the pair (t, k); the pair with the HIGHEST score is placed at its earliest start on k (idle gaps between placed tasks may be used); this repeats until all tasks are placed. Your job is to design the function score.

def score(f) -> float
f is a dict of floats:
  task features (static): p_mean, p_min, p_med, p_max, p_std (task's time over the machines: mean, min, median, max, std dev);
    rank_mean, rank_min, rank_med (longest chain of work from the task to the end of the graph, own time included, with tasks counted at mean / min / median time);
    desc (number of descendants), nsucc (direct successors), npred (direct predecessors), depth (edges on the longest path from a start task)
  pair features: p (time of t on k), est (earliest start of t on k), eft (= est + p), ready (time when all predecessors of t are done),
    mfree (time when machine k finishes its last placed task), eft_best, eft_2nd (best and second-best eft of t over all machines), p_rel (= p / p_min)
  global: n (tasks), m (machines), n_ready (ready tasks now), done (tasks already placed)
f is a FLAT dict: read a feature with f["name"], for example f["eft"] or f["rank_mean"]. The group names above (task features, pair features, global) are only headings, not keys; there are no nested dicts.
The function must be fast and must use only Python built-ins and the module math (already imported). It must return a finite float."""

FORMAT = ("First, describe your new heuristic and main steps in one sentence. The description must be inside a brace {}. "
          "Next, implement it in Python as a function named score that takes the single argument f and returns a float. "
          "Do not give additional explanations.")


def _show(inds):
    return "\n".join(f"No.{i + 1} heuristic and the corresponding code are:\n{{{x['thought']}}}\n```python\n{x['code']}\n```" for i, x in enumerate(inds))


def initial():
    return f"{TASK}\n\nPlease design a new heuristic.\n{FORMAT}"


def e1(parents):
    return (f"{TASK}\n\nI have {len(parents)} existing heuristics with their codes as follows:\n{_show(parents)}\n\n"
            f"Please help me create a new heuristic that has a totally different form from the given ones.\n{FORMAT}")


def e2(parents):
    return (f"{TASK}\n\nI have {len(parents)} existing heuristics with their codes as follows:\n{_show(parents)}\n\n"
            "Please help me create a new heuristic that has a totally different form from the given ones but can be motivated from them.\n"
            "Firstly, identify the common backbone idea in the provided heuristics. Secondly, based on the backbone idea describe your new "
            f"heuristic in one sentence.\n{FORMAT}")


def m1(parent):
    return (f"{TASK}\n\nI have one heuristic with its code as follows:\n{_show([parent])}\n\n"
            f"Please assist me in creating a new heuristic that has a different form but can be a modified version of the heuristic provided.\n{FORMAT}")


def m2(parent):
    return (f"{TASK}\n\nI have one heuristic with its code as follows:\n{_show([parent])}\n\n"
            "Please identify the main heuristic parameters and assist me in creating a new heuristic that has different parameter settings "
            f"of the score function provided.\n{FORMAT}")


def m3(parent):
    return (f"{TASK}\n\nI have one heuristic with its code as follows:\n{_show([parent])}\n\n"
            "First, you need to identify the main components in the function above. Next, analyze whether any of these components are "
            "redundant or can be overfit to the training instances. Then, based on your analysis, simplify the components to enhance the "
            f"generalization to unseen instances.\n{FORMAT}")


STRATEGIES = {"E1": e1, "E2": e2, "M1": m1, "M2": m2, "M3": m3}
N_PARENTS = {"E1": None, "E2": None, "M1": 1, "M2": 1, "M3": 1}   # None = p parents (EoH: p = 5)
