"""
data/heterogeneity.py - controlled machine-time matrices p_ij (after Ali et al. 2000 / Braun et al. 2001).

p_ij = base_i * f_ij with f_ij ~ U(1, r_mach)  (r_mach = 1 means identical machines).
  base_i    : the task's own size (given, e.g. a Pegasus runtime, or drawn U(1, r_task) for TGFF graphs).
  r_mach    : machine heterogeneity (largest possible ratio between two machines for one task).
  consistency:
    "inconsistent"  : factors independent (machine A may be faster for task 1 and slower for task 2);
    "consistent"    : every row sorted, so machine 0 is the fastest for ALL tasks, machine 1 second, ...;
    "semi"          : rows of half of the tasks sorted on the even-numbered machines (a consistent sub-matrix),
                      the rest unsorted.
"""
import numpy as np


def make_p_ij(base, m, r_mach, consistency, rng):
    n = len(base)
    f = rng.uniform(1.0, max(1.0, r_mach), size=(n, m))
    if consistency == "consistent":
        f.sort(axis=1)
    elif consistency == "semi":
        even = list(range(0, m, 2))
        for i in rng.choice(n, size=n // 2, replace=False):
            f[i, even] = np.sort(f[i, even])
    elif consistency != "inconsistent":
        raise ValueError(consistency)
    return (np.asarray(base, dtype=float)[:, None] * f).tolist()
