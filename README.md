# Task Graph Scheduling on Unrelated Machines

HEFT, Portfolios and LLM-Evolved Heuristics

B.Tech project by Divyam Kulshrestha, Department of Computer Science and Engineering, IIT Guwahati.
Supervisor: Prof. Aryabartta Sahu.

The project studies the scheduling problem `R_m | prec | C_max`: a graph of dependent tasks is run on machines whose speed
depends on the task, and the goal is the shortest finish time (makespan) of the whole graph. Communication costs are not modelled.

The work has three parts:

1. Eleven list-scheduling heuristics, including HEFT and DLS, are measured against an exact solver (OR-Tools CP-SAT) on
   814 instances (610 made from TGFF task graphs, 204 from 17 Pegasus workflows). HEFT is on average 5.21% above the best known
   schedule, and its loss comes from its task order. A portfolio that keeps the shortest of seven schedules reduces the gap to 1.98%.
2. Adversarial search on small instances finds a case where HEFT needs 2.71 times the optimal time.
3. Evolution of Heuristics (EoH) lets a small LLM (NVIDIA Nemotron-3.5-Bolt-e4b) write the scoring function of a list scheduler.
   On 653 unseen instances the evolved function lowers HEFT's mean gap from 5.11% to 3.74%.

## Contents

```
report/        final report (PDF)
slides/        final presentation (PDF and PPTX)
algorithms/    HEFT, its rank variants, DLS, Sufferage, Min-Min, Max-Min, the exact CP-SAT solver
data/          Pegasus workflow parser, TGFF graph generation, machine-time (heterogeneity) generation
experiments/   instance suite and all methods, first failure study, adversarial search, reproducibility check
eoh/           training and test split and the list scheduler driven by a scoring function
eoh_auto/      the automated EoH search, replay of a logged run, evaluation on the test set, and the logs of the two reported runs
results/       raw result files that the report is built from
WF/            the 17 Pegasus workflow files used (from the Pegasus workflow benchmark)
tests/         check of the exact solver against brute force
```

## Setup

Python 3.10 or newer.

```
pip install -r requirements.txt
```

Random task graphs are made with TGFF 3.5 ("Task Graphs For Free", Dick, Rhodes and Wolf, CODES 1998). It is not part of this
repository. Download it, build it in a folder called `tgff-3.5/` in the project root, so that `tgff-3.5/tgff` exists.
The code calls it through WSL (`wsl -d Ubuntu -e ./tgff-3.5/tgff ...`); on Linux, remove the `wsl -d Ubuntu -e` prefix in
`data/tgff_graphs.py` and `data/tgff_pipeline.py`.

The stored results in `results/` and the EoH logs in `eoh_auto/runs/` are enough to inspect everything without TGFF.
TGFF is needed only to rebuild the instances (the suite, the first study and the EoH evaluation).

## Running

All commands are run from the project root.

Check of the exact solver (no TGFF needed):

```
python tests/test_exact_vs_bruteforce.py
```

Instance suite with all methods. Each call writes one batch to `results/`; the report uses four batches
(`step7_suite.csv` is the training batch of the EoH search, the other three are the test batches):

```
python -u experiments/step7_variety_suite.py step7_suite.csv 1,2 2026
python -u experiments/step7_variety_suite.py step7_suite_b.csv 3,4,5 11
python -u experiments/step7_variety_suite.py step7_suite_c.csv 6,7,8 12
python -u experiments/step7_variety_suite.py step7_suite_d.csv 9,10,11 13
```

First study of why HEFT loses (68 instances, task-order search), reproducibility check, and adversarial search:

```
python -u experiments/step3_failure_analysis.py
python experiments/check_reproducibility.py
python experiments/step5_adversarial_search.py HEFT 12 3 600 0
```

The arguments of the adversarial search are the target method (`HEFT`, `DLS`, `RANKMIN`, `BESTOF4`, `SUFFERAGE`, `DLSPAPER`,
`PORTFOLIO7`), the number of tasks, the number of machines, the time in seconds and the random seed. The stored searches are in
`results/step5_adv_*.json`.

## Evolution of Heuristics

The search that is reported was run with

```
python -m eoh_auto.run --name nvidia_bolt_e4b_s0 --provider nvidia --model nvidia/nemotron-3.5-bolt-e4b --N 10 --gens 20 --seed 0 --workers 6 --procs 8 --max_tokens 2048 --interval 2
```

It needs an API key for NVIDIA's free API in a file `.nvidia_key` in the project root (not included, and ignored by git).
It makes about 1,000 LLM calls. The logs of the two runs described in the report (seeds 0 and 1) are in `eoh_auto/runs/`, so the
search does not have to be repeated:

```
python -m eoh_auto.replay nvidia_bolt_e4b_s0
python -m eoh_auto.test_best nvidia_bolt_e4b_s0 1
```

`replay` re-evaluates every logged scoring function on the 161 training instances without calling the LLM and checks that the
fitness equals the logged one. `test_best` evaluates the final function on the 653 test instances and compares it with HEFT,
DLS and Best of 7. For run `nvidia_bolt_e4b_s0` it prints the evolved score

```
score(t, k) = rank_mean(t) + p_min(t) - eft(t, k)
```

with a mean gap of 3.74% against 5.11% for HEFT on the test instances.

## References

- H. Topcuoglu, S. Hariri, M.-Y. Wu, "Performance-effective and low-complexity task scheduling for heterogeneous computing", IEEE TPDS, 2002 (HEFT).
- F. Liu et al., "Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Model", ICML 2024 (EoH).
- R. P. Dick, D. L. Rhodes, W. Wolf, "TGFF: Task Graphs For Free", CODES 1998.
- The full list is in the report.
