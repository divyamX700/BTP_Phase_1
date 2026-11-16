"""
eoh_auto/evaluate.py - fitness of a scoring function given as source code.
The code is executed in worker processes (never in the main process), on the training instances of eoh/framework.py,
with the verified list scheduler and strict schedule validation. Fitness = mean gap (%) to the best known makespan.
A candidate that raises an error, returns no valid schedule, or exceeds the time limit gets no fitness.
"""
import math, multiprocessing as mp, os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from eoh import framework as fw

_INST = None
_FN = {}


def _init(split):
    global _INST
    _INST = fw.load(split)


def _compile(code):
    if code not in _FN:
        ns = {"math": math}
        exec(compile(code, "<candidate>", "exec"), ns)
        if not callable(ns.get("score")):
            raise ValueError("no function named score")
        _FN.clear(); _FN[code] = ns["score"]
    return _FN[code]


def _work(args):
    code, idx = args
    try:
        fn = _compile(code)
        def safe(f):
            v = float(fn(f))
            return v if math.isfinite(v) else -math.inf
        return fw.schedule(_INST[idx], safe), None
    except Exception as e:  # any error of the candidate code
        return None, f"{type(e).__name__}: {str(e)[:200]}"


class Evaluator:
    def __init__(self, split="train", procs=8, timeout=120):
        self.split, self.procs, self.timeout = split, procs, timeout
        self.insts = fw.load(split)
        self.best_known = [x["best_known"] for x in self.insts]
        self._new_pool()

    def _new_pool(self):
        self.pool = mp.get_context("spawn").Pool(self.procs, initializer=_init, initargs=(self.split,))

    def __call__(self, code):
        """Returns (fitness or None, error or None, list of makespans or None)."""
        job = self.pool.map_async(_work, [(code, i) for i in range(len(self.insts))], chunksize=4)
        try:
            res = job.get(timeout=self.timeout)
        except mp.TimeoutError:
            self.pool.terminate(); self._new_pool()
            return None, f"timeout after {self.timeout}s", None
        errs = [e for _, e in res if e]
        if errs:
            return None, errs[0], None
        ms = [m for m, _ in res]
        gaps = [(m / b - 1) * 100 for m, b in zip(ms, self.best_known)]
        return sum(gaps) / len(gaps), None, ms

    def close(self):
        self.pool.terminate()
