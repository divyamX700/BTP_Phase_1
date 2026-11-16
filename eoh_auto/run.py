"""
eoh_auto/run.py - automated Evolution of Heuristics (Liu et al., ICML 2024) for the scoring function of the list scheduler.

  initial population: HEFT and DLS (expert heuristics) + N heuristics written by the LLM; the best N are kept
  every generation:   each strategy E1, E2, M1, M2, M3 is used N times; parents are drawn with probability
                      proportional to 1/(r + N), r = rank by fitness (0 = best); offspring are evaluated;
                      population + offspring -> the best N distinct heuristics survive
Everything is logged in runs/<name>/: calls.jsonl (every prompt and response), candidates.jsonl (every heuristic with
its fitness or error), population_gXX.json (checkpoint after each generation). --resume continues from the last checkpoint.

Usage:
  python -m eoh_auto.run --name smoke --provider groq --model openai/gpt-oss-120b --N 3 --gens 1
"""
import argparse, json, os, random, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from eoh_auto import llm, prompts
from eoh_auto.evaluate import Evaluator
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEEDS = [
    ("HEFT: the task with the highest upward rank (mean times) goes first, on the machine where it finishes earliest.",
     'def score(f):\n    return f["rank_mean"] * 1e6 - f["eft"]'),
    ("DLS: trade the length of the chain behind the task plus its mean time against how early it can finish.",
     'def score(f):\n    return f["rank_mean"] + f["p_mean"] - f["eft"]'),
]


def parse(text):
    """Thought = first {...}; code = first python code block (or from 'def score' on). Returns (thought, code) or raises."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, flags=re.S)
    code = m.group(1).strip() if m else None
    if code is None and "def score" in text:
        code = text[text.index("def score"):].strip()
    if not code or "def score" not in code:
        raise ValueError("no code with def score in response")
    pre = text[:m.start()] if m else text
    t = re.search(r"\{(.*?)\}", pre, flags=re.S)
    thought = " ".join((t.group(1) if t else pre).split())[:600]
    return thought, code


def pick(pop, k, rng, N):
    """k distinct parents, probability proportional to 1/(rank + N) (EoH)."""
    idx = list(range(len(pop))); out = []
    for _ in range(min(k, len(pop))):
        w = [1.0 / (r + N) for r in idx]
        j = rng.choices(idx, weights=w)[0]
        out.append(pop[j]); idx.remove(j)
    return out


class Run:
    def __init__(self, a):
        self.a = a
        self.dir = os.path.join(ROOT, "eoh_auto", "runs", a.name); os.makedirs(self.dir, exist_ok=True)
        self.rng = random.Random(a.seed)
        self.ev = Evaluator(procs=a.procs, portfolio=a.portfolio)
        self.seen = {}           # code -> candidate record (avoids re-evaluating identical code)
        self.next_id = 0
        self.tokens = 0
        self._lock = threading.Lock(); self._last = 0.0

    def log(self, fname, rec):
        with open(os.path.join(self.dir, fname), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def ask(self, prompt, tag):
        extra = {"reasoning_effort": self.a.reasoning} if self.a.reasoning else {}
        with self._lock:  # pacing: at most one call start every --interval seconds (keeps under tokens-per-minute limits)
            w = self._last + self.a.interval - time.time()
            if w > 0:
                time.sleep(w)
            self._last = time.time()
        r = llm.chat(self.a.provider, self.a.model, prompt, temperature=self.a.temperature, max_tokens=self.a.max_tokens, extra=extra, http_timeout=self.a.http_timeout)
        self.tokens += (r["prompt_tokens"] or 0) + (r["completion_tokens"] or 0)
        self.log("calls.jsonl", dict(tag=tag, time=time.strftime("%Y-%m-%d %H:%M:%S"), prompt=prompt, **r))
        return r

    def evaluate(self, thought, code, gen, strategy, parents):
        rec = dict(id=self.next_id, gen=gen, strategy=strategy, parents=parents, thought=thought, code=code); self.next_id += 1
        if code in self.seen:
            rec.update(fitness=self.seen[code]["fitness"], error=self.seen[code]["error"], duplicate=True)
        else:
            t0 = time.time(); fit, err, _ = self.ev(code)
            rec.update(fitness=fit, error=err, eval_seconds=round(time.time() - t0, 2))
            self.seen[code] = rec
        self.log("candidates.jsonl", rec)
        status = f"gap {rec['fitness']:.3f}%" if rec["fitness"] is not None else f"FAILED ({rec['error'][:60]})"
        print(f"  #{rec['id']:4d} {strategy:5s} {status:28s} {thought[:90]}", flush=True)
        return rec

    def offspring(self, jobs, gen):
        """jobs = list of (strategy, prompt, parent ids); LLM calls in parallel, evaluation one by one."""
        def call(job):
            s, p, par = job
            try:
                return job, self.ask(p, f"g{gen}-{s}"), None
            except Exception as e:
                return job, None, str(e)
        out = []
        with ThreadPoolExecutor(self.a.workers) as ex:
            for (s, p, par), r, err in ex.map(call, jobs):
                if err:
                    print(f"  LLM call failed ({s}): {err[:150]}", flush=True); continue
                try:
                    thought, code = parse(r["text"])
                except ValueError as e:
                    self.log("candidates.jsonl", dict(id=None, gen=gen, strategy=s, parents=par, error=f"parse: {e}", fitness=None))
                    print(f"  ----- {s:5s} unparseable response", flush=True); continue
                out.append(self.evaluate(thought, code, gen, s, par))
        return [x for x in out if x["fitness"] is not None]

    @staticmethod
    def survivors(cands, N):
        best = {}
        for c in sorted(cands, key=lambda c: c["fitness"]):
            best.setdefault(c["code"], c)
        return sorted(best.values(), key=lambda c: c["fitness"])[:N]

    def save(self, pop, gen):
        json.dump(dict(gen=gen, next_id=self.next_id, tokens=self.tokens, rng=self.rng.getstate(), population=pop),
                  open(os.path.join(self.dir, f"population_g{gen:02d}.json"), "w"), indent=1, default=list)
        print(f"== generation {gen}: best gap {pop[0]['fitness']:.3f}%  |  population "
              + ", ".join(f"{c['fitness']:.2f}" for c in pop) + f"  |  tokens so far {self.tokens}", flush=True)

    def load_checkpoint(self):
        files = sorted(f for f in os.listdir(self.dir) if f.startswith("population_g"))
        if not files:
            return None
        d = json.load(open(os.path.join(self.dir, files[-1])))
        st = d["rng"]; self.rng.setstate((st[0], tuple(st[1]), st[2]))
        self.next_id, self.tokens = d["next_id"], d["tokens"]
        for c in d["population"]:
            self.seen[c["code"]] = c
        return d["gen"], d["population"]

    def main(self):
        a = self.a; N = a.N
        ck = self.load_checkpoint() if a.resume else None
        if ck:
            start, pop = ck[0] + 1, ck[1]
            print(f"resumed from generation {ck[0]}", flush=True)
        else:
            json.dump(vars(a), open(os.path.join(self.dir, "config.json"), "w"), indent=1)
            print(f"run {a.name}: {a.provider} / {a.model}, N={N}, generations={a.gens}, p={a.p}", flush=True)
            print("generation 0 (initial population)", flush=True)
            pop = [self.evaluate(t, c, 0, "seed", []) for t, c in SEEDS]
            pop += self.offspring([("init", prompts.initial(), []) for _ in range(N)], 0)
            pop = self.survivors([c for c in pop if c["fitness"] is not None], N)
            self.save(pop, 0); start = 1
        for gen in range(start, a.gens + 1):
            print(f"generation {gen}", flush=True)
            jobs = []
            for s in a.strategies.split(","):
                for _ in range(N):
                    par = pick(pop, a.p if prompts.N_PARENTS[s] is None else 1, self.rng, N)
                    arg = par if prompts.N_PARENTS[s] is None else par[0]
                    jobs.append((s, prompts.STRATEGIES[s](arg), [c["id"] for c in par]))
            pop = self.survivors(pop + self.offspring(jobs, gen), N)
            self.save(pop, gen)
        print(f"\nbest heuristic (train gap {pop[0]['fitness']:.3f}%):\n{{{pop[0]['thought']}}}\n{pop[0]['code']}", flush=True)
        self.ev.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--provider", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--N", type=int, default=10); ap.add_argument("--gens", type=int, default=20); ap.add_argument("--p", type=int, default=5)
    ap.add_argument("--strategies", default="E1,E2,M1,M2,M3"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4, help="parallel LLM calls")
    ap.add_argument("--procs", type=int, default=8, help="processes for evaluation")
    ap.add_argument("--temperature", type=float, default=1.0); ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--reasoning", default=None, help="reasoning_effort for reasoning models (low/medium/high)")
    ap.add_argument("--interval", type=float, default=0.0, help="minimum seconds between LLM call starts")
    ap.add_argument("--http_timeout", type=float, default=300, help="seconds before a hung LLM request is retried")
    ap.add_argument("--portfolio", action="store_true", help="extension: fitness = gap of Best of 7 + candidate")
    ap.add_argument("--resume", action="store_true")
    Run(ap.parse_args()).main()
