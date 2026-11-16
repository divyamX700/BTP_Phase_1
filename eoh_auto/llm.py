"""
eoh_auto/llm.py - minimal client for OpenAI-compatible chat APIs (NVIDIA NIM, Groq, or a local server).
Keys are read from files in the project root (.nvidia_key, .groq_key), which are excluded from git.
Rate-limit (HTTP 429) and server errors (5xx) are retried with waiting; other errors are raised.
"""
import json, os, time, urllib.request, urllib.error

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROVIDERS = {
    "nvidia": {"url": "https://integrate.api.nvidia.com/v1", "key_file": ".nvidia_key"},
    "groq": {"url": "https://api.groq.com/openai/v1", "key_file": ".groq_key"},
}


def _key(provider):
    return open(os.path.join(ROOT, PROVIDERS[provider]["key_file"]), encoding="utf-8").read().strip()


def _request(provider, path, body=None, timeout=300):
    req = urllib.request.Request(PROVIDERS[provider]["url"] + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + _key(provider), "Content-Type": "application/json",
                                          "User-Agent": "eoh-dag-scheduling/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def list_models(provider):
    return sorted(m["id"] for m in _request(provider, "/models")["data"])


def chat(provider, model, prompt, temperature=1.0, max_tokens=4096, retries=30, extra=None, http_timeout=300):
    """Returns dict(text, prompt_tokens, completion_tokens, seconds, provider, model)."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": temperature, "max_tokens": max_tokens}
    body.update(extra or {})
    wait = 10
    for attempt in range(retries):
        t0 = time.time()
        try:
            r = _request(provider, "/chat/completions", body, timeout=http_timeout)
            u = r.get("usage", {}) or {}
            return dict(text=r["choices"][0]["message"].get("content") or "", prompt_tokens=u.get("prompt_tokens"),
                        completion_tokens=u.get("completion_tokens"), seconds=round(time.time() - t0, 2), provider=provider, model=model)
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:300]
            if e.code == 429 or e.code >= 500:
                ra = e.headers.get("retry-after")
                w = float(ra) if ra and ra.replace(".", "").isdigit() else wait
                print(f"  [{provider}] HTTP {e.code}, waiting {w:.0f}s (attempt {attempt + 1}/{retries}): {msg[:120]}", flush=True)
                time.sleep(w); wait = min(wait * 2, 120)
                continue
            raise RuntimeError(f"{provider} HTTP {e.code}: {msg}")
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"  [{provider}] network error {e}, waiting {wait}s", flush=True)
            time.sleep(wait); wait = min(wait * 2, 120)
    raise RuntimeError(f"{provider}: gave up after {retries} attempts")
