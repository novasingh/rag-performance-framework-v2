"""
benchmark_backends.py
=====================
Speed benchmark: Google AI Studio vs Ollama llama3.1:8b
"""
import sys, time, json, statistics, requests
sys.path.insert(0, 'f:/rs2')

from rq1_experiment.config import GOOGLE_AI_API_KEY, RAG_PROMPT_TEMPLATE

TEST_QUERIES = [
    {
        "question": "What are the main benefits of retrieval-augmented generation over pure LLM generation?",
        "context": "Retrieval-Augmented Generation (RAG) combines information retrieval with large language model generation. Retrieved documents provide grounding, reducing hallucination rates. Studies show RAG systems reduce factual errors by 28% compared to pure LLM systems. The key advantage is that knowledge can be updated without retraining the model."
    },
    {
        "question": "How does dataset freshness affect information retrieval precision?",
        "context": "Dataset freshness refers to how recently documents were published relative to the query time. Fresher documents have higher relevance for time-sensitive queries. Research shows retrieval precision drops by 15-20% when documents are older than 6 months for high-volatility domains like technology."
    },
    {
        "question": "What is the role of source type diversity in RAG system performance?",
        "context": "Source type diversity refers to mixing academic papers, news articles, and technical documentation in a retrieval corpus. Systems using diverse sources show 12% higher response accuracy compared to single-source configurations."
    },
]

def build_prompt(q):
    return RAG_PROMPT_TEMPLATE.format(context=q["context"], question=q["question"])

def benchmark_google(n=2):
    print("\n=== GOOGLE AI STUDIO (gemma-4-26b-a4b-it) ===")
    from google import genai
    from google.genai import types as genai_types
    client = genai.Client(api_key=GOOGLE_AI_API_KEY)
    latencies = []
    errors = 0
    for i, q in enumerate(TEST_QUERIES[:n]):
        print(f"  Query {i+1}/{n} ... ", end="", flush=True)
        try:
            t0 = time.perf_counter()
            resp = client.models.generate_content(
                model="models/gemma-4-26b-a4b-it",
                contents=build_prompt(q),
                config=genai_types.GenerateContentConfig(temperature=0.0, max_output_tokens=256),
            )
            lat = time.perf_counter() - t0
            latencies.append(lat)
            text = (resp.text or "").strip()
            print(f"{lat:.1f}s | {len(text)} chars | OK")
        except Exception as e:
            errors += 1
            print(f"ERROR: {str(e)[:80]}")
    if latencies:
        print(f"  -> Mean: {statistics.mean(latencies):.1f}s | Errors: {errors}/{n}")
    return {"backend": "google/gemma-4-26b-a4b-it", "latencies": latencies, "errors": errors,
            "mean_s": statistics.mean(latencies) if latencies else 9999}

def benchmark_ollama(model="llama3.1:8b", n=3):
    print(f"\n=== OLLAMA ({model}) ===")
    latencies = []
    errors = 0
    for i, q in enumerate(TEST_QUERIES[:n]):
        print(f"  Query {i+1}/{n} ... ", end="", flush=True)
        try:
            t0 = time.perf_counter()
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "prompt": build_prompt(q), "stream": False,
                      "options": {"temperature": 0.0, "num_predict": 256}},
                timeout=120,
            )
            lat = time.perf_counter() - t0
            if resp.status_code == 200:
                data = resp.json()
                text = (data.get("response") or "").strip()
                tps  = (data.get("eval_count", 0) / max(data.get("eval_duration", 1), 1)) * 1e9
                latencies.append(lat)
                print(f"{lat:.1f}s | {len(text)} chars | {tps:.1f} tok/s | OK")
            else:
                errors += 1
                print(f"HTTP {resp.status_code} (ignored - likely cold start)")
        except Exception as e:
            errors += 1
            print(f"ERROR: {str(e)[:80]}")
    if latencies:
        print(f"  -> Mean: {statistics.mean(latencies):.1f}s | Errors: {errors}/{n}")
    return {"backend": f"ollama/{model}", "latencies": latencies, "errors": errors,
            "mean_s": statistics.mean(latencies) if latencies else 9999}

print("Starting backend benchmark...")
print("=" * 60)

r_ollama = benchmark_ollama("llama3.1:8b", n=3)
r_google = benchmark_google(n=2)

print("\n" + "=" * 60)
print("RESULTS:")
for r in [r_ollama, r_google]:
    mean = r['mean_s']
    ok = len(r['latencies'])
    n_total = ok + r['errors']
    mean_str = f"{mean:.1f}s" if mean < 9000 else "N/A"
    print(f"  {r['backend']:40s} mean={mean_str:8s} success={ok}/{n_total}")

valid = [r for r in [r_ollama, r_google] if r['latencies']]
if len(valid) == 2:
    ratio = r_google['mean_s'] / r_ollama['mean_s'] if r_ollama['mean_s'] > 0 else 0
    print(f"\n  Speed ratio: Ollama is {ratio:.1f}x faster than Google AI")

winner = min(valid, key=lambda r: r['mean_s']) if valid else None
if winner:
    print(f"\nWINNER: {winner['backend']} ({winner['mean_s']:.1f}s avg)")
    backend = "ollama" if "ollama" in winner["backend"] else "google"
    print(f"USE_BACKEND={backend}")
