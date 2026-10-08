"""Profiler script for measuring RAG pipeline latency percentiles.

Runs N sequential queries against a running API instance and reports
P50, P95, P99, mean, and RPS throughput.

Usage
-----
    python -m benchmarks.profiler --url http://localhost:8000 --requests 100
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request

SAMPLE_QUESTIONS = [
    "ما هي شروط صحة عقد البيع في القانون المدني المصري؟",
    "ما حكم القوة القاهرة في المسئولية المدنية؟",
    "ما هي أحكام فسخ العقد وفقاً للمادة 157؟",
    "ما هو سن الرشد في القانون المدني المصري؟",
    "ما هي شروط الأهلية في التعاقد؟",
    "ما حكم الغلط في العقود المدنية؟",
    "ما هي أحكام التعويض عن الضرر الأدبي؟",
    "ما هي مدة التقادم في الدعاوى المدنية؟",
    "ما هي شروط صحة الوكالة؟",
    "ما هي أحكام حق الملكية؟",
]


def _percentile(data: list[float], p: float) -> float:
    """Calculate the p-th percentile of a sorted list."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    d = k - f
    return sorted_data[f] + d * (sorted_data[c] - sorted_data[f])


def run_profiler(base_url: str, num_requests: int) -> dict:
    """Run the profiler against the API and return a results dict."""
    latencies: list[float] = []
    errors = 0
    url = f"{base_url}/ask"

    print(f"🔬 Profiling {url} with {num_requests} requests …\n")

    overall_start = time.perf_counter()
    for i in range(num_requests):
        question = SAMPLE_QUESTIONS[i % len(SAMPLE_QUESTIONS)]
        payload = json.dumps({"question": question, "top_k": 3}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp.read()
            latency_ms = (time.perf_counter() - t0) * 1000
            latencies.append(latency_ms)
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            errors += 1
            print(f"  ✗ Request {i + 1} failed: {exc}")

        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{num_requests}")

    total_time = time.perf_counter() - overall_start

    if not latencies:
        print("❌ All requests failed.")
        return {}

    results = {
        "total_requests": num_requests,
        "successful_requests": len(latencies),
        "errors": errors,
        "total_time_s": round(total_time, 2),
        "rps": round(len(latencies) / total_time, 2),
        "mean_ms": round(statistics.mean(latencies), 2),
        "median_ms": round(statistics.median(latencies), 2),
        "p50_ms": round(_percentile(latencies, 50), 2),
        "p95_ms": round(_percentile(latencies, 95), 2),
        "p99_ms": round(_percentile(latencies, 99), 2),
        "min_ms": round(min(latencies), 2),
        "max_ms": round(max(latencies), 2),
    }

    print("\n" + "=" * 60)
    print("📊  PROFILING RESULTS")
    print("=" * 60)
    for key, val in results.items():
        print(f"  {key:.<30} {val}")
    print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(description="Profile the Egyptian Legal RAG API")
    parser.add_argument(
        "--url", default="http://localhost:8000", help="Base URL of the API"
    )
    parser.add_argument(
        "--requests", type=int, default=50, help="Number of requests to send"
    )
    parser.add_argument(
        "--output", default=None, help="Optional JSON file to save results"
    )
    args = parser.parse_args()

    results = run_profiler(args.url, args.requests)

    if args.output and results:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {args.output}")

    sys.exit(0 if results else 1)


if __name__ == "__main__":
    main()
