import argparse
import json


def compare_benchmarks(file_a: str, file_b: str):
    with open(file_a, "r") as f:
        data_a = json.load(f)
    with open(file_b, "r") as f:
        data_b = json.load(f)

    branch_a = data_a.get("branch", "Branch A")
    branch_b = data_b.get("branch", "Branch B")

    results_a = data_a.get("results", {})
    results_b = data_b.get("results", {})

    all_keys = list(dict.fromkeys(list(results_a.keys()) + list(results_b.keys())))

    print("\n" + "=" * 105)
    print(f" API Latency Comparison: {branch_a} vs {branch_b}")
    print("=" * 105)
    header = f"{'Endpoint / Action':<35} | {branch_a + ' (Mean)':<16} | {branch_b + ' (Mean)':<16} | {'Delta (ms)':>12} | {'Delta (%)':>10}"
    print(header)
    print("-" * len(header))

    for key in all_keys:
        a_metrics = results_a.get(key)
        b_metrics = results_b.get(key)

        if not a_metrics:
            print(f"{key:<35} | {'N/A':<16} | {b_metrics['mean_ms']:>8.2f}ms     | {'NEW':>12} | {'-':>10}")
            continue
        if not b_metrics:
            print(f"{key:<35} | {a_metrics['mean_ms']:>8.2f}ms     | {'N/A':<16} | {'REMOVED':>12} | {'-':>10}")
            continue

        mean_a = a_metrics["mean_ms"]
        mean_b = b_metrics["mean_ms"]
        delta_ms = mean_b - mean_a
        delta_pct = ((mean_b - mean_a) / mean_a * 100.0) if mean_a > 0 else 0.0

        sign = "+" if delta_ms > 0 else ""
        print(
            f"{key:<35} | {mean_a:>8.2f}ms     | {mean_b:>8.2f}ms     | {sign + f'{delta_ms:.2f}ms':>12} | {sign + f'{delta_pct:.1f}%':>10}"
        )

    print("=" * 105 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two benchmark result JSONs")
    parser.add_argument("file_a", help="Path to base branch benchmark JSON (e.g. master)")
    parser.add_argument("file_b", help="Path to target branch benchmark JSON (e.g. feature/durable-generations)")
    args = parser.parse_args()

    compare_benchmarks(args.file_a, args.file_b)
