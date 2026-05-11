import argparse
import json
import statistics
from pathlib import Path


def load_trace(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_stage_map(trace: dict) -> dict[str, float]:
    stages = trace.get("stages") or []
    result: dict[str, float] = {}
    for stage in stages:
        name = stage.get("name")
        elapsed = stage.get("elapsed_secs")
        if not isinstance(name, str) or not isinstance(elapsed, (int, float)):
            continue
        result[name] = float(elapsed)
    return result


def fmt_secs(value: float) -> str:
    return f"{value:8.3f}s"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize Sub-Zero runtime trace sidecars (*.sub-zero.trace.json)."
    )
    parser.add_argument("traces", nargs="+", help="Trace json files")
    args = parser.parse_args()

    trace_paths = [Path(p) for p in args.traces]
    traces = [load_trace(p) for p in trace_paths]

    totals = []
    stage_values: dict[str, list[float]] = {}
    stage_order: list[str] = []

    for trace in traces:
        total = trace.get("total_elapsed_secs")
        if isinstance(total, (int, float)):
            totals.append(float(total))

        stage_map = get_stage_map(trace)
        for name, value in stage_map.items():
            if name not in stage_values:
                stage_values[name] = []
                stage_order.append(name)
            stage_values[name].append(value)

    def summarize(values: list[float]) -> tuple[float, float, float]:
        values = list(values)
        if not values:
            return (0.0, 0.0, 0.0)
        mean = statistics.fmean(values)
        p50 = statistics.median(values)
        worst = max(values)
        return (mean, p50, worst)

    print(f"traces: {len(traces)}")
    if totals:
        mean, p50, worst = summarize(totals)
        print(f"total: mean={fmt_secs(mean)} p50={fmt_secs(p50)} worst={fmt_secs(worst)}")
    else:
        print("total: (missing total_elapsed_secs)")
    print("")

    header = f"{'stage':28} {'mean':>10} {'p50':>10} {'worst':>10} {'n':>4}"
    print(header)
    print("-" * len(header))

    for name in stage_order:
        values = stage_values.get(name, [])
        mean, p50, worst = summarize(values)
        print(
            f"{name:28} {fmt_secs(mean):>10} {fmt_secs(p50):>10} {fmt_secs(worst):>10} {len(values):4d}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

