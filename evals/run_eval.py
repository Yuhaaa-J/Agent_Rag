"""独立跑评测：uv run python evals/run_eval.py -k 5

等价于 `uv run agent-rag eval`，单独放一个脚本是为了让它在 CI 里好调用。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent_rag.config import get_settings
from agent_rag.evaluation import evaluate, format_report, load_cases
from agent_rag.logging_setup import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default="evals/golden_set.yaml")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--min-recall", type=float, default=0.0, help="低于该值则退出码为 1")
    args = parser.parse_args()

    setup_logging(get_settings().log_level)
    cases = load_cases(Path(args.golden))
    report = evaluate(cases, k=args.k)
    print(format_report(report))

    recall = report["metrics"]["source_recall"]
    if recall < args.min_recall:
        print(f"\n❌ source_recall {recall:.1%} 低于阈值 {args.min_recall:.1%}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
