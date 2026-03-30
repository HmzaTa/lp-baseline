"""
utils.py
Logging helpers and small utilities.
"""

import logging
import json
from datetime import datetime
from pathlib import Path


def setup_logger(log_dir: str = "logs") -> logging.Logger:
    Path(log_dir).mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"run_{ts}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("lp_baseline")


def save_results(results: list[dict], summary: dict, out_dir: str = "results") -> None:
    Path(out_dir).mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(Path(out_dir) / f"results_{ts}.json", "w") as f:
        json.dump({"summary": summary, "per_problem": results}, f, indent=2)

    print(f"\n[utils] Results saved to {out_dir}/results_{ts}.json")


def print_summary(summary: dict) -> None:
    print("\n" + "=" * 50)
    print("  BASELINE EVALUATION SUMMARY")
    print("=" * 50)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:<35} {v:.4f}")
        else:
            print(f"  {k:<35} {v}")
    print("=" * 50)
