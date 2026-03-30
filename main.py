from dotenv import load_dotenv
load_dotenv()

"""
main.py
Baseline LP evaluation pipeline — adapted for OptiBench_Linear_Only.json

Dataset schema per record:
{
    "question": str,          ← problem text
    "index":    int,          ← unique ID
    "type":     str,          ← e.g. "linear-notable", "linear-table"
    "results":  {             ← ground truth (all values are strings of floats)
        "<label>": "<float>",
        ...
    }
}

The LAST key in "results" is conventionally the overall objective
(e.g. "Total profit", "The maximum profit you can get").
All other keys are individual variable / sub-result values.

Usage:
    GROQ_API_KEY=<your_key> python main.py [--dataset OptiBench_Linear_Only.json]
"""

import argparse
import json
import sys

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from llm_client import generate_lp_json
from parser import parse_lp_json
from model_builder import build_model
from solver import solve_model
from evaluator import evaluate, aggregate_metrics
from utils import setup_logger, save_results, print_summary


# ── Dataset loading ──────────────────────────────────────────────────────────

def load_dataset(path: str) -> list[dict]:
    """
    Loads OptiBench_Linear_Only.json.
    Supports both a JSON array (the actual format) and .jsonl (one obj per line).
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # JSON array format (OptiBench default)
    if content.startswith("["):
        return json.loads(content)

    # Fallback: JSONL (one object per line)
    problems = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            problems.append(json.loads(line))
    return problems


def extract_ground_truth(problem: dict) -> dict:
    """
    Parse the "results" dict into a structured ground truth.

    OptiBench stores all result values as *strings* of floats, e.g.:
        {"Total profit": "655.0", "The number of black milk tea": "136.0"}

    Strategy:
    - The LAST entry is treated as the primary objective value.
    - All entries are also stored as named results for variable-level comparison.

    Returns:
        {
            "objective": float | None,
            "objective_label": str,
            "variables": {label: float, ...}   ← all results including objective
        }
    """
    raw_results: dict = problem.get("results", {})

    parsed: dict[str, float] = {}
    for label, val in raw_results.items():
        try:
            parsed[label] = float(val)
        except (ValueError, TypeError):
            pass   # skip non-numeric entries

    if not parsed:
        return {"objective": None, "objective_label": None, "variables": {}}

    # Last key = primary objective (OptiBench convention)
    objective_label = list(parsed.keys())[-1]
    objective_value = parsed[objective_label]

    return {
        "objective": objective_value,
        "objective_label": objective_label,
        "variables": parsed,   # all results, including the objective line
    }


# ── Single-problem pipeline ──────────────────────────────────────────────────

def run_pipeline(problem: dict, logger) -> dict:
    """
    Run the full pipeline for a single OptiBench problem.
    Returns a result dict with per-problem metrics.
    """
    pid  = str(problem.get("index", "unknown"))
    text = problem["question"]
    ptype = problem.get("type", "unknown")
    ground_truth = extract_ground_truth(problem)

    result = {
        "id":          pid,
        "type":        ptype,
        "json_valid":  False,
        "model_built": False,
        "metrics":     None,
        "ground_truth": ground_truth,
    }

    # Step 1: LLM → raw JSON dict
    logger.info(f"[{pid}] Calling LLM  (type={ptype})...")
    raw_dict = generate_lp_json(text)

    if raw_dict is None:
        logger.warning(f"[{pid}] LLM returned no usable output.")
        result["metrics"] = evaluate(None, ground_truth)
        return result

    # Step 2: Validate JSON structure
    lp_dict = parse_lp_json(raw_dict)

    if lp_dict is None:
        logger.warning(f"[{pid}] JSON parsing/validation failed.")
        result["metrics"] = evaluate(None, ground_truth)
        return result

    result["json_valid"] = True

    # Step 3: Build solver model
    try:
        prob, var_map = build_model(lp_dict)
        result["model_built"] = True
    except Exception as exc:
        logger.error(f"[{pid}] Model build error: {exc}")
        result["metrics"] = evaluate(None, ground_truth)
        return result

    # Step 4: Solve
    solution = solve_model(prob, var_map)

    if solution is None:
        logger.warning(f"[{pid}] Solver returned no solution.")
        result["metrics"] = evaluate(None, ground_truth)
        return result

    logger.info(
        f"[{pid}] Solved. Status={solution['status']}  "
        f"Objective={solution['objective']:.4f}  "
        f"(GT={ground_truth['objective']})"
    )

    # Step 5: Evaluate
    result["metrics"] = evaluate(solution, ground_truth)
    result["predicted"] = {
        "objective": solution["objective"],
        "variables": solution["variables"],
    }

    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LP Baseline Evaluation Pipeline (OptiBench)"
    )
    parser.add_argument(
        "--dataset",
        default="OptiBench_Linear_Only.json",
        help="Path to OptiBench JSON dataset (default: OptiBench_Linear_Only.json)",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory to write results (default: results)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N problems (useful for quick tests)",
    )
    args = parser.parse_args()

    logger = setup_logger()
    logger.info("Starting LP Baseline Evaluation Pipeline (OptiBench)")

    # Load
    try:
        problems = load_dataset(args.dataset)
    except FileNotFoundError:
        logger.error(f"Dataset not found: {args.dataset}")
        sys.exit(1)

    if args.limit:
        problems = problems[: args.limit]

    logger.info(f"Loaded {len(problems)} problems from {args.dataset}")

    # Iterate
    results = []
    iterator = tqdm(problems, desc="Evaluating") if HAS_TQDM else problems

    for problem in iterator:
        result = run_pipeline(problem, logger)
        results.append(result)

        m = result.get("metrics") or {}
        logger.info(
            f"  json_valid={result['json_valid']}  "
            f"exec_success={m.get('execution_success', False)}  "
            f"obj_correct={m.get('objective_correct', False)}  "
            f"obj_error={m.get('objective_abs_error')}"
        )

    # Aggregate & report
    summary = aggregate_metrics(results)
    print_summary(summary)
    save_results(results, summary, out_dir=args.output_dir)
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
