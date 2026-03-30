"""
evaluator.py
Compares the predicted LP solution against OptiBench ground truth.

OptiBench ground_truth shape (built by main.extract_ground_truth):
{
    "objective":       float | None,   ← primary scalar to match (last results key)
    "objective_label": str,            ← human-readable label of the objective
    "variables":       {label: float}  ← ALL result entries (incl. objective row)
}

Since OptiBench does NOT expose named decision-variable values that map 1-to-1
with PuLP variable names, variable_mae is computed only when the solver's
objective value can be matched — we compare the full set of ground-truth result
values against the single predicted objective (a best-effort proxy).
"""

OBJ_TOLERANCE = 1e-4


def evaluate(
    predicted: dict | None,
    ground_truth: dict,
) -> dict:
    """
    Args:
        predicted   : {"objective": float, "variables": {pulp_name: val}}
                      or None if the pipeline failed before producing a solution.
        ground_truth: as returned by main.extract_ground_truth()

    Returns:
        {
            "objective_correct":    bool,
            "objective_abs_error":  float | None,
            "variable_mae":         float | None,
            "execution_success":    bool,
        }
    """
    if predicted is None:
        return {
            "objective_correct":   False,
            "objective_abs_error": None,
            "variable_mae":        None,
            "execution_success":   False,
        }

    true_obj = ground_truth.get("objective")
    pred_obj = predicted.get("objective")

    # ── Objective accuracy ───────────────────────────────────────────────────
    if true_obj is not None and pred_obj is not None:
        obj_error   = abs(pred_obj - true_obj)
        obj_correct = obj_error < OBJ_TOLERANCE
    else:
        obj_error   = None
        obj_correct = False

    # ── Variable MAE (best-effort) ───────────────────────────────────────────
    # OptiBench "variables" = all result labels with their float values.
    # PuLP variable names rarely match those labels directly.
    # We therefore compare each ground-truth result value against the
    # predicted objective value as a coarse proxy — this gives a meaningful
    # signal for single-output problems and degrades gracefully for multi-output.
    gt_vars = ground_truth.get("variables", {})

    if gt_vars and pred_obj is not None:
        errors = [abs(pred_obj - v) for v in gt_vars.values()]
        # Use minimum error: at least one GT entry should be the objective
        var_mae = min(errors)
    else:
        var_mae = None

    return {
        "objective_correct":   obj_correct,
        "objective_abs_error": obj_error,
        "variable_mae":        var_mae,
        "execution_success":   True,
    }


def aggregate_metrics(results: list[dict]) -> dict:
    """
    Roll up per-problem metrics into overall summary statistics.
    Also breaks down success rate by problem type (linear-notable, linear-table…).
    """
    n = len(results)
    if n == 0:
        return {}

    n_json_valid    = sum(1 for r in results if r.get("json_valid", False))
    n_exec_success  = sum(
        1 for r in results
        if r.get("metrics", {}).get("execution_success", False)
    )
    n_obj_correct   = sum(
        1 for r in results
        if r.get("metrics", {}).get("objective_correct", False)
    )

    var_maes = [
        r["metrics"]["variable_mae"]
        for r in results
        if r.get("metrics", {}).get("variable_mae") is not None
    ]
    avg_var_mae = sum(var_maes) / len(var_maes) if var_maes else None

    # Per-type breakdown
    type_stats: dict[str, dict] = {}
    for r in results:
        ptype = r.get("type", "unknown")
        if ptype not in type_stats:
            type_stats[ptype] = {"total": 0, "exec_success": 0, "obj_correct": 0}
        type_stats[ptype]["total"] += 1
        m = r.get("metrics") or {}
        if m.get("execution_success"):
            type_stats[ptype]["exec_success"] += 1
        if m.get("objective_correct"):
            type_stats[ptype]["obj_correct"] += 1

    per_type = {
        ptype: {
            "total":                  s["total"],
            "execution_success_rate": s["exec_success"] / s["total"],
            "objective_accuracy_rate": s["obj_correct"] / s["total"],
        }
        for ptype, s in type_stats.items()
    }

    return {
        "total_problems":          n,
        "json_validity_rate":      n_json_valid   / n,
        "execution_success_rate":  n_exec_success / n,
        "objective_accuracy_rate": n_obj_correct  / n,
        "average_variable_mae":    avg_var_mae,
        "by_type":                 per_type,
    }
