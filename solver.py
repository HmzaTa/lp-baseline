"""
solver.py
Solves a PuLP model and returns the objective value + variable values.
"""

import pulp


def solve_model(prob: pulp.LpProblem, var_map: dict) -> dict | None:
    """
    Solve the LP and return:
        {
            "objective": float,
            "variables": {name: value, ...},
            "status": str
        }
    Returns None if the model is infeasible / unbounded / errored.
    """
    # Suppress solver output
    solver = pulp.PULP_CBC_CMD(msg=False)

    try:
        prob.solve(solver)
    except Exception as exc:
        print(f"[solver] Solver exception: {exc}")
        return None

    status = pulp.LpStatus[prob.status]

    if prob.status != pulp.LpStatusNotSolved and pulp.value(prob.objective) is None:
        print(f"[solver] No objective value. Status: {status}")
        return None

    if status not in ("Optimal",):
        print(f"[solver] Non-optimal status: {status}")
        return None

    variables = {
        name: pulp.value(var) if pulp.value(var) is not None else 0.0
        for name, var in var_map.items()
    }

    return {
        "objective": pulp.value(prob.objective),
        "variables": variables,
        "status": status,
    }
