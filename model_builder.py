"""
model_builder.py
Converts a validated LP JSON dict into a PuLP model.
Falls back gracefully — raises ValueError on unrecoverable issues.
"""

import pulp


def build_model(lp_dict: dict) -> pulp.LpProblem:
    """
    Build and return a PuLP LpProblem from the structured LP dict.
    """
    sense = (
        pulp.LpMinimize
        if lp_dict["sense"] == "minimize"
        else pulp.LpMaximize
    )

    prob = pulp.LpProblem("lp_problem", sense)

    # ── Variables ────────────────────────────────────────────────────────────
    var_map: dict[str, pulp.LpVariable] = {}
    for v in lp_dict["variables"]:
        name = v["name"]
        lb = v.get("lb")
        ub = v.get("ub")

        # Default lb = 0 when null/missing
        if lb is None:
            lb = 0

        var_map[name] = pulp.LpVariable(
            name=name,
            lowBound=lb,
            upBound=ub,  # None means unbounded above
            cat="Continuous",
        )

    # ── Objective ────────────────────────────────────────────────────────────
    obj_terms = lp_dict["objective"]["terms"]
    objective = pulp.lpSum(
        term["coeff"] * var_map[term["var"]] for term in obj_terms
    )
    prob += objective, "objective"

    # ── Constraints ──────────────────────────────────────────────────────────
    for c in lp_dict["constraints"]:
        lhs = pulp.lpSum(
            term["coeff"] * var_map[term["var"]] for term in c["terms"]
        )
        rhs = c["rhs"]
        cid = c["id"]

        if c["sense"] == "<=":
            prob += lhs <= rhs, cid
        elif c["sense"] == ">=":
            prob += lhs >= rhs, cid
        elif c["sense"] == "=":
            prob += lhs == rhs, cid
        else:
            raise ValueError(f"Unknown constraint sense: {c['sense']}")

    return prob, var_map
