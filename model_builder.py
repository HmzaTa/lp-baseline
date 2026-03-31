"""
model_builder.py
Converts a validated LP/MILP JSON dict into a PuLP model.
Supports continuous, integer, and binary variable types.
"""

import pulp


# Mapping from JSON type strings to PuLP categories
VAR_TYPE_MAP = {
    "continuous": "Continuous",
    "integer":    "Integer",
    "binary":     "Integer",   # PuLP handles binary as Integer with lb=0, ub=1
}


def build_model(lp_dict: dict) -> tuple[pulp.LpProblem, dict]:
    """
    Build and return a PuLP LpProblem from the structured LP/MILP dict.
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
        name  = v["name"]
        lb    = v.get("lb")
        ub    = v.get("ub")
        vtype = v.get("type", "continuous").lower()

        # Default lb = 0 when null/missing
        if lb is None:
            lb = 0

        # Binary variables are always bounded [0, 1]
        if vtype == "binary":
            lb = 0
            ub = 1

        # Get PuLP category, default to Continuous if unknown type
        cat = VAR_TYPE_MAP.get(vtype, "Continuous")

        var_map[name] = pulp.LpVariable(
            name=name,
            lowBound=lb,
            upBound=ub,
            cat=cat,
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
