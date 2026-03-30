"""
parser.py
Safely parses the raw LLM output into a validated LP dict.
Returns None on any error — no correction, no retry (pure baseline).
"""

import json
from typing import Any


REQUIRED_TOP_KEYS = {"sense", "variables", "objective", "constraints"}
VALID_SENSES = {"minimize", "maximize"}
VALID_CONSTRAINT_SENSES = {"<=", ">=", "="}


def parse_lp_json(raw: Any) -> dict | None:
    """
    Accept either a raw string or an already-parsed dict.
    Perform minimal structural validation.
    Returns the dict if it looks valid, else None.
    """
    if raw is None:
        return None

    # If we received a string (shouldn't happen after llm_client, but be safe)
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            print(f"[parser] JSON decode error: {exc}")
            return None
    elif isinstance(raw, dict):
        data = raw
    else:
        print(f"[parser] Unexpected type: {type(raw)}")
        return None

    # Top-level keys
    if not REQUIRED_TOP_KEYS.issubset(data.keys()):
        missing = REQUIRED_TOP_KEYS - data.keys()
        print(f"[parser] Missing required keys: {missing}")
        return None

    # Sense
    if data["sense"] not in VALID_SENSES:
        print(f"[parser] Invalid sense: {data['sense']}")
        return None

    # Variables list
    if not isinstance(data["variables"], list) or len(data["variables"]) == 0:
        print("[parser] 'variables' must be a non-empty list")
        return None

    for v in data["variables"]:
        if "name" not in v:
            print("[parser] Variable missing 'name'")
            return None

    # Objective
    if "terms" not in data["objective"]:
        print("[parser] Objective missing 'terms'")
        return None

    # Constraints
    if not isinstance(data["constraints"], list):
        print("[parser] 'constraints' must be a list")
        return None

    for c in data["constraints"]:
        if not {"id", "terms", "sense", "rhs"}.issubset(c.keys()):
            print(f"[parser] Constraint missing keys: {c}")
            return None
        if c["sense"] not in VALID_CONSTRAINT_SENSES:
            print(f"[parser] Invalid constraint sense: {c['sense']}")
            return None

    return data
