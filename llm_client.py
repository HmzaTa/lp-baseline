"""
llm_client.py
Handles Groq API calls to generate LP JSON from problem text.
"""

import os
import json
from groq import Groq

# ── Schema shown to the LLM ────────────────────────────────────────────────
SCHEMA = """
{
  "sense": "minimize" | "maximize",

  "variables": [
    {
      "name": "<var_name>",
      "type": "continuous",
      "lb": <float|null>,
      "ub": <float|null>
    }
  ],

  "objective": {
    "terms": [
      {"var": "<var_name>", "coeff": <float>}
    ]
  },

  "constraints": [
    {
      "id": "<string>",
      "terms": [
        {"var": "<var_name>", "coeff": <float>}
      ],
      "sense": "<=" | ">=" | "=",
      "rhs": <float>
    }
  ]
}
"""

PROMPT_TEMPLATE = """\
You are an expert in linear programming.

Convert the following optimization problem into a structured JSON model following the EXACT schema below.

{schema}

Instructions:
- Identify all decision variables
- Define the objective function
- Define ALL constraints explicitly
- Use correct inequality directions
- Use only numeric coefficients
- Assume all variables are continuous unless explicitly stated
- Ensure all variables appear in the variables list
- Output ONLY valid JSON
- Do NOT include any explanations or text outside JSON

Problem:
{problem_text}
"""


def generate_lp_json(problem_text: str) -> dict | None:
    """
    Send problem_text to the LLM and return the parsed JSON dict.
    Returns None if the API call fails or the response is not valid JSON.
    """
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    prompt = PROMPT_TEMPLATE.format(schema=SCHEMA, problem_text=problem_text)

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
    except Exception as exc:
        print(f"[llm_client] API error: {exc}")
        return None

    # Strip accidental markdown fences
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        )

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"[llm_client] JSON decode error: {exc}")
        return None
