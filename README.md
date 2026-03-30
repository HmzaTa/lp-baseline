# LP Baseline Evaluation Pipeline

A clean, modular baseline that evaluates how well an LLM (llama-3.3-70b-versatile via Groq) can convert natural language LP problems into structured models and solve them.

---

## Project Structure

```
lp_baseline/
├── main.py           # Pipeline orchestrator
├── llm_client.py     # Groq API calls
├── parser.py         # JSON validation
├── model_builder.py  # JSON → PuLP model
├── solver.py         # PuLP solver wrapper
├── evaluator.py      # Metrics computation
├── utils.py          # Logging & I/O helpers
├── dataset.json      # Sample LP problems + ground truth
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your Groq API key

```bash
export GROQ_API_KEY="gsk_your_key_here"
```

On Windows (PowerShell):
```powershell
$env:GROQ_API_KEY = "gsk_your_key_here"
```

---

## Running the pipeline

```bash
python main.py
```

With a custom dataset:
```bash
python main.py --dataset my_problems.json --output-dir my_results
```

---

## Dataset Format

`dataset.json` must be a JSON array of objects:

```json
[
  {
    "id": "lp_001",
    "problem": "Maximize 3x + 4y subject to ...",
    "ground_truth": {
      "objective": 220.0,
      "variables": {"x": 40.0, "y": 20.0}
    }
  }
]
```

**`ground_truth.variables`** keys must match the variable names the LLM will likely choose. If you don't know them in advance, you can omit `variables` and only supply `objective`.

---

## Output

After the run, a JSON file is written to `results/results_<timestamp>.json`:

```json
{
  "summary": {
    "total_problems": 5,
    "json_validity_rate": 1.0,
    "execution_success_rate": 0.8,
    "objective_accuracy_rate": 0.8,
    "average_variable_mae": 0.5
  },
  "per_problem": [ ... ]
}
```

Logs are written to `logs/run_<timestamp>.log`.

---

## Metrics

| Metric | Description |
|---|---|
| `json_validity_rate` | % of problems where the LLM produced valid, schema-compliant JSON |
| `execution_success_rate` | % of problems that were solved to optimality |
| `objective_accuracy_rate` | % of problems where `|predicted_obj - true_obj| < 1e-4` |
| `average_variable_mae` | Mean absolute error across all matched decision variables |

---

## Design Principles (Baseline Rules)

- **No retries** — if the LLM fails, the problem is marked as failed
- **No correction loops** — LLM output is used as-is
- **No multi-stage reasoning** — single prompt, single response
- **No modification of LLM outputs** — parse only, never fix
