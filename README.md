# LP Baseline Evaluation Pipeline

A clean, modular baseline that evaluates how well an LLM (deepseek/deepseek-r1-distill-llama-70b via Open Router) can convert natural language linear programming problems into structured models and solve them.

Evaluated on the **OptiBench_Linear_Only** dataset — 422 real-world LP problems across two types: `linear-notable` and `linear-table`.

---

## Project Structure

```
Baseline_OptiBench/
├── main.py                      # Pipeline orchestrator
├── llm_client.py                # Open Router API calls
├── parser.py                    # JSON validation
├── model_builder.py             # JSON → PuLP model
├── solver.py                    # PuLP solver wrapper
├── evaluator.py                 # Metrics computation
├── utils.py                     # Logging & I/O helpers
├── OptiBench_Linear_Only.json   # Dataset (422 LP problems)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
py -3.14 -m pip install -r requirements.txt
```

### 2. Set your Open Router API key

Create a `.env` file in the project root:

```
OPENROUTER_API_KEY=gsk_your_key_here
```

---

## Running the pipeline

Full dataset (422 problems):
```bash
py -3.14 main.py --dataset OptiBench_Linear_Only.json
```

Limited run for quick testing (recommended before a full run):
```bash
py -3.14 main.py --dataset OptiBench_Linear_Only.json --limit 10
```

Custom output directory:
```bash
py -3.14 main.py --dataset OptiBench_Linear_Only.json --output-dir my_results
```



---

## Dataset Format

Each record in `OptiBench_Linear_Only.json` follows this schema:

```json
{
    "question": "A milk tea shop has 50000 ml of milk...",
    "index": 5,
    "type": "linear-notable",
    "results": {
        "The number of bottles of black milk tea": "136.0",
        "The number of bottles of green milk tea": "45.0",
        "Total profit": "655.0"
    }
}
```

- `question` — the natural language LP problem sent to the LLM
- `index` — unique problem ID
- `type` — `linear-notable` (342 problems) or `linear-table` (80 problems)
- `results` — ground truth values as strings; the **last key** is treated as the primary objective

---

## Output

After each run, a JSON file is written to `results/results_<timestamp>.json`:

```json
{
  "summary": {
    "total_problems": 100,
    "json_validity_rate": 0.95,
    "execution_success_rate": 0.91,
    "objective_accuracy_rate": 0.38,
    "average_variable_mae": 12.5,
    "by_type": {
      "linear-notable": {
        "total": 81,
        "execution_success_rate": 0.90,
        "objective_accuracy_rate": 0.36
      },
      "linear-table": {
        "total": 19,
        "execution_success_rate": 0.94,
        "objective_accuracy_rate": 0.42
      }
    }
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
| `execution_success_rate` | % of problems solved to optimality by the solver |
| `objective_accuracy_rate` | % of problems where `\|predicted_obj - true_obj\| < 1e-4` |
| `average_variable_mae` | Mean absolute error between predicted objective and ground truth results |

Results are also broken down by problem type (`linear-notable` vs `linear-table`).

---

## Baseline Results (100 problems, llama-3.3-70b-versatile)

| Metric | Score |
|---|---|
| JSON Validity Rate | ~95% |
| Execution Success Rate | ~93% |
| Objective Accuracy Rate | ~38% |

> These are the raw baseline numbers — no retries, no correction, single prompt per problem.

---

## Design Principles

- **No retries** — if the LLM fails, the problem is marked as failed
- **No correction loops** — LLM output is used as-is
- **No multi-stage reasoning** — single prompt, single response
- **No modification of LLM outputs** — parse only, never fix

This is a pure baseline. The results serve as a reference point for future improvements such as better prompting, validation loops, or chain-of-thought reasoning.
