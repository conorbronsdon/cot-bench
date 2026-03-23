# First Test Run

Step-by-step guide to running your first COT Bench evaluation.

## Prerequisites

- Python 3.11+ with `pip install -e ".[dev]"` done
- An OpenAI API key (for user/tool simulators — GPT-4.1-mini)
- An Anthropic API key (for the Opus judge)

## Steps

### 1. Open a terminal in the repo

```bash
cd ~/cot-bench
```

### 2. Create your `.env` file

```bash
cp .env.example .env
```

Edit `.env` and paste in two keys:
- **OPENAI_API_KEY** — the user simulator and tool simulator both run on GPT-4.1-mini (OpenAI)
- **ANTHROPIC_API_KEY** — the judge is Claude Opus

You don't need any other keys for the first run.

### 3. Load the keys

```bash
source .env
```

### 4. Run the preflight check

```bash
python -m scripts.preflight
```

Every line should say `[PASS]`. If something says `[FAIL]`, fix that before proceeding.

### 5. Run the evaluation

```bash
bash scripts/first_run.sh
```

**What happens under the hood:**
- Loads the 2 banking scenarios (Margaret Chen's multi-task banking visit, James Okafor's bounced rent payment)
- For each scenario, runs GPT-4.1-mini through a multi-turn conversation (up to 10 turns)
- Then runs Claude Haiku 4.5 through the same scenarios
- After each simulation, sends the transcript to Claude Opus to score task completion and tool selection
- Saves results to `data/results/`

**Expected cost:** ~$1-3 (cheap models + 2 scenarios = minimal tokens)

**Expected time:** 5-15 minutes

### 6. View results

```bash
# Terminal summary prints automatically, or:
cat data/results/latest.csv

# Or open the leaderboard in a browser — it loads data/results/leaderboard.json
open frontend/index.html
```

## Next steps after the first run

```bash
# More models
python -m scripts.run_eval \
  --models "GPT-4.1" "Claude Sonnet 4.6" "Gemini 2.5 Pro" \
  --judges opus

# Both domains
python -m scripts.run_eval \
  --domains banking customer_success \
  --judges opus

# Generate more scenarios first (~$10-20 in API costs)
python -m scripts.generate_data --domain banking --scenarios-per-category 10
python -m scripts.generate_data --domain customer_success --scenarios-per-category 10

# Re-run with more data
python -m scripts.run_eval --judges opus

# Regenerate leaderboard
python -m scripts.aggregate_results
```

## Troubleshooting

**"No scenarios loaded"** — You're in the wrong directory. `cd ~/cot-bench`.

**API timeout/rate limit** — The run will log the error and move on. Re-run with `--models` targeting just the failed model.

**Judge returns unparseable response** — Logged as a warning, score defaults to 0.0. Check the logs for which scenario triggered it.
