#!/usr/bin/env bash
# COT Bench — First Test Run
#
# Prerequisites:
#   1. pip install -e ".[dev]"
#   2. Set OPENAI_API_KEY and ANTHROPIC_API_KEY in your environment
#
# This runs a minimal evaluation:
#   - 2 cheap/fast models (GPT-4.1-mini, Claude Haiku 4.5)
#   - Banking domain only (2 scenarios)
#   - Opus judge only (no GPU needed)
#   - 1 reliability run (fastest possible)
#
# Estimated cost: ~$1-3 in API calls
# Estimated time: 5-15 minutes

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=== COT Bench First Run ==="
echo ""

# Pre-flight check
echo "Running pre-flight checks..."
echo ""
python -m scripts.preflight
PREFLIGHT_EXIT=$?

if [ $PREFLIGHT_EXIT -ne 0 ]; then
    echo ""
    echo "Pre-flight failed. Fix the issues above and re-run."
    exit 1
fi

echo ""
echo "Pre-flight passed. Starting evaluation..."
echo ""

# Run evaluation — minimal config for first test
python -m scripts.run_eval \
    --domains banking \
    --models "GPT-4.1-mini" "Claude Haiku 4.5" \
    --judges opus \
    --reliability-runs 1 \
    --output data/results/results_first_run.parquet

echo ""
echo "Evaluation complete. Generating leaderboard..."
echo ""

# Aggregate results
python -m scripts.aggregate_results

echo ""
echo "=== Done! ==="
echo ""
echo "Results saved to:"
echo "  data/results/results_first_run.parquet"
echo "  data/results/results_first_run.csv"
echo "  data/results/leaderboard.json"
echo "  data/results/latest.csv"
echo ""
echo "View the leaderboard: open frontend/index.html in a browser"
echo ""
echo "Next steps:"
echo "  - Add more models: --models \"GPT-4.1\" \"Claude Sonnet 4.6\" \"Gemini 2.5 Pro\""
echo "  - Add customer_success domain: --domains banking customer_success"
echo "  - Generate more scenarios: python -m scripts.generate_data --domain banking"
echo "  - Full run with all judges (needs GPU): --judges qwen3 deepseek opus"
