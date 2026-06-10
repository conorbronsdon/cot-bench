# Figures plan

Every figure in the paper is generated from a published run artifact, so populating
the Experiments section after the first eval run is mechanical: run the eval, then
run the generator. Nothing here is hand-drawn.

Data sources (all produced by `scripts/aggregate_results.py` unless noted):

- `data/results/leaderboard.json` — per-model dimensions, CIs, rank bands,
  `judge_deltas`, `same_lab_check`, `reliability_pass_hat_k`, `judge_alpha`,
  `length_bias`, `premature_end_rate`, and (private runs) holdout aggregates.
- `data/results/<run_id>.parquet` — per-row results (one row per
  scenario/model/run): `task_completion`, `tool_selection`, `state_score`,
  `output_tokens`, per-judge score columns, `premature_end`, `domain`, `category`.
- `data/results/artifacts/<run_id>/...` — per-evaluation transcripts + raw judge
  outputs (for any per-scenario drill-down).
- `data/results/latest.csv` — flat leaderboard for the table.

| # | Figure / table | File to emit | Data source | Generator |
|---|----------------|--------------|-------------|-----------|
| Tab 1 | Leaderboard (CLEAR + CI + band, efficacy, $/task, reliability, latency; per-domain beneath) | `leaderboard_table.tex` | `leaderboard.json`, `latest.csv` | `aggregate_results.py` output → small table-builder script (TBD `scripts/paper/make_leaderboard_table.py`) |
| Fig 1 | Judge-consensus efficacy vs deterministic state score (scatter, diagonal marked) | `judge_vs_state.pdf` | parquet: `task_completion`/`tool_selection` consensus vs `state_score` | TBD `scripts/paper/plot_judge_vs_state.py` |
| Fig 2 | Per-judge vs consensus deltas; same-lab pairing highlighted | `judge_deltas.pdf` | `leaderboard.json` `judge_deltas`, `same_lab_check` | TBD `scripts/paper/plot_judge_deltas.py` |
| Fig 3 | pass^k decay (pass^k vs k, per model) | `pass_hat_k.pdf` | `leaderboard.json` `reliability_pass_hat_k` | TBD `scripts/paper/plot_pass_hat_k.py` |
| Fig 4 | Cost-performance Pareto ($/task vs efficacy or reliability, frontier marked) | `cost_pareto.pdf` | `leaderboard.json` `cost`, `efficacy`, `reliability` | TBD `scripts/paper/plot_cost_pareto.py` |
| Tab 2 | Public vs holdout gap (per model) | `holdout_gap.tex` | `leaderboard.json` holdout aggregates (private run) | `aggregate_results.py` w/ `--holdout-dir` |
| Tab 3 | Krippendorff alpha + length-bias regression | `alpha_lengthbias.tex` | `leaderboard.json` `judge_alpha`, `length_bias` | `compute_judge_alpha` / `compute_length_bias` |
| Fig 5 (opt) | Failure taxonomy distribution (wrong tool / bad params / sequence / missed clarification / scope breach) | `failure_taxonomy.pdf` | OpenInference spans `traces/<run_id>/spans.jsonl` + artifacts | TBD `scripts/paper/plot_failure_taxonomy.py` |

Notes:
- The `scripts/paper/` generators are not built yet (no run to plot). Their job is
  pure read-and-render over the JSON/parquet above; no recomputation of scores in
  the plotting layer.
- Holdout figures (Tab 2) come only from a private run with `--holdout-dir` set; the
  public CI never produces them.
- Suggested plotting stack: matplotlib, no seaborn, one figure per file, vector PDF
  for the camera-ready.
- Keep figure captions consistent with the placeholders already written in
  `sections/06-experiments.tex`.
