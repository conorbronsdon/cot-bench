# Rehearsal Runbook — One-Session Execution Guide

The approved pre-launch rehearsal, written so the whole thing is one copy-paste
session once API keys exist. Every command below was verified against the code
at the commit this file landed in. Run everything from the repo root
(`C:\Users\conor\cot-bench`) — the harness loads `data/scenarios/` by relative
path and writes to `data/results/`.

**What the rehearsal is:** one cheap-capable contestant (Claude Haiku 4.5)
across the full public corpus (92 scenarios) plus the private holdout
(10 scenarios), 3 reliability runs, full 3-judge panel on the default combined
judge path, hard budget cap. Plus a null-agent subset pass and a user-sim
sensitivity subset pass. Results stay local. Nothing publishes.

**Total expected spend:** ~$26 main run (priors estimate; actuals usually come
in below) + ~$0.20 smoke + ~$0.70 null-agent + ~$2.30 sim-sensitivity ≈ **$30**,
hard-capped at $50 by `--max-cost`.

**Total wall-clock:** the main run is a single model evaluated sequentially
(306 evaluations, each a multi-turn simulation plus judge calls) — expect
roughly **5–12 hours**. Start it, watch the first few per-evaluation cost lines,
go to bed. Every completed evaluation is checkpointed to artifacts, and
`--resume` recovers a crash or budget stop without re-paying for finished work.

---

## 0. Prerequisites + loading keys

One-time checks before spending anything:

- Repo is current: `git -C C:\Users\conor\cot-bench pull --rebase`
- Package installed: `pip install -e ".[dev]"` (Python 3.11+)
- Holdout tree exists: `C:\Users\conor\cot-bench-holdout` with `banking\`
  (5 scenarios) and `customer_success\` (5 scenarios)
- `C:\Users\conor\cot-bench\.env` exists, created from `.env.example`, with
  real values for `OPENAI_API_KEY` (simulators), `ANTHROPIC_API_KEY` (Opus
  judge + Haiku contestant), `OPENROUTER_API_KEY` (Kimi/GLM judges).
  `GOOGLE_API_KEY` is **not needed** — no Gemini model runs in this rehearsal.

The harness reads plain environment variables. **There is no dotenv autoload** —
sourcing/parsing `.env` into the session is on you, once per terminal session.

### PowerShell (primary)

`.env` lines look like `export NAME=value` (bash format), so strip the
`export ` prefix while parsing:

```powershell
Set-Location C:\Users\conor\cot-bench
Get-Content .\.env | ForEach-Object {
    $line = $_.Trim()
    if (-not $line -or $line.StartsWith('#')) { return }
    if ($line.StartsWith('export ')) { $line = $line.Substring(7).Trim() }
    $eq = $line.IndexOf('=')
    if ($eq -lt 1) { return }
    $name  = $line.Substring(0, $eq).Trim()
    $value = $line.Substring($eq + 1).Trim().Trim('"').Trim("'")
    if ($value) { Set-Item -Path ("Env:" + $name) -Value $value }
}
```

Verify (prints a key prefix, never the full key):

```powershell
foreach ($k in 'OPENAI_API_KEY','ANTHROPIC_API_KEY','OPENROUTER_API_KEY') {
    $v = (Get-Item -Path ("Env:" + $k) -ErrorAction SilentlyContinue).Value
    if ($v) { "{0}: set ({1}...)" -f $k, $v.Substring(0, [Math]::Min(8, $v.Length)) }
    else    { "{0}: MISSING" -f $k }
}
```

### Bash variant

The file already uses `export`, so sourcing is enough:

```bash
cd /c/Users/conor/cot-bench
source .env
```

### Holdout location

The holdout tree is consumed via `--holdout-dir` on the command line, with the
`COT_BENCH_HOLDOUT_DIR` environment variable as fallback when the flag is
omitted (the flag wins when both are set — see `scripts/run_eval.py`). This
runbook passes the flag explicitly on the one command that needs it, so there
is no env var to remember to unset for the subset passes.

---

## 1. Preflight + token-prior sanity check

### 1a. Preflight

```powershell
python -m scripts.preflight
```

Expect every line `[PASS]` except `GOOGLE_API_KEY`, which reports as optional
("not set" is fine for this rehearsal). Preflight validates OpenAI
connectivity with a real API call, but only constructs a client for Anthropic
and **does not test the OpenRouter key at all** — the smoke run below is the
first real validation of `ANTHROPIC_API_KEY` and `OPENROUTER_API_KEY`.

### 1b. Smoke run (~$0.20, ~5–10 min)

Two scenarios, one reliability run, all three judges — exercises every key and
produces real token counts to check the cost priors against:

```powershell
python -m scripts.run_eval --domains banking --models "Claude Haiku 4.5" --scenario-limit 2 --reliability-runs 1
```

### 1c. Prior sanity check

The per-evaluation token priors live in `eval/config.py`
(`PER_EVAL_TOKEN_PRIORS`: agent 9000 in / 1500 out, sims 12000 / 2000 summed,
judges 7000 / 600 each, combined path). They were calibrated from the
2026-06-09 smoke runs and are deliberately conservative — the preflight
estimate should **over**-state the bill. Check that holds:

```powershell
python -c "import json; m=json.load(open('data/results/run_manifest.json')); c=m['cost']; print('estimate $%.4f  actual $%.4f  ratio %.2f' % (c['estimate_usd'], c['actual_usd'], c['actual_usd']/c['estimate_usd']))"
```

Also compare measured agent tokens per row against the priors:

```powershell
python -c "import glob, pandas as pd; p=sorted(glob.glob('data/results/results_*.parquet'))[-1]; df=pd.read_parquet(p); print(p); print(df[['input_tokens','output_tokens','total_turns','latency_ms']].describe().loc[['mean','max']])"
```

**What deviation matters:**

- ratio ≤ 1.0 — priors hold; proceed.
- ratio 1.0–1.5 — priors understate. Multiply the main run's printed estimate
  (~$26) by the ratio and confirm it still clears `--max-cost 50` comfortably.
- ratio > 1.5, or mean agent `input_tokens` well above 9000 — **stop**. Update
  `PER_EVAL_TOKEN_PRIORS` in `eval/config.py` from the measured numbers before
  the main run, or the budget math for the full leaderboard run is fiction.

Two scenarios is a small sample; judge a big ratio by the token columns, not
the ratio alone.

---

## 2. Main rehearsal run (~$26 est., 5–12 h)

One contestant, full public corpus + holdout, 3 reliability runs, all three
judges (`--judges` defaults to `kimi glm opus`), combined judge calls (the
default — do **not** pass `--separate-judge-calls`), artifacts on (default),
pre-registration and environment capture automatic:

```powershell
python -m scripts.run_eval --models "Claude Haiku 4.5" --holdout-dir C:\Users\conor\cot-bench-holdout --reliability-runs 3 --max-cost 50
```

What you should see in the first minute, in order:

1. `Loaded 49 scenarios for banking` / `Loaded 43 scenarios for customer_success`
2. `Loaded 5 HOLDOUT scenarios for banking` / `... 5 ... for customer_success`
3. `Pre-registration written to data\results\pre_registration.json (before any model call)`
4. `Cost ESTIMATE (priors, before any call): $25.85 total over 306 evaluations`
   (give or take pricing-table updates) and `Budget cap (--max-cost): $50.00`
5. Per-evaluation lines with a `running cost:` tally after each one.

Note the run id from the log (`results_YYYYMMDD_HHMMSS`, the parquet stem), or
recover it later:

```powershell
(Get-ChildItem data\results\results_*.parquet | Sort-Object Name | Select-Object -Last 1).BaseName
```

If you want the console log on disk, append
`2>&1 | Tee-Object -FilePath data\results\rehearsal-main.log` (Windows
PowerShell renders the tee'd stderr lines red; harmless). Everything needed for
analysis is persisted regardless: `pre_registration.json`, `run_manifest.json`,
per-evaluation artifacts, traces.

### Exit codes

Check with `echo $LASTEXITCODE` (PowerShell) / `echo $?` (bash):

- **0** — clean finish.
- **7** — `BUDGET_EXCEEDED_EXIT_CODE` (`eval/cost.py`): actual spend reached
  `--max-cost`. This is a **graceful stop, not a crash** — in-flight
  evaluations finished, and all artifacts, the parquet, and the manifest for
  completed work are on disk. Resume below.
- anything else — a crash. Completed evaluations are still checkpointed as
  artifacts; resume below.

### Resume procedure (crash or budget stop)

```powershell
python -m scripts.run_eval --resume results_YYYYMMDD_HHMMSS --models "Claude Haiku 4.5" --holdout-dir C:\Users\conor\cot-bench-holdout --reliability-runs 3 --max-cost 25
```

Substitute the real run id and the same flags as the original run. Resume reads
the run's artifact directory, skips every completed (model, scenario, run)
tuple — never paying twice — merges the old rows back in, and recomputes
reliability across the merged set. It continues under the **original**
pre-registration and aborts if the scenario corpus changed (so don't pull or
edit scenarios mid-rehearsal). Two cautions:

- `--max-cost` on a resumed run counts **only the new session's spend** (the
  accumulator starts at zero per process). Set it to roughly what's left of the
  $50, not $50 again.
- `--resume` is incompatible with `--no-artifacts` (the harness refuses).

### Aggregate immediately — before steps 3 and 4

`scripts/aggregate_results.py` always loads the **newest** `results_*.parquet`.
The subset passes below write newer parquets, so build the leaderboard from the
main run *now*:

```powershell
python -m scripts.aggregate_results
```

This writes `data/results/leaderboard.json`, `latest.csv`, and appends one line
to `history.jsonl` — all local-only for this rehearsal (see DO-NOTs). Do not
re-run aggregation after steps 3/4; it would aggregate the wrong parquet.

---

## 3. Null-agent subset pass (~$0.70, ~15–30 min)

Anti-gaming validation: the deterministic do-nothing agent should score near
zero. `--models null-agent` runs **only** the null agent
(`--include-null-agent` exists too, but it *adds* the null agent to other
models — not what we want here). Subset = first 5 scenarios per domain
(`--scenario-limit` truncates each domain's sorted file list, so the subset is
deterministic). No holdout flag — keep holdout transcripts out of an extra
artifact tree:

```powershell
python -m scripts.run_eval --domains banking customer_success --models null-agent --scenario-limit 5 --reliability-runs 1
```

Check the parquet directly (the leaderboard excludes the null agent by design,
so don't look for it there):

```powershell
python -c "import glob, pandas as pd; p=sorted(glob.glob('data/results/results_*.parquet'))[-1]; df=pd.read_parquet(p); print(p); print(df.groupby('model')[['efficacy','task_completion','tool_selection','state_score']].mean().round(4)); print(df['failure_mode'].value_counts())"
```

**Healthy:** mean efficacy and state_score near zero (≤ ~0.1), every row
carrying a failure mode. **Blocks the full run:** the null agent scoring
materially above ~0.1–0.15 means the judges reward doing nothing politely —
a scoring problem to fix before any paid leaderboard run.

---

## 4. Sim-sensitivity subset pass (~$2.30, ~30–60 min) — #32 part 2

Same contestant, same first-10-per-domain subset, but the **user simulator**
swapped from the default (`gpt-4.1-mini-2025-04-14`, see `SimulationConfig` in
`eval/config.py`) to a Claude model. `claude-sonnet-4-6` is used because it is
already priced in `TOKEN_COSTS` and is not the contestant (avoids an
agent-simulates-itself confound). The provider is inferred from the id
(`claude*` → anthropic) and the override is recorded in the pre-registration
and on every result row (`user_sim_model` column). Tool sim stays default.

```powershell
python -m scripts.run_eval --domains banking customer_success --models "Claude Haiku 4.5" --user-sim-model claude-sonnet-4-6 --scenario-limit 10 --reliability-runs 1
```

The subset overlaps the main run (same sorted-order scenarios), so compare
per-scenario efficacy against the main run's rows for the same ids. Fill in the
two run ids:

```powershell
python -c "import pandas as pd; main=pd.read_parquet('data/results/MAIN_RUN_ID.parquet'); sens=pd.read_parquet('data/results/SENS_RUN_ID.parquet'); ids=sens['scenario_id'].unique(); base=main[main['scenario_id'].isin(ids) & ~main['holdout']].groupby('scenario_id')['efficacy'].mean(); swap=sens.groupby('scenario_id')['efficacy'].mean(); print('n=%d  gpt-4.1-mini sim: %.4f  claude sim: %.4f  delta: %+.4f' % (len(ids), base.mean(), swap.mean(), base.mean()-swap.mean()))"
```

**Healthy:** |delta| under ~0.05. **Worth a pause:** |delta| ≥ ~0.09 — the
"Lost in Simulation" finding (sim choice alone swings agent scores ~9 points)
showing up in our harness; the full run's numbers would be partly a property of
the simulator. Doesn't necessarily block the rehearsal sign-off, but it must be
understood (and disclosed) before publishing leaderboard claims.

Note: this is the sim-**model** sensitivity test (#50/#32). It is distinct from
the behavioral `--sim-profile` flag (#59: `cooperative` / `impatient` /
`technically-confused` / `adversarial`) — profiles are NOT part of this
rehearsal; everything here runs cooperative by default.

---

## 5. Post-run analysis checklist

All from the **main run's** outputs unless noted. Work through in order; each
item says what healthy looks like and what blocks the full leaderboard run.

### 5a. Run manifest — `data/results/run_manifest.json`

- `models_failed`: `[]`. Anything listed = that provider path is broken.
- `cost.estimate_usd` vs `cost.actual_usd`: actual at or below estimate
  (priors are conservative). `budget_stopped: false` for a clean finish.
- `judges.resolved`: all three (`Kimi K2.6`, `GLM-4.6`, `Claude Opus 4.6`).
- `holdout`: `{"corpus_sha256": ..., "n_scenarios": 10}` — hash + count only,
  never content.
- `environment`: python version + `freeze_sha256` + `n_packages` present, and
  `env_freeze.txt` sits next to the manifest (automatic environment capture).
- `sim_profile`: `cooperative`.

### 5b. Pre-registration — `data/results/pre_registration.json`

Exists, timestamped **before** the first model call, public scenario set hashed
with its index, `holdout_set` with hash + count only (no IDs, no content).

### 5c. Results parquet — `data/results/<run_id>.parquet` (+ same-stem `.csv`)

Expect 306 rows (1 model × 102 scenarios × 3 runs). Quick health pass:

```powershell
python -c "import pandas as pd; df=pd.read_parquet('data/results/MAIN_RUN_ID.parquet'); print('rows:', len(df)); print('holdout rows:', int(df['holdout'].sum())); print('ended_by:'); print(df['ended_by'].value_counts()); print('premature_end rate: %.3f' % df['premature_end'].mean()); print('judge parse failures:', int(df['tc_parse_failures'].sum()+df['ts_parse_failures'].sum())); print('degraded rows:', int((df['tc_degraded']|df['ts_degraded']).sum())); print('high_disagreement rows:', int(df['high_disagreement'].sum())); print(df['failure_mode'].value_counts(dropna=True))"
```

- `holdout` rows: 30 (10 scenarios × 3 runs).
- Judge parse failures ~0; **more than ~5% of rows degraded blocks the full
  run** (consensus quality problem).
- `ended_by` mostly `user_sim`; a large `max_turns` share means conversations
  are hitting the wall — inspect transcripts before scaling up.
- `premature_end` rate: low (≲ 0.15). High = the sim quits before goals are
  verifiably met (the #32 decoupling) — needs investigation, it would distort
  every model's reliability in the full run.
- `failure_mode` populated on failed rows, with `failure_mode_source` mostly
  deterministic (state-grader / premature-flag), not all `fallback`.

### 5d. Leaderboard — `data/results/leaderboard.json` (from step 2's aggregate)

Single-model caveat: with one contestant, `clear_score` equals `efficacy` (no
field to normalize against) and rank bands are trivial — ignore both. The
fields that matter for the rehearsal:

- `models[0].efficacy` + `efficacy_ci`: a sane mid-range score with a CI that
  is not absurdly wide.
- `models[0].holdout_score` / `holdout_gap` (gap = public − holdout): healthy
  is a small gap, |gap| ≲ 0.1. A large **positive** gap is the overfitting
  tripwire firing — on a never-trained-against corpus that mostly means the
  holdout is harder than intended or too small to read; eyeball the holdout
  transcripts before drawing conclusions.
- `judge_alpha.task_completion` / `.tool_selection` (Krippendorff): ≥ 0.8
  solid, ≥ 0.667 tentative, **below 0.667 blocks the full run** — the panel
  isn't measuring one thing.
- `judge_deltas` / `same_lab_check` on the Claude contestant: Opus delta in
  line with the open judges' deltas. A materially negative Opus delta (Opus
  more generous to its sibling than the panel) must be flagged in any
  publication; a large one blocks.
- `length_bias`: per-dimension OLS slope of judge score on agent output
  tokens. Healthy: `significant: false`, or a significant slope with tiny
  `r_squared` (< ~0.05). A significant positive slope with real r² =
  verbosity bias; investigate before the full run.
- `models[0].premature_end_rate`, `failure_profile`,
  `reliability_pass_hat_k` (should decay gently from k=1 to k=3; a cliff means
  high run-to-run variance), `reliability` (pass@3) and
  `reliability_consistency`.
- `holdout.present`: `true`. `sim_profile_robustness`: **absent** — correct,
  the key only appears when non-cooperative profile rows exist, and this
  rehearsal runs cooperative only.

### 5e. Halo delta — criterion vs holistic, per judge (atomic rubrics, #54)

Every scenario (public and holdout) carries `rubric_criteria`, so every judge
result in the artifacts records both the criterion-informed `overall_score`
(what counts) and the judge's holistic template score (`holistic_score`, what
the score *would have been* pre-#54), plus per-criterion `criteria_verdicts`.
Measure the halo delta per judge per dimension (positive = the holistic score
was inflated relative to criterion-grounded grading):

```powershell
python -c "import json, glob, collections, statistics; deltas=collections.defaultdict(list); files=glob.glob('data/results/artifacts/MAIN_RUN_ID/*/*.json'); [deltas[(j['judge_name'],dim)].append(j['holistic_score']-j['overall_score']) for f in files for dim in ('task_completion','tool_selection') for j in json.load(open(f,encoding='utf-8'))['judges'][dim] if j.get('criterion_informed') and j.get('holistic_score') is not None and not j.get('parse_failed')]; [print('%-18s %-16s n=%-4d mean=%+.4f stdev=%.4f' % (k[0], k[1], len(v), statistics.mean(v), statistics.pstdev(v))) for k,v in sorted(deltas.items())]"
```

- **Healthy:** `criterion_informed` true on essentially every non-parse-failed
  judge entry (if it's false everywhere, criteria aren't reaching the judges —
  harness bug, blocks); small positive mean deltas (criteria pulling inflated
  holistic grades down a bit is the expected halo correction).
- **Investigate before full run:** one judge with a much larger |delta| than
  the others (that judge's holistic grading is halo-dominated — relevant to
  judge-panel trust), or large negative deltas (criteria systematically
  *raising* scores — check the criteria aren't too easy).

This number is the rehearsal's new deliverable since the plan was made: record
the per-judge halo delta in the rehearsal notes.

### 5f. Traces

`data/results/traces/<run_id>/spans.jsonl` exists and is non-trivial.

### 5g. Corpus-health / never-passed diagnostics — only if PR #72 has merged

As of this runbook's commit, PR #72 ("Leaderboard: consistency bands +
corpus-health stats", #71) is **open, not merged** — the main-run
`leaderboard.json` will not contain these fields. Check before the rehearsal:

```powershell
gh pr view 72 --repo conorbronsdon/cot-bench --json state,mergedAt
```

If it merged and you pulled master before running: also inspect the
corpus-health block it adds to `leaderboard.json` (per-scenario consistency /
never-passed diagnostics) — a scenario that **no** run ever passed is either
genuinely hard or broken; read its transcript and flag it for corpus review
before the full run.

---

## 6. DO NOT

- **Do not publish anything.** No `python -m scripts.check_publish_ready`, no
  committing `data/results/leaderboard.json`, `latest.csv`, or `history.jsonl`
  (they are deliberately un-gitignored because the weekly workflow commits
  them — after the rehearsal they will show as uncommitted changes/untracked
  files; leave them that way). The publish gate would block this non-default
  config anyway (single model, and `--max-cost` runs aren't leaderboard runs);
  the rehearsal does not get near it.
- **Do not trigger GitHub Actions.** No `gh workflow run weekly-eval.yml`, no
  enabling schedules. The rehearsal is local-only.
- **Never put keys in files that commit.** Keys live in `.env` (gitignored)
  and session env vars only. Never paste a key into a script, doc, log you
  intend to share, or a `--flag`. Before any later commit from this machine:
  `git status` and confirm nothing under `data/results/` or `.env` is staged.
- **Artifacts stay local.** `data/results/artifacts/` holds full transcripts
  **including private-holdout scenario content** whenever the holdout ran. It
  is gitignored; never commit it, never upload it, never share it for
  calibration without stripping `holdout: true` files first.
- **Do not pass `--no-artifacts`** on any rehearsal run — it kills both crash
  recovery (`--resume` requires artifacts) and the halo-delta analysis (5e).
- **Do not re-run `aggregate_results` after steps 3/4** — it reads the newest
  parquet and would build a junk leaderboard from a subset pass (and append a
  junk line to `history.jsonl`).
- **Do not pull, rebase, or edit scenarios mid-rehearsal** — `--resume`
  verifies the corpus hash against the original pre-registration and aborts on
  drift.
