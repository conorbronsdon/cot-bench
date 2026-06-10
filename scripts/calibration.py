"""Human judge-calibration tooling for COT Bench (issue #33).

The judge research (MT-Bench / PoLL) is clear that inter-judge agreement is
necessary but not sufficient: three judges can agree with each other and all be
wrong. The single strongest pre-launch credibility move is to double-label a
small set of transcripts by hand and report judge-vs-human agreement. This
module is the complete code path for that study, so that once the rehearsal run
has produced artifacts it is one command to set up and one to score.

This is TOOLING ONLY. The human labeling itself runs after the rehearsal run
(see docs/methodology.md "Human judge calibration"). Nothing here spends API
budget; everything works offline against the per-run artifact JSON written by
``scripts/run_eval.py`` (see ``eval/artifacts.py``).

Two subcommands:

    sample   From a run's artifact directory, draw a stratified, seeded sample
             of transcripts and emit a BLIND labeling workbook (markdown sheets
             with the conversation rendered readable + rubric anchors + empty
             score fields) plus a SEPARATE key file holding the judge scores.
             Judge scores never appear in the workbook — blind labeling is the
             whole point.

    score    Given the filled-in workbook + the key file, compute human-vs-judge
             agreement: Krippendorff's alpha (reusing eval/scoring/agreement.py),
             mean absolute difference, and per-dimension Pearson correlation,
             both against the judge consensus and against each individual judge.
             Emits a markdown report suitable for pasting into methodology.

Workbook format is MARKDOWN, not CSV. A calibration sheet has to show a
multi-turn conversation with tool calls and the full rubric anchors; that is
unreadable crammed into spreadsheet cells, and a human labels far more reliably
reading a rendered transcript. The score fields are a tiny fixed block at the
top of each sheet that this script parses back out, so the round trip stays
machine-checkable without sacrificing readability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from eval.scoring.agreement import krippendorff_alpha
from eval.scoring.rubrics import compute_efficacy

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# The two rubric dimensions a human labels, matching the judge dimensions.
DIMENSIONS = ("task_completion", "tool_selection")

# Efficacy bands for stratification, so the human sees the full difficulty
# range (hard/ambiguous cases as well as easy ones), not just easy passes.
# Bands are over the per-evaluation efficacy reconstructed from the artifact's
# judge consensus + deterministic state score (the same compute_efficacy the
# leaderboard uses).
BAND_EDGES = (("low", 0.0, 0.34), ("mid", 0.34, 0.67), ("high", 0.67, 1.000001))

DEFAULT_SAMPLE_SIZE = 60
MIN_SAMPLE_SIZE = 50
MAX_SAMPLE_SIZE = 80
DEFAULT_SEED = 33  # issue #33; fixed so the sample is reproducible


# The rubric anchors shown inline on every sheet. These are the human-facing
# distillation of the published rubrics in eval/scoring/rubrics.py — the SAME
# 0-1 anchors the judges score against, so a human and a judge are answering the
# identical question. Kept terse for a labeler's working memory.
RUBRIC_ANCHORS = {
    "task_completion": (
        "**Task Completion (0.0-1.0)** — did the agent accomplish the user's goals?\n"
        "Score each goal COMPLETE (1.0) / PARTIAL (0.5) / FAILED (0.0), then take the\n"
        "mean; nudge +/-0.1 for: appropriate clarifying questions, graceful error\n"
        "recovery, staying in scope (no hallucinated capabilities), correct multi-step\n"
        "dependency handling. Clamp to [0.0, 1.0].\n"
        "- 1.0 = every goal fully met with correct tools and information\n"
        "- 0.5 = meaningful progress but not fully resolved\n"
        "- 0.0 = goals not addressed, wrong tools, or incorrect information"
    ),
    "tool_selection": (
        "**Tool Selection Quality (0.0-1.0)** — were the right tools called correctly?\n"
        "Per tool call weigh: selection correctness, parameter accuracy, sequencing\n"
        "(dependencies respected, e.g. lookup before update), necessity (no redundant\n"
        "calls), and omissions (calls that should have happened but didn't).\n"
        "- 1.0 = right tools, right params, right order, nothing missed or redundant\n"
        "- 0.5 = roughly right but wrong params / poor sequencing / a redundant call\n"
        "- 0.0 = wrong tools, missing required calls, or unnecessary calls throughout"
    ),
}


# --------------------------------------------------------------------------- #
# Artifact loading + per-evaluation reconstruction
# --------------------------------------------------------------------------- #


@dataclass
class Evaluation:
    """One per-evaluation artifact, with derived stratification keys.

    ``artifact_path`` is relative to the artifacts run directory so the key file
    and the workbook can both refer to the same stable id without absolute
    paths leaking machine-specific layout.
    """

    artifact_id: str  # stable real id: "{model-slug}/{scenario}_run{idx}"
    artifact_path: Path
    scenario_id: str
    model: str
    run_index: int
    domain: str
    category: str
    holdout: bool
    transcript: list[dict]
    # Judge consensus (median of valid judges) per dimension, and per-judge.
    consensus: dict[str, float | None] = field(default_factory=dict)
    per_judge: dict[str, dict[str, float]] = field(default_factory=dict)
    state_score: float | None = None
    efficacy: float = 0.0
    band: str = ""


def sheet_id_for(artifact_id: str, seed: int) -> str:
    """Opaque, deterministic per-sheet token shown to the labeler.

    The real ``artifact_id`` embeds the model slug, so printing it on a sheet
    tells the labeler which model produced the transcript and invites pro/anti-
    model priors. The visible sheet id is instead an opaque hash of the real id
    (salted with the sample seed); the real id and the model live only in the key
    file. ``score`` joins labels to the key by this same token, so the round trip
    is unaffected. 12 hex chars (48 bits) — collision-free for an 80-sheet sample.
    """
    digest = hashlib.sha256(f"{seed}|{artifact_id}".encode()).hexdigest()
    return f"tx-{digest[:12]}"


def _valid_judge_scores(judge_list: list[dict]) -> dict[str, float]:
    """Map judge_name -> overall_score for judges that did not parse-fail.

    Mirrors the consensus rule in eval/scoring/judge.py: a parse-failed judge is
    not a genuine 0.0 grade and is excluded from consensus/agreement math.
    """
    out: dict[str, float] = {}
    for jr in judge_list:
        if jr.get("parse_failed"):
            continue
        score = jr.get("overall_score")
        if score is None:
            continue
        out[jr["judge_name"]] = float(score)
    return out


def _median_or_none(values: list[float]) -> float | None:
    """Median of valid judge scores; None when no valid judge scored.

    Matches docs/methodology.md: consensus is the median of valid judge scores
    (robust to one outlier on an n=3 panel).
    """
    if not values:
        return None
    return float(statistics.median(values))


def _band_for(efficacy: float) -> str:
    """Bucket a 0-1 efficacy into low/mid/high for stratification."""
    for name, lo, hi in BAND_EDGES:
        if lo <= efficacy < hi:
            return name
    return "high"  # efficacy == 1.0 falls into the final (inclusive) band


def load_evaluations(artifacts_run_dir: Path) -> list[Evaluation]:
    """Load every per-evaluation artifact under a run's artifact directory.

    ``artifacts_run_dir`` is the per-run subtree
    ``data/results/artifacts/{run_id}`` written by run_eval. Files live one
    level down per model slug: ``{model-slug}/{scenario}_run{idx}.json``.

    Domain and category are persisted at the artifact top level (issue #46) and
    read directly. For legacy artifacts that predate that field, domain is derived
    from the known scenario-id prefixes and category falls back to "unknown" (see
    ``_derive_domain`` / ``_derive_category``). For each evaluation we reconstruct
    the judge consensus (median of valid judges) per dimension and the efficacy
    band, so the sample can span the full difficulty range.
    """
    evals: list[Evaluation] = []
    for path in sorted(artifacts_run_dir.glob("*/*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        judges = data.get("judges", {})
        per_judge: dict[str, dict[str, float]] = {}
        consensus: dict[str, float | None] = {}
        for dim in DIMENSIONS:
            scores = _valid_judge_scores(judges.get(dim, []))
            per_judge[dim] = scores
            consensus[dim] = _median_or_none(list(scores.values()))

        state = data.get("state")
        state_score = state.get("score") if isinstance(state, dict) else None

        # Reconstruct efficacy from the consensus the leaderboard would use, so
        # the band reflects the published difficulty. Missing consensus (no
        # valid judge on a dimension) is treated as 0.0 for banding only.
        tc = consensus["task_completion"] or 0.0
        ts = consensus["tool_selection"] or 0.0
        efficacy = compute_efficacy(tc, ts, state_score)

        domain = _derive_domain(data, path)
        category = _derive_category(data)
        model = data["model"]
        run_index = int(data["run_index"])
        scenario_id = data["scenario_id"]
        artifact_id = f"{path.parent.name}/{path.stem}"

        evals.append(
            Evaluation(
                artifact_id=artifact_id,
                artifact_path=path,
                scenario_id=scenario_id,
                model=model,
                run_index=run_index,
                domain=domain,
                category=category,
                holdout=bool(data.get("holdout", False)),
                transcript=data.get("transcript", []),
                consensus=consensus,
                per_judge=per_judge,
                state_score=state_score,
                efficacy=efficacy,
                band=_band_for(efficacy),
            )
        )
    return evals


# Maps a real scenario-id prefix to its canonical domain. Used ONLY for the
# legacy-artifact fallback (artifacts written before issue #46 added top-level
# domain/category). Real ids are e.g. "banking_adaptive_tool_use_0001" and
# "cs_adaptive_tool_use_0001" — note "cs_" -> "customer_success", which the old
# regex got wrong. Three generated-batch scenarios carry the long
# "customer_success_" prefix instead of "cs_", so both forms are mapped.
# Keyed by the underscore-terminated prefix; first match wins, so longer
# prefixes are listed before any prefix they contain.
_DOMAIN_ID_PREFIXES = {
    "customer_success_": "customer_success",
    "banking_": "banking",
    "cs_": "customer_success",
}


def _derive_domain(data: dict, path: Path) -> str:
    """Domain for stratification: explicit field, else scenario-id prefix.

    Post-#46 artifacts carry an explicit top-level ``domain`` (the authoritative
    ``Domain.value``) and this is used directly. The fallback exists only for
    legacy artifacts that predate that field: it matches the known corpus id
    prefixes (``banking_`` -> ``banking``, ``cs_`` / ``customer_success_`` ->
    ``customer_success``) and returns ``"unknown"`` for anything unrecognized
    rather than guessing.
    """
    if data.get("domain"):
        return str(data["domain"])
    scenario_id = str(data.get("scenario_id", ""))
    for prefix, domain in _DOMAIN_ID_PREFIXES.items():
        if scenario_id.startswith(prefix):
            return domain
    return "unknown"


def _derive_category(data: dict) -> str:
    """Category for stratification.

    Post-#46 artifacts carry an explicit top-level ``category`` (the
    authoritative ``Scenario.category``). Legacy artifacts do not, and the
    category is NOT reliably recoverable from the id (the id's middle segment
    happens to match for current scenarios but is not guaranteed), so the honest
    fallback is ``"unknown"`` — the stratification is coarser but never mislabeled.
    """
    return str(data.get("category") or "unknown")


# --------------------------------------------------------------------------- #
# Stratified sampling
# --------------------------------------------------------------------------- #


def exclude_holdout(evals: list[Evaluation]) -> tuple[list[Evaluation], int]:
    """Drop holdout-scenario evaluations; return (kept, n_dropped).

    A calibration workbook may be shared with an external (guest-network) labeler
    (issue #33). The private holdout (issue #31) must never leave the team, so its
    transcripts are excluded by default. ``--include-holdout`` overrides this with
    a loud warning at the call site.
    """
    kept = [e for e in evals if not e.holdout]
    return kept, len(evals) - len(kept)


def dedup_reliability_runs(evals: list[Evaluation], seed: int) -> list[Evaluation]:
    """Collapse to ONE evaluation per (scenario_id, model), seeded.

    Reliability runs of the same (scenario, model) produce near-identical
    transcripts. Labeling several of them wastes human effort and, worse, injects
    correlated rows into the alpha/Pearson math (pseudo-replication), biasing the
    headline agreement number. So by default the sampler keeps a single run per
    (scenario_id, model), chosen by a seeded draw for reproducibility.
    ``--all-runs`` keeps every run.
    """
    buckets: dict[tuple[str, str], list[Evaluation]] = {}
    for ev in evals:
        buckets.setdefault((ev.scenario_id, ev.model), []).append(ev)
    chosen: list[Evaluation] = []
    for key in sorted(buckets):
        bucket = sorted(buckets[key], key=lambda e: e.artifact_id)
        # Per-key derived seed so the choice is independent of iteration order.
        rng = random.Random(f"{seed}|dedup|{key[0]}|{key[1]}")
        chosen.append(rng.choice(bucket))
    return chosen


def stratify_key(ev: Evaluation) -> tuple[str, str, str]:
    """Stratification cell: (domain, category, efficacy band)."""
    return (ev.domain, ev.category, ev.band)


def stratified_sample(evals: list[Evaluation], n: int, seed: int) -> list[Evaluation]:
    """Draw ``n`` evaluations stratified by (domain, category, band), seeded.

    Allocation is proportional to each stratum's share of the population, with
    largest-remainder rounding so the parts sum to exactly ``n`` (or to the
    population size when fewer than ``n`` evaluations exist). Within a stratum
    the draw is a seeded ``random.sample``. The whole procedure is deterministic
    for a fixed ``seed`` and a fixed input set: strata are processed in sorted
    key order and each gets its own derived, order-independent RNG.

    Determinism is essential — the calibration sample must be reproducible so the
    study can be re-run or audited and land on the same transcripts.
    """
    if n <= 0:
        return []
    population = len(evals)
    target = min(n, population)

    # Group by stratum in deterministic (sorted) key order.
    buckets: dict[tuple[str, str, str], list[Evaluation]] = {}
    for ev in evals:
        buckets.setdefault(stratify_key(ev), []).append(ev)
    ordered_keys = sorted(buckets)

    # Proportional allocation with largest-remainder rounding, capped per
    # stratum by how many evaluations it actually holds.
    raw = {k: target * len(buckets[k]) / population for k in ordered_keys}
    alloc = {k: min(int(raw[k]), len(buckets[k])) for k in ordered_keys}
    assigned = sum(alloc.values())
    # Distribute the remaining slots to the largest fractional remainders,
    # skipping strata already at capacity. Ties break on sorted key order.
    remainders = sorted(
        ordered_keys,
        key=lambda k: (-(raw[k] - int(raw[k])), k),
    )
    i = 0
    while assigned < target and any(alloc[k] < len(buckets[k]) for k in ordered_keys):
        k = remainders[i % len(remainders)]
        if alloc[k] < len(buckets[k]):
            alloc[k] += 1
            assigned += 1
        i += 1

    # Draw within each stratum with a per-stratum derived seed so the result is
    # independent of dict/iteration order.
    sample: list[Evaluation] = []
    for k in ordered_keys:
        take = alloc[k]
        if take <= 0:
            continue
        bucket = sorted(buckets[k], key=lambda e: e.artifact_id)
        # Per-stratum derived seed (str is a supported Random seed type) so the
        # draw is independent of input/iteration order.
        rng = random.Random(f"{seed}|{k[0]}|{k[1]}|{k[2]}")
        sample.extend(rng.sample(bucket, take))

    # Stable, reproducible ordering of the final workbook.
    sample.sort(key=lambda e: (e.domain, e.category, e.band, e.artifact_id))
    return sample


# --------------------------------------------------------------------------- #
# Workbook rendering (BLIND) + key file
# --------------------------------------------------------------------------- #


def render_transcript(transcript: list[dict]) -> str:
    """Render the conversation in readable markdown, calls before results.

    Mirrors run_eval.format_transcript's ordering (user -> agent(with calls) ->
    tool result -> agent follow-up) but in markdown a human can skim. Tool calls
    are shown on the agent turn that issued them.
    """
    lines: list[str] = []
    for t in transcript:
        role = {"user": "USER", "agent": "AGENT", "tool": "TOOL"}.get(
            t.get("role", ""), str(t.get("role", "")).upper()
        )
        turn_no = t.get("turn_number", "")
        content = (t.get("content") or "").strip()
        lines.append(f"**[Turn {turn_no} | {role}]**")
        if content:
            lines.append("")
            lines.append(content)
        for tc in t.get("tool_calls", []) or []:
            args = json.dumps(tc.get("arguments", {}), ensure_ascii=False)
            lines.append("")
            lines.append(f"> tool call -> `{tc.get('tool_name')}({args})`")
        lines.append("")
    return "\n".join(lines).strip()


# A parseable score block at the top of every sheet. The labeler fills the two
# numbers; score() reads them back via SCORE_FIELD_RE. Blank/`_` = unlabeled.
SCORE_FIELD_RE = re.compile(r"^-\s*(task_completion|tool_selection)\s*:\s*(.*)$", re.MULTILINE)


def render_sheet(ev: Evaluation, index: int, total: int, seed: int) -> str:
    """Render ONE blind labeling sheet for an evaluation.

    Contains: an OPAQUE sheet token (not the real artifact id — that embeds the
    model slug and would reveal model identity to the labeler), the
    stratification context (domain, category, difficulty band — the band is the
    human-facing 'how hard', NOT a judge score), the empty score block, the
    rubric anchors, and the rendered transcript. It does NOT contain any judge
    score or the model identity — those live only in the key file, so labeling is
    blind to both the scores and which model produced the transcript.
    """
    anchors = "\n\n".join(RUBRIC_ANCHORS[d] for d in DIMENSIONS)
    return f"""# Calibration sheet {index}/{total}

**Sheet:** `{sheet_id_for(ev.artifact_id, seed)}`
**Domain:** {ev.domain}  |  **Category:** {ev.category}  |  **Difficulty band:** {ev.band}

> Difficulty band is a coarse stratification label so the sample spans easy and
> hard cases. It is NOT a judge score. You are labeling blind: no judge scores
> appear in this workbook.

## Your scores (fill these in)

Enter a number from 0.0 to 1.0 for each dimension. Leave as `_` if you skip it.

- task_completion: _
- tool_selection: _

## Rubric anchors

{anchors}

## Conversation transcript

{render_transcript(ev.transcript)}
"""


def render_index(sample: list[Evaluation], seed: int, source: str) -> str:
    """Render the workbook index / instructions sheet."""
    rows = "\n".join(
        f"| {i + 1} | `{sheet_id_for(ev.artifact_id, seed)}` "
        f"| {ev.domain} | {ev.category} | {ev.band} |"
        for i, ev in enumerate(sample)
    )
    strata = sorted({stratify_key(ev) for ev in sample})
    strata_lines = "\n".join(f"- {d} / {c} / {b}" for d, c, b in strata)
    return f"""# COT Bench — human judge calibration workbook

Blind double-labeling set (issue #33). {len(sample)} transcripts sampled from
`{source}` with seed {seed}, stratified by domain x category x difficulty band so
the full difficulty range is covered. Sheets are identified by an opaque token;
the model that produced each transcript is held in the key file, not shown here,
so labeling is blind to both judge scores and model identity.

## How to label

1. Open each `sheet_NNN.md` in order.
2. Read the conversation and score BOTH dimensions 0.0-1.0 using the rubric
   anchors printed on the sheet (the same anchors the judges use).
3. Fill the two numbers under "Your scores". Leave `_` to skip a dimension.
4. Do NOT look at the key file or any leaderboard output while labeling — the
   point is an independent human judgment.
5. Ideally two people label the same sheets independently (double-labeling).

When done, run:

    python -m scripts.calibration score --workbook <this dir> --key <key.json>

## Strata covered

{strata_lines}

## Sheets

| # | Sheet | Domain | Category | Band |
|---|-------|--------|----------|------|
{rows}
"""


def build_key(sample: list[Evaluation], seed: int, source: str) -> dict:
    """Build the SEPARATE key file: judge scores kept OUT of the workbook.

    Holds, per sampled artifact, the per-judge scores and the median consensus
    for each dimension, plus the state score and reconstructed efficacy/band.
    score() joins the human labels to this by artifact id.
    """
    return {
        "schema": "cot-bench-calibration-key/2",
        "issue": 33,
        "seed": seed,
        "source": source,
        "n": len(sample),
        # Keyed by the OPAQUE sheet token (what the labeler and `score` see); the
        # real artifact id and model identity are fields inside each entry, kept
        # in the key file only so the workbook stays blind to model identity.
        "evaluations": {
            sheet_id_for(ev.artifact_id, seed): {
                "artifact_id": ev.artifact_id,
                "scenario_id": ev.scenario_id,
                "model": ev.model,
                "run_index": ev.run_index,
                "domain": ev.domain,
                "category": ev.category,
                "band": ev.band,
                "efficacy": round(ev.efficacy, 4),
                "state_score": ev.state_score,
                "consensus": {
                    d: (None if ev.consensus[d] is None else round(ev.consensus[d], 4))
                    for d in DIMENSIONS
                },
                "per_judge": {
                    d: {name: round(s, 4) for name, s in ev.per_judge[d].items()}
                    for d in DIMENSIONS
                },
            }
            for ev in sample
        },
    }


def write_workbook(sample: list[Evaluation], out_dir: Path, seed: int, source: str) -> Path:
    """Write the blind workbook (index + one sheet per evaluation) and key file.

    The key file is written OUTSIDE the workbook directory (a sibling) so it
    cannot be opened by accident while labeling. Returns the key file path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README.md").write_text(render_index(sample, seed, source), encoding="utf-8")
    width = max(3, len(str(len(sample))))
    for i, ev in enumerate(sample):
        name = f"sheet_{str(i + 1).zfill(width)}.md"
        (out_dir / name).write_text(render_sheet(ev, i + 1, len(sample), seed), encoding="utf-8")

    key_path = out_dir.parent / f"{out_dir.name}_key.json"
    key_path.write_text(json.dumps(build_key(sample, seed, source), indent=2), encoding="utf-8")
    return key_path


# --------------------------------------------------------------------------- #
# Reading labels back + scoring
# --------------------------------------------------------------------------- #


def _parse_float_field(raw: str) -> float | None:
    """Parse a human-entered score field; '_'/blank/non-numeric -> None."""
    raw = raw.strip().strip("`")
    if raw in ("", "_", "-"):
        return None
    try:
        val = float(raw)
    except ValueError:
        return None
    return val


def read_labels(workbook_dir: Path) -> dict[str, dict[str, float | None]]:
    """Read filled-in human labels from a workbook directory.

    Returns ``{sheet_id: {dimension: score_or_None}}`` keyed by the opaque sheet
    token (the same token used as the key file's evaluation id). A sheet's token
    is read from its ``**Sheet:** `...` `` line; the two score fields are read from
    the parseable "Your scores" block.
    """
    labels: dict[str, dict[str, float | None]] = {}
    for path in sorted(workbook_dir.glob("sheet_*.md")):
        text = path.read_text(encoding="utf-8")
        m = re.search(r"\*\*Sheet:\*\*\s*`([^`]+)`", text)
        if not m:
            logger.warning("No sheet id in %s; skipping", path.name)
            continue
        sheet_id = m.group(1)
        scores: dict[str, float | None] = {d: None for d in DIMENSIONS}
        for fm in SCORE_FIELD_RE.finditer(text):
            dim, raw = fm.group(1), fm.group(2)
            scores[dim] = _parse_float_field(raw)
        labels[sheet_id] = scores
    return labels


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation; None when undefined (n<2 or zero variance)."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0 or syy == 0:
        return None
    return sxy / (sxx * syy) ** 0.5


@dataclass
class DimensionAgreement:
    """Human-vs-judge agreement metrics for one dimension and one comparison."""

    n: int
    alpha: float | None
    mean_abs_diff: float | None
    pearson: float | None


def _agreement(human: list[float], judge: list[float]) -> DimensionAgreement:
    """Compute alpha / MAD / Pearson for paired human and judge scores.

    Krippendorff's alpha is computed over the units x raters matrix [human,
    judge] reusing eval/scoring/agreement.py (interval level), so the same
    chance-corrected metric reported for inter-judge reliability is reported for
    human-vs-judge calibration.
    """
    n = len(human)
    if n == 0:
        return DimensionAgreement(0, None, None, None)
    reliability = [[h, j] for h, j in zip(human, judge)]
    alpha = krippendorff_alpha(reliability)
    mad = sum(abs(h - j) for h, j in zip(human, judge)) / n
    pearson = _pearson(human, judge)
    return DimensionAgreement(n, alpha, mad, pearson)


def compute_calibration(labels: dict[str, dict[str, float | None]], key: dict) -> dict:
    """Join human labels to the key and compute all agreement metrics.

    For each dimension, computes human-vs-consensus and human-vs-each-judge
    agreement (alpha / mean absolute difference / Pearson). Only artifacts with
    a human score AND the relevant judge score on a dimension are paired (a
    skipped label or a parse-failed judge drops that pair, not the whole study).
    """
    evals = key.get("evaluations", {})
    judge_names = sorted(
        {
            name
            for e in evals.values()
            for d in DIMENSIONS
            for name in e.get("per_judge", {}).get(d, {})
        }
    )

    result: dict = {
        "n_labeled": 0,
        "n_matched": 0,
        "judge_names": judge_names,
        "dimensions": {},
        "unmatched_artifacts": [],
    }

    matched_ids: set[str] = set()
    for dim in DIMENSIONS:
        # Human-vs-consensus pairs.
        h_cons: list[float] = []
        j_cons: list[float] = []
        # Human-vs-individual-judge pairs.
        h_judge: dict[str, list[float]] = {name: [] for name in judge_names}
        j_judge: dict[str, list[float]] = {name: [] for name in judge_names}

        for artifact_id, hscores in labels.items():
            hv = hscores.get(dim)
            if hv is None:
                continue
            entry = evals.get(artifact_id)
            if entry is None:
                if artifact_id not in result["unmatched_artifacts"]:
                    result["unmatched_artifacts"].append(artifact_id)
                continue
            matched_ids.add(artifact_id)
            cons = entry.get("consensus", {}).get(dim)
            if cons is not None:
                h_cons.append(hv)
                j_cons.append(float(cons))
            for name, js in entry.get("per_judge", {}).get(dim, {}).items():
                h_judge[name].append(hv)
                j_judge[name].append(float(js))

        result["dimensions"][dim] = {
            "consensus": _agreement(h_cons, j_cons).__dict__,
            "per_judge": {
                name: _agreement(h_judge[name], j_judge[name]).__dict__
                for name in judge_names
                if h_judge[name]
            },
        }

    result["n_labeled"] = sum(1 for s in labels.values() if any(v is not None for v in s.values()))
    result["n_matched"] = len(matched_ids)
    return result


def _fmt(value: float | None, places: int = 3) -> str:
    """Format a metric for the report; None -> 'n/a'."""
    return "n/a" if value is None else f"{value:.{places}f}"


def render_report(calib: dict, key: dict) -> str:
    """Render the human-vs-judge agreement report as markdown for methodology."""
    lines: list[str] = []
    lines.append("# COT Bench — judge calibration report (issue #33)")
    lines.append("")
    lines.append(
        f"Human-labeled transcripts: **{calib['n_labeled']}** "
        f"(matched to key: {calib['n_matched']}). "
        f"Sample seed {key.get('seed')}, source `{key.get('source')}`."
    )
    if calib["unmatched_artifacts"]:
        lines.append("")
        lines.append(
            "> Warning: labeled artifacts not in key (ignored): "
            + ", ".join(f"`{a}`" for a in calib["unmatched_artifacts"])
        )
    lines.append("")
    lines.append(
        "Metrics: Krippendorff's alpha (interval level, the same chance-corrected "
        "agreement metric used for inter-judge reliability), mean absolute "
        "difference (MAD, lower is better), and Pearson correlation."
    )
    lines.append("")

    for dim in DIMENSIONS:
        d = calib["dimensions"].get(dim, {})
        lines.append(f"## {dim}")
        lines.append("")
        lines.append("| Comparison | n | alpha | MAD | Pearson |")
        lines.append("|------------|---|-------|-----|---------|")
        cons = d.get("consensus", {})
        lines.append(
            f"| human vs **consensus** | {cons.get('n', 0)} "
            f"| {_fmt(cons.get('alpha'))} | {_fmt(cons.get('mean_abs_diff'))} "
            f"| {_fmt(cons.get('pearson'))} |"
        )
        for name in calib["judge_names"]:
            pj = d.get("per_judge", {}).get(name)
            if not pj:
                continue
            lines.append(
                f"| human vs {name} | {pj.get('n', 0)} "
                f"| {_fmt(pj.get('alpha'))} | {_fmt(pj.get('mean_abs_diff'))} "
                f"| {_fmt(pj.get('pearson'))} |"
            )
        lines.append("")

    lines.append(
        "Interpretation: human-vs-consensus alpha is the headline number — it is "
        'what turns "we use three judges" into "our judges agree with humans at '
        'alpha = X." A materially higher human-vs-consensus alpha than the worst '
        "human-vs-single-judge alpha is evidence the panel beats any one judge, "
        "the PoLL result."
    )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def cmd_sample(args: argparse.Namespace) -> int:
    artifacts_run_dir = Path(args.artifacts)
    if not artifacts_run_dir.exists():
        logger.error("Artifacts directory not found: %s", artifacts_run_dir)
        return 1
    if not (MIN_SAMPLE_SIZE <= args.n <= MAX_SAMPLE_SIZE):
        logger.warning(
            "Sample size %d is outside the recommended %d-%d range (issue #33).",
            args.n,
            MIN_SAMPLE_SIZE,
            MAX_SAMPLE_SIZE,
        )

    evals = load_evaluations(artifacts_run_dir)
    if not evals:
        logger.error(
            "No artifacts found under %s (expected {model-slug}/*.json). "
            "Run the rehearsal run first.",
            artifacts_run_dir,
        )
        return 1
    logger.info("Loaded %d evaluations from %s", len(evals), artifacts_run_dir)

    # Exclude private-holdout transcripts by default (issue #31): a workbook may
    # be shared with an external labeler and the holdout must not leak.
    if args.include_holdout:
        n_holdout = sum(1 for e in evals if e.holdout)
        if n_holdout:
            logger.warning(
                "!!! --include-holdout: %d PRIVATE HOLDOUT transcript(s) WILL be "
                "included in this workbook. Do NOT share it outside the team — "
                "holdout exposure defeats the overfitting tripwire (issue #31).",
                n_holdout,
            )
    else:
        evals, n_dropped = exclude_holdout(evals)
        if n_dropped:
            logger.info(
                "Excluded %d holdout transcript(s) from the workbook "
                "(use --include-holdout to override; not recommended for shared sheets).",
                n_dropped,
            )

    # Collapse reliability repeats to one transcript per (scenario, model) so the
    # human never labels near-duplicates (pseudo-replication biases alpha).
    if not args.all_runs:
        before = len(evals)
        evals = dedup_reliability_runs(evals, args.seed)
        if len(evals) < before:
            logger.info(
                "Collapsed %d reliability repeats to %d unique (scenario, model) "
                "transcripts (use --all-runs to keep every run).",
                before,
                len(evals),
            )

    if not evals:
        logger.error("No eligible evaluations left after filtering. Nothing to sample.")
        return 1

    sample = stratified_sample(evals, args.n, args.seed)
    out_dir = Path(args.out)
    key_path = write_workbook(sample, out_dir, args.seed, str(artifacts_run_dir))

    # Stratum coverage summary.
    coverage: dict[tuple[str, str, str], int] = {}
    for ev in sample:
        coverage[stratify_key(ev)] = coverage.get(stratify_key(ev), 0) + 1
    logger.info("Wrote %d sheets to %s", len(sample), out_dir)
    logger.info("Key file (judge scores, kept out of workbook): %s", key_path)
    logger.info("Strata covered: %d", len(coverage))
    for k in sorted(coverage):
        logger.info("  %s / %s / %s: %d", k[0], k[1], k[2], coverage[k])
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    workbook_dir = Path(args.workbook)
    key_path = Path(args.key)
    if not workbook_dir.exists():
        logger.error("Workbook directory not found: %s", workbook_dir)
        return 1
    if not key_path.exists():
        logger.error("Key file not found: %s", key_path)
        return 1

    labels = read_labels(workbook_dir)
    key = json.loads(key_path.read_text(encoding="utf-8"))
    calib = compute_calibration(labels, key)
    report = render_report(calib, key)

    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        logger.info("Report written to %s", args.out)
    else:
        print(report)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="COT Bench human judge-calibration tooling (issue #33)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_sample = sub.add_parser(
        "sample",
        help="Sample transcripts from a run and emit a blind labeling workbook + key.",
    )
    p_sample.add_argument(
        "--artifacts",
        required=True,
        help="A run's artifact directory: data/results/artifacts/{run_id}",
    )
    p_sample.add_argument(
        "--out",
        required=True,
        help="Output workbook directory (sheets + README). Key is written alongside.",
    )
    p_sample.add_argument(
        "-n",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Number of transcripts to sample (default {DEFAULT_SAMPLE_SIZE}, "
        f"recommended {MIN_SAMPLE_SIZE}-{MAX_SAMPLE_SIZE}).",
    )
    p_sample.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for the stratified draw (default {DEFAULT_SEED}).",
    )
    p_sample.add_argument(
        "--include-holdout",
        action="store_true",
        help=(
            "Include private-holdout (issue #31) transcripts in the workbook. OFF "
            "by default — a workbook may be shared with an external labeler and "
            "the holdout must not leak. Prints a loud warning when used."
        ),
    )
    p_sample.add_argument(
        "--all-runs",
        action="store_true",
        help=(
            "Keep every reliability run as a separate labeling candidate. OFF by "
            "default — the sampler collapses to one transcript per (scenario, "
            "model) so the human does not label near-duplicates, which would "
            "inject correlated rows into the agreement math (pseudo-replication)."
        ),
    )
    p_sample.set_defaults(func=cmd_sample)

    p_score = sub.add_parser(
        "score",
        help="Score filled-in human labels against the key (judge agreement).",
    )
    p_score.add_argument("--workbook", required=True, help="Filled-in workbook directory.")
    p_score.add_argument("--key", required=True, help="Key file written by `sample`.")
    p_score.add_argument(
        "--out",
        default=None,
        help="Write the markdown report here (default: print to stdout).",
    )
    p_score.set_defaults(func=cmd_score)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
