"""Deterministic-first failure-mode taxonomy for COT Bench (issue #55).

A leaderboard rank says *that* a model failed; practitioners need to know *how*.
This module classifies every FAILED evaluation — a row whose efficacy lands
below the same 0.7 threshold reliability already uses for pass/fail — into one
of six failure modes:

- ``tool-selection-error``    — wrong/missing/unnecessary tool choice
- ``policy-violation``        — acted outside authorization or policy
- ``incomplete-task``         — verifiable progress made but goals unmet
- ``premature-end``           — conversation ended before goals were met (#32)
- ``wrong-parameters``        — right tool, wrong arguments/values
- ``hallucinated-capability`` — claimed an ability or action it did not have

Classification is **deterministic-first** and makes NO new LLM calls. The
signals already on every result row are consulted in a fixed precedence order:

1. State grader (deterministic): the no-unauthorized-mutation contract failed
   (an empty assertion list with a mutated world) -> ``policy-violation``.
2. Premature-end instrumentation (deterministic, #32): the user sim ended the
   conversation while the state check was still below 1.0 -> ``premature-end``.
3. State grader (deterministic): partial verifiable progress (some assertions
   passed, some failed) -> ``incomplete-task``.
4. Judge-reasoning keywords (assist, not authority): the already-collected
   reasoning text from valid judges is scanned against per-mode keyword lists
   in a fixed priority order (most specific first): ``hallucinated-capability``
   > ``policy-violation`` > ``wrong-parameters`` > ``tool-selection-error``.
5. Fallback: ``incomplete-task`` — the one thing every failure has in common
   is that the task verifiably did not get done.

When a deterministic signal and a keyword signal disagree, the deterministic
signal wins (it is graded fact; the keyword match is an opinion heuristic).
Each classification records its ``source`` so the provenance of every published
failure count is auditable.
"""

import re

from eval.scoring.rubrics import PASS_THRESHOLD
from eval.scoring.state_check import UNAUTHORIZED_MUTATION_DETAIL

# The published failure-mode vocabulary (issue #55). Order matters: it is the
# stable display/reporting order used by aggregate_results' failure profiles.
TOOL_SELECTION_ERROR = "tool-selection-error"
POLICY_VIOLATION = "policy-violation"
INCOMPLETE_TASK = "incomplete-task"
PREMATURE_END = "premature-end"
WRONG_PARAMETERS = "wrong-parameters"
HALLUCINATED_CAPABILITY = "hallucinated-capability"

FAILURE_MODES = (
    TOOL_SELECTION_ERROR,
    POLICY_VIOLATION,
    INCOMPLETE_TASK,
    PREMATURE_END,
    WRONG_PARAMETERS,
    HALLUCINATED_CAPABILITY,
)

# Classification provenance values (published per row as failure_mode_source).
SOURCE_STATE_GRADER = "state-grader"
SOURCE_PREMATURE_FLAG = "premature-flag"
SOURCE_JUDGE_KEYWORD = "judge-keyword"
SOURCE_FALLBACK = "fallback"

# --- Keyword assist ----------------------------------------------------------
# Scanned against the lowercased concatenation of all VALID judges' reasoning
# (both rubric dimensions). Priority order is most-specific-first so a rare,
# sharp signal ("hallucinated") is not swallowed by a generic one ("wrong
# tool"). Keywords match at word boundaries (so "policy" cannot fire inside
# "policyholder"); the entries in _STEM_TOKENS are deliberate prefixes that
# match their inflections ("violat" -> violated/violation). Sense-level false
# positives remain possible — keyword hits are assist-only, behind every
# deterministic signal, and auditable via failure_mode_source.
_KEYWORD_PRIORITY: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        HALLUCINATED_CAPABILITY,
        (
            "hallucinat",
            "fabricat",
            "invented",
            "made up",
            "made-up",
            "claimed to have",
            "claims to have",
            "claimed it had",
            "nonexistent",
            "non-existent",
            "no such tool",
            "does not exist",
            "doesn't exist",
            "pretended",
        ),
    ),
    (
        POLICY_VIOLATION,
        (
            "policy",
            "unauthorized",
            "unauthorised",
            "without verifying",
            "without verification",
            "before verifying",
            "not permitted",
            "not allowed",
            "violat",  # violated / violation
            "out of scope",
            "outside its scope",
            "outside the scope",
            "compliance",
        ),
    ),
    (
        WRONG_PARAMETERS,
        (
            "wrong parameter",
            "incorrect parameter",
            "invalid parameter",
            "wrong argument",
            "incorrect argument",
            "wrong value",
            "incorrect value",
            "wrong amount",
            "incorrect amount",
            "wrong account",
            "missing parameter",
            "missing required",
        ),
    ),
    (
        TOOL_SELECTION_ERROR,
        (
            "wrong tool",
            "incorrect tool",
            "suboptimal tool",
            "better tool",
            "should have used",
            "should have called",
            "failed to call",
            "never called",
            "did not call",
            "didn't call",
            "missed tool",
            "missing tool call",
            "unnecessary call",
            "unnecessary tool",
            "redundant",
        ),
    ),
)


def judge_reasoning_text(*consensus_results) -> str:
    """Concatenated reasoning from all VALID judges across consensus results.

    Parse-failed judges are excluded — their reasoning is the parse-failure
    placeholder, not an assessment of the agent. Pure accessor (no I/O), shared
    by the live row builder and the resume path so both classify identically.
    """
    parts: list[str] = []
    for cr in consensus_results:
        for jr in getattr(cr, "judge_results", []) or []:
            if getattr(jr, "parse_failed", False):
                continue
            reasoning = getattr(jr, "reasoning", "") or ""
            if reasoning:
                parts.append(str(reasoning))
    return "\n".join(parts)


# Deliberate prefix stems: no trailing boundary, so inflections match.
_STEM_TOKENS = frozenset({"hallucinat", "fabricat", "violat"})

# "made up" as a verb of fabrication, not the idiom "made up their mind".
_MIND_IDIOM_GUARD = r"(?! (?:their|his|her|its|my|your|our) mind)"


def _compile_keyword(keyword: str) -> re.Pattern[str]:
    pattern = r"\b" + re.escape(keyword)
    if keyword == "made up":
        pattern += _MIND_IDIOM_GUARD
    if keyword not in _STEM_TOKENS and keyword[-1].isalnum():
        pattern += r"\b"
    return re.compile(pattern)


_KEYWORD_PATTERNS: tuple[tuple[str, tuple[re.Pattern[str], ...]], ...] = tuple(
    (mode, tuple(_compile_keyword(k) for k in keywords)) for mode, keywords in _KEYWORD_PRIORITY
)


def _keyword_mode(judge_reasoning: str) -> str | None:
    """First failure mode (in fixed priority order) with a keyword hit."""
    if not judge_reasoning:
        return None
    low = judge_reasoning.lower()
    for mode, patterns in _KEYWORD_PATTERNS:
        if any(p.search(low) for p in patterns):
            return mode
    return None


def classify_failure(
    efficacy: float | None,
    *,
    state_result: dict | None = None,
    premature_end: bool = False,
    judge_reasoning: str = "",
    pass_threshold: float = PASS_THRESHOLD,
) -> dict | None:
    """Classify one evaluation's failure mode; ``None`` when it is not a failure.

    Args:
        efficacy: The row's hybrid efficacy score. At or above
            ``pass_threshold`` the evaluation PASSED and there is nothing to
            classify (returns ``None``). ``None`` efficacy (no score at all)
            also returns ``None`` — an unscored row is not a classified failure.
        state_result: The deterministic state-grading dict from
            ``score_state_changes`` (``score``/``checks``/``n_passed``/
            ``n_total``), or ``None`` for legacy scenarios without ground truth.
        premature_end: The #32 instrumentation flag — the user sim ended the
            conversation while the state check was still below 1.0.
        judge_reasoning: Concatenated reasoning text from the valid judges
            (see :func:`judge_reasoning_text`). Keyword assist only — never
            overrides a deterministic signal.
        pass_threshold: Failure cutoff; defaults to the same 0.7 reliability
            already uses for pass/fail so "a failure" means one thing repo-wide.

    Returns:
        ``{"mode": <one of FAILURE_MODES>, "source": <provenance>}`` for a
        failed evaluation, ``None`` otherwise. Pure function (no I/O).
    """
    if efficacy is None or efficacy >= pass_threshold:
        return None

    # 1. Deterministic: the state grader's no-unauthorized-mutation contract
    # failed — the agent mutated a world it had no authorization to touch.
    if state_result is not None:
        for check in state_result.get("checks") or []:
            detail = str(check.get("detail", ""))
            if not check.get("passed") and detail.startswith(UNAUTHORIZED_MUTATION_DETAIL):
                return {"mode": POLICY_VIOLATION, "source": SOURCE_STATE_GRADER}

    # 2. Deterministic: the user sim quit before the goals were verifiably met
    # (#32 instrumentation, already on every row).
    if premature_end:
        return {"mode": PREMATURE_END, "source": SOURCE_PREMATURE_FLAG}

    # 3. Deterministic: partial verifiable progress — the agent demonstrably
    # started the task (some assertions passed) but did not finish it.
    if state_result is not None:
        n_total = state_result.get("n_total") or 0
        n_passed = state_result.get("n_passed") or 0
        if n_total > 0 and 0 < n_passed < n_total:
            return {"mode": INCOMPLETE_TASK, "source": SOURCE_STATE_GRADER}

    # 4. Keyword assist over the judges' already-written reasoning. No LLM call;
    # the text was collected when the row was scored.
    mode = _keyword_mode(judge_reasoning)
    if mode is not None:
        return {"mode": mode, "source": SOURCE_JUDGE_KEYWORD}

    # 5. Fallback: the task did not get done and no sharper signal exists.
    return {"mode": INCOMPLETE_TASK, "source": SOURCE_FALLBACK}
