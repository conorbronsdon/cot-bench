"""COT Bench scoring rubrics — the published evaluation criteria for each judge.

These rubrics are the intellectual core of COT Bench. They are intentionally
published in full so that scores are reproducible and auditable by anyone.
"""

# --- Efficacy: Task Completion ---
# Evaluates whether the agent accomplished all user goals in the scenario.

TASK_COMPLETION_RUBRIC = """\
You are an expert evaluator assessing whether an AI agent successfully completed \
a user's goals in a multi-turn conversation.

## Context
- Domain: {domain}
- User Goals: {user_goals}
- Available Tools: {available_tools}

## Conversation Transcript
{transcript}

## Evaluation Criteria

Score each user goal as COMPLETE, PARTIAL, or FAILED:

1. **COMPLETE**: The agent fully addressed the goal. The correct tools were called, \
correct information was provided, and the user's need was met.
2. **PARTIAL**: The agent made meaningful progress but did not fully resolve the goal. \
For example: correct tool selected but wrong parameters, or correct information \
retrieved but not communicated clearly.
3. **FAILED**: The agent did not address the goal, used wrong tools, provided incorrect \
information, or gave up without resolution.

## Additional Factors (adjust score up or down):
- Did the agent ask appropriate clarifying questions when the request was ambiguous?
- Did the agent recover gracefully from tool errors or unexpected responses?
- Did the agent stay within its authorized scope (no hallucinated capabilities)?
- Did the agent handle multi-step dependencies correctly (e.g., verify identity before account access)?

## Response Format
Return a JSON object:
{{
    "goal_scores": [
        {{"goal": "<goal text>", "score": "COMPLETE|PARTIAL|FAILED", "reasoning": "<brief explanation>"}},
        ...
    ],
    "overall_score": <float 0.0-1.0>,
    "overall_reasoning": "<1-2 sentence summary>"
}}

Scoring: COMPLETE=1.0, PARTIAL=0.5, FAILED=0.0. Overall score is the mean of goal scores, \
adjusted +/- 0.1 for the additional factors above (clamped to [0.0, 1.0]).\
"""


# --- Efficacy: Tool Selection Quality ---
# Evaluates whether the agent selected the right tools with correct parameters.

TOOL_SELECTION_RUBRIC = """\
You are an expert evaluator assessing the quality of an AI agent's tool selection \
and parameter usage in a multi-turn conversation.

## Context
- Domain: {domain}
- Available Tools: {available_tools}

## Conversation Transcript (with tool calls)
{transcript}

## Evaluation Criteria

For each tool call the agent made, evaluate:

1. **Selection Correctness**: Was this the right tool for the user's need at this point \
in the conversation? Consider whether a better tool was available.
2. **Parameter Accuracy**: Were the parameters correct and complete? Were required fields \
provided? Were values accurate based on the conversation context?
3. **Sequencing**: Were tools called in a logical order? Were dependencies respected \
(e.g., lookup before update)?
4. **Necessity**: Was the tool call necessary, or did the agent make redundant/unnecessary calls?
5. **Omissions**: Were there tool calls the agent should have made but didn't?

## Response Format
Return a JSON object:
{{
    "tool_call_scores": [
        {{
            "tool_name": "<name>",
            "turn": <int>,
            "selection_correct": true|false,
            "parameters_correct": true|false,
            "was_necessary": true|false,
            "reasoning": "<brief explanation>"
        }},
        ...
    ],
    "missed_tool_calls": [
        {{"expected_tool": "<name>", "context": "<why it should have been called>"}},
        ...
    ],
    "overall_score": <float 0.0-1.0>,
    "overall_reasoning": "<1-2 sentence summary>"
}}

Scoring: Each correct, necessary tool call with correct parameters scores 1.0. \
Deductions: wrong tool (-1.0), wrong parameters (-0.5), unnecessary call (-0.3), \
missed call (-0.5 each). Normalize to [0.0, 1.0].\
"""


# --- Efficacy: Combined Task Completion + Tool Selection ---
# Presents BOTH evaluation tasks in a single judge prompt. Context (domain,
# user goals, available tools) and the transcript are sent ONCE — this halves
# the judge call count AND cuts judge input tokens nearly in half versus the
# two-prompt path, because the transcript (the bulk of the prompt) is no longer
# duplicated across two calls.
#
# The per-dimension scoring instructions below are preserved VERBATIM from
# TASK_COMPLETION_RUBRIC and TOOL_SELECTION_RUBRIC above (the published rubric
# text is the methodology — this only reorganizes the two sets of criteria under
# a single shared context, it does not rewrite them). The single JSON response
# nests the two existing per-rubric response shapes under "task_completion" and
# "tool_selection" keys, so each dimension's parsed object is byte-identical to
# what the separate prompts would have returned.

COMBINED_RUBRIC = """\
You are an expert evaluator assessing an AI agent's performance in a multi-turn \
conversation. You will score TWO independent dimensions of the same conversation \
in a single pass: (A) Task Completion and (B) Tool Selection Quality.

## Context
- Domain: {domain}
- User Goals: {user_goals}
- Available Tools: {available_tools}

## Conversation Transcript (with tool calls)
{transcript}

---

# Dimension A: Task Completion

Assess whether the agent successfully completed the user's goals.

## Evaluation Criteria

Score each user goal as COMPLETE, PARTIAL, or FAILED:

1. **COMPLETE**: The agent fully addressed the goal. The correct tools were called, \
correct information was provided, and the user's need was met.
2. **PARTIAL**: The agent made meaningful progress but did not fully resolve the goal. \
For example: correct tool selected but wrong parameters, or correct information \
retrieved but not communicated clearly.
3. **FAILED**: The agent did not address the goal, used wrong tools, provided incorrect \
information, or gave up without resolution.

## Additional Factors (adjust score up or down):
- Did the agent ask appropriate clarifying questions when the request was ambiguous?
- Did the agent recover gracefully from tool errors or unexpected responses?
- Did the agent stay within its authorized scope (no hallucinated capabilities)?
- Did the agent handle multi-step dependencies correctly (e.g., verify identity before account access)?

Scoring: COMPLETE=1.0, PARTIAL=0.5, FAILED=0.0. Overall score is the mean of goal scores, \
adjusted +/- 0.1 for the additional factors above (clamped to [0.0, 1.0]).

---

# Dimension B: Tool Selection Quality

Assess the quality of the agent's tool selection and parameter usage.

## Evaluation Criteria

For each tool call the agent made, evaluate:

1. **Selection Correctness**: Was this the right tool for the user's need at this point \
in the conversation? Consider whether a better tool was available.
2. **Parameter Accuracy**: Were the parameters correct and complete? Were required fields \
provided? Were values accurate based on the conversation context?
3. **Sequencing**: Were tools called in a logical order? Were dependencies respected \
(e.g., lookup before update)?
4. **Necessity**: Was the tool call necessary, or did the agent make redundant/unnecessary calls?
5. **Omissions**: Were there tool calls the agent should have made but didn't?

Scoring: Each correct, necessary tool call with correct parameters scores 1.0. \
Deductions: wrong tool (-1.0), wrong parameters (-0.5), unnecessary call (-0.3), \
missed call (-0.5 each). Normalize to [0.0, 1.0].

---

## Response Format
Return a SINGLE JSON object with both dimensions. Score each dimension \
independently against its own criteria above:
{{
    "task_completion": {{
        "goal_scores": [
            {{"goal": "<goal text>", "score": "COMPLETE|PARTIAL|FAILED", "reasoning": "<brief explanation>"}},
            ...
        ],
        "overall_score": <float 0.0-1.0>,
        "overall_reasoning": "<1-2 sentence summary>"
    }},
    "tool_selection": {{
        "tool_call_scores": [
            {{
                "tool_name": "<name>",
                "turn": <int>,
                "selection_correct": true|false,
                "parameters_correct": true|false,
                "was_necessary": true|false,
                "reasoning": "<brief explanation>"
            }},
            ...
        ],
        "missed_tool_calls": [
            {{"expected_tool": "<name>", "context": "<why it should have been called>"}},
            ...
        ],
        "overall_score": <float 0.0-1.0>,
        "overall_reasoning": "<1-2 sentence summary>"
    }}
}}\
"""


# --- Efficacy: Combined Score ---
# Hybrid Efficacy (schema v0.2): task completion and tool selection are LLM-judge
# dimensions; state verification is the deterministic, judge-independent third.
# When a scenario carries no ground_truth, state verification is inapplicable and
# Efficacy renormalizes to the legacy equal 0.5/0.5 over the two judge dimensions
# (see ``compute_efficacy``).

EFFICACY_WEIGHTS = {
    "task_completion": 0.4,
    "tool_selection": 0.3,
    "state_verification": 0.3,
}

# Fallback weights for legacy scenarios with no state score (no ground_truth).
_LEGACY_EFFICACY_WEIGHTS = {
    "task_completion": 0.5,
    "tool_selection": 0.5,
}


def compute_efficacy(
    task_completion: float, tool_selection: float, state_score: float | None
) -> float:
    """Combine the Efficacy sub-scores, degrading gracefully without state.

    With a state score (scenario has ground_truth)::

        0.4 * task_completion + 0.3 * tool_selection + 0.3 * state_score

    Without one (``state_score is None`` — legacy scenario), the state weight
    is dropped and the two judge dimensions are renormalized to 0.5/0.5::

        0.5 * task_completion + 0.5 * tool_selection

    Pure function — no I/O — so it is unit-tested directly.
    """
    if state_score is None:
        return (
            _LEGACY_EFFICACY_WEIGHTS["task_completion"] * task_completion
            + _LEGACY_EFFICACY_WEIGHTS["tool_selection"] * tool_selection
        )
    return (
        EFFICACY_WEIGHTS["task_completion"] * task_completion
        + EFFICACY_WEIGHTS["tool_selection"] * tool_selection
        + EFFICACY_WEIGHTS["state_verification"] * state_score
    )


# --- Judge Meta-Prompt ---
# Wraps the rubric with judge-specific instructions for consistent output.

JUDGE_SYSTEM_PROMPT = """\
You are a judge in the COT Bench agent evaluation system. Your role is to provide \
accurate, consistent, and well-reasoned evaluations of AI agent performance.

Rules:
1. Be strict but fair. Do not inflate scores.
2. Judge based on what happened in the transcript, not what could have happened.
3. If the transcript is ambiguous, note the ambiguity and score conservatively.
4. Your reasoning must be specific — cite exact turns and tool calls.
5. Respond ONLY with the requested JSON format. No preamble or explanation outside the JSON.
"""


# --- Reliability Scoring ---
# Not judge-based — computed from repeated runs.


def compute_pass_hat_k(n_passes: int, n_runs: int) -> dict:
    """tau-bench style pass^k estimator from a count of passing runs.

    pass^k is the probability that ALL k independent trials of a task succeed
    (contrast pass@k = at least one succeeds); for i.i.d. trials it decays as
    p^k, so it is a far sharper reliability construct than a pass-rate-above-
    threshold. Following tau-bench, we estimate pass^k empirically as the average
    over all C(n, k) size-k subsets of the n collected trials of the indicator
    that every trial in the subset passed. With ``c`` passing runs out of ``n``,
    that average has the closed form::

        pass^k = C(c, k) / C(n, k)

    (the fraction of size-k subsets drawn entirely from the passing runs), which
    is the unbiased estimator of p^k. pass^1 equals the ordinary pass rate.

    Args:
        n_passes: Number of runs that passed (efficacy >= threshold).
        n_runs: Total number of runs.

    Returns:
        ``{k: pass_hat_k for k in 1..n_runs}`` keyed by int k. Empty when there
        are no runs.
    """
    from math import comb

    if n_runs <= 0:
        return {}
    return {k: comb(n_passes, k) / comb(n_runs, k) for k in range(1, n_runs + 1)}


def compute_reliability(run_scores: list[float], threshold: float = 0.7) -> dict:
    """Compute reliability metrics from repeated evaluation runs.

    Args:
        run_scores: Efficacy scores from k repeated runs of the same scenario.
        threshold: Minimum efficacy score to count as a "pass".

    Returns:
        Dict with pass_rate, consistency, score_variance, and ``pass_hat_k`` (the
        tau-bench pass^k estimator for each k in 1..n, published alongside — not
        replacing — pass_rate and consistency).
    """
    if not run_scores:
        return {
            "pass_rate": 0.0,
            "consistency": 0.0,
            "score_variance": 0.0,
            "pass_hat_k": {},
        }

    passes = sum(1 for s in run_scores if s >= threshold)
    mean = sum(run_scores) / len(run_scores)
    variance = sum((s - mean) ** 2 for s in run_scores) / len(run_scores)

    consistency = max(0.0, 1.0 - (max(run_scores) - min(run_scores)))

    return {
        "pass_rate": passes / len(run_scores),
        "consistency": consistency,
        "score_variance": variance,
        "pass_hat_k": compute_pass_hat_k(passes, len(run_scores)),
    }
