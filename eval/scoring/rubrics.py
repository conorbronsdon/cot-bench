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


# --- Efficacy: Combined Score ---
# Task Completion and Tool Selection are weighted equally for the Efficacy dimension.

EFFICACY_WEIGHTS = {
    "task_completion": 0.5,
    "tool_selection": 0.5,
}


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


def compute_reliability(run_scores: list[float], threshold: float = 0.7) -> dict:
    """Compute reliability metrics from repeated evaluation runs.

    Args:
        run_scores: Efficacy scores from k repeated runs of the same scenario.
        threshold: Minimum efficacy score to count as a "pass".

    Returns:
        Dict with pass_rate, consistency, and score_variance.
    """
    if not run_scores:
        return {"pass_rate": 0.0, "consistency": 0.0, "score_variance": 0.0}

    passes = sum(1 for s in run_scores if s >= threshold)
    mean = sum(run_scores) / len(run_scores)
    variance = sum((s - mean) ** 2 for s in run_scores) / len(run_scores)

    consistency = max(0.0, 1.0 - (max(run_scores) - min(run_scores)))

    return {
        "pass_rate": passes / len(run_scores),
        "consistency": consistency,
        "score_variance": variance,
    }
