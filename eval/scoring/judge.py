"""Multi-judge orchestration for COT Bench.

Runs each scenario through all configured judges independently,
then computes consensus scores and inter-judge agreement.
"""

import json
import logging
import time
from dataclasses import dataclass

from openai import OpenAI

from eval.config import JUDGES, JudgeConfig

logger = logging.getLogger(__name__)


@dataclass
class JudgeResult:
    """Result from a single judge evaluating a single scenario."""

    judge_name: str
    rubric_type: str  # "task_completion" or "tool_selection"
    overall_score: float
    reasoning: str
    raw_response: dict
    latency_ms: float


@dataclass
class ConsensusResult:
    """Aggregated result across all judges for a single scenario."""

    scenario_id: str
    rubric_type: str
    judge_results: list[JudgeResult]
    consensus_score: float
    agreement_rate: float
    max_disagreement: float


def _build_client(judge: JudgeConfig) -> OpenAI:
    """Create an OpenAI-compatible client for a judge model."""
    if judge.provider == "max":
        return OpenAI(base_url=judge.endpoint, api_key="not-needed")
    elif judge.provider == "anthropic":
        # Use Anthropic's OpenAI-compatible endpoint
        import os

        return OpenAI(
            base_url="https://api.anthropic.com/v1/",
            api_key=os.environ["ANTHROPIC_API_KEY"],
            default_headers={"anthropic-version": "2023-06-01"},
        )
    else:
        raise ValueError(f"Unknown judge provider: {judge.provider}")


def score_with_judge(
    judge: JudgeConfig,
    system_prompt: str,
    rubric_prompt: str,
    rubric_type: str,
) -> JudgeResult:
    """Score a scenario using a single judge.

    Args:
        judge: Judge model configuration.
        system_prompt: The judge system prompt.
        rubric_prompt: The filled-in rubric with transcript and context.
        rubric_type: "task_completion" or "tool_selection".

    Returns:
        JudgeResult with score and reasoning.
    """
    client = _build_client(judge)

    start = time.perf_counter()
    response = client.chat.completions.create(
        model=judge.model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": rubric_prompt},
        ],
        temperature=judge.temperature,
        max_tokens=judge.max_tokens,
    )
    latency_ms = (time.perf_counter() - start) * 1000

    content = response.choices[0].message.content
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # Try extracting JSON from markdown code block
        if "```" in content:
            json_str = content.split("```json")[-1].split("```")[0].strip()
            if not json_str:
                json_str = content.split("```")[-2].strip()
            parsed = json.loads(json_str)
        else:
            logger.error("Judge %s returned non-JSON: %s", judge.name, content[:200])
            parsed = {"overall_score": 0.0, "overall_reasoning": "Failed to parse judge response"}

    return JudgeResult(
        judge_name=judge.name,
        rubric_type=rubric_type,
        overall_score=float(parsed.get("overall_score", 0.0)),
        reasoning=parsed.get("overall_reasoning", ""),
        raw_response=parsed,
        latency_ms=latency_ms,
    )


def score_with_all_judges(
    system_prompt: str,
    rubric_prompt: str,
    rubric_type: str,
    scenario_id: str,
    judge_keys: list[str] | None = None,
) -> ConsensusResult:
    """Score a scenario with all judges and compute consensus.

    Args:
        system_prompt: The judge system prompt.
        rubric_prompt: The filled-in rubric with transcript and context.
        rubric_type: "task_completion" or "tool_selection".
        scenario_id: Unique identifier for the scenario being judged.
        judge_keys: Which judges to use (defaults to all).

    Returns:
        ConsensusResult with individual and aggregated scores.
    """
    keys = judge_keys or list(JUDGES.keys())
    results = []

    for key in keys:
        judge = JUDGES[key]
        try:
            result = score_with_judge(judge, system_prompt, rubric_prompt, rubric_type)
            results.append(result)
            logger.info(
                "Judge %s scored %.2f for %s (%s)",
                judge.name,
                result.overall_score,
                scenario_id,
                rubric_type,
            )
        except Exception:
            logger.exception("Judge %s failed on %s", judge.name, scenario_id)

    if not results:
        return ConsensusResult(
            scenario_id=scenario_id,
            rubric_type=rubric_type,
            judge_results=[],
            consensus_score=0.0,
            agreement_rate=0.0,
            max_disagreement=0.0,
        )

    scores = [r.overall_score for r in results]
    consensus = sum(scores) / len(scores)
    max_disagreement = max(scores) - min(scores)

    # Agreement rate: fraction of judge pairs within 0.2 of each other
    pairs = 0
    agreements = 0
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            pairs += 1
            if abs(scores[i] - scores[j]) <= 0.2:
                agreements += 1
    agreement_rate = agreements / pairs if pairs > 0 else 1.0

    return ConsensusResult(
        scenario_id=scenario_id,
        rubric_type=rubric_type,
        judge_results=results,
        consensus_score=consensus,
        agreement_rate=agreement_rate,
        max_disagreement=max_disagreement,
    )
