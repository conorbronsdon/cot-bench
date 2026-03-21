"""Multi-judge orchestration for COT Bench.

Runs each scenario through all configured judges concurrently,
then computes consensus scores and inter-judge agreement.
"""

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache

import anthropic
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


@lru_cache(maxsize=8)
def _get_openai_client(base_url: str, api_key: str) -> OpenAI:
    """Cached OpenAI-compatible client factory."""
    return OpenAI(base_url=base_url, api_key=api_key)


@lru_cache(maxsize=1)
def _get_anthropic_client() -> anthropic.Anthropic:
    """Cached Anthropic client factory."""
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def _parse_judge_response(content: str) -> dict:
    """Parse JSON from judge response, handling markdown code blocks."""
    import re

    # Try direct parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block (```json or bare ```)
    code_block = re.search(r"```(?:json)?\s*\n(.*?)\n\s*```", content, re.DOTALL)
    if code_block:
        try:
            return json.loads(code_block.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Last resort: find first { ... } or [ ... ] boundary
    for start_ch, end_ch in [("{", "}"), ("[", "]")]:
        start = content.find(start_ch)
        end = content.rfind(end_ch)
        if start != -1 and end > start:
            try:
                return json.loads(content[start : end + 1])
            except json.JSONDecodeError:
                continue

    logger.error("Failed to parse judge JSON: %s", content[:200])
    return {
        "overall_score": 0.0,
        "overall_reasoning": "Failed to parse judge response",
    }


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
    start = time.perf_counter()

    if judge.provider == "anthropic":
        client = _get_anthropic_client()
        response = client.messages.create(
            model=judge.model_id,
            system=system_prompt,
            messages=[{"role": "user", "content": rubric_prompt}],
            temperature=judge.temperature,
            max_tokens=judge.max_tokens,
        )
        content = response.content[0].text
    else:
        # MAX and any other OpenAI-compatible providers
        base_url = judge.endpoint or "http://localhost:8000/v1"
        client = _get_openai_client(base_url, "not-needed")
        response = client.chat.completions.create(
            model=judge.model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": rubric_prompt},
            ],
            temperature=judge.temperature,
            max_tokens=judge.max_tokens,
        )
        content = response.choices[0].message.content

    latency_ms = (time.perf_counter() - start) * 1000
    parsed = _parse_judge_response(content)

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
    """Score a scenario with all judges concurrently and compute consensus.

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
    results: list[JudgeResult] = []

    # Run judges concurrently — they're independent API calls
    with ThreadPoolExecutor(max_workers=len(keys)) as executor:
        futures = {
            executor.submit(
                score_with_judge, JUDGES[key], system_prompt, rubric_prompt, rubric_type
            ): key
            for key in keys
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                result = future.result()
                results.append(result)
                logger.info(
                    "Judge %s scored %.2f for %s (%s)",
                    result.judge_name,
                    result.overall_score,
                    scenario_id,
                    rubric_type,
                )
            except Exception:
                logger.exception("Judge %s failed on %s", key, scenario_id)

    if not results:
        logger.warning(
            "All judges failed for %s (%s)", scenario_id, rubric_type
        )
        return ConsensusResult(
            scenario_id=scenario_id,
            rubric_type=rubric_type,
            judge_results=[],
            consensus_score=0.0,
            agreement_rate=0.0,  # No judges = no agreement, not perfect agreement
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
