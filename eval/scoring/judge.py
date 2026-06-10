"""Multi-judge orchestration for COT Bench.

Runs each scenario through all configured judges concurrently,
then computes consensus scores and inter-judge agreement.
"""

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import lru_cache

import anthropic
from openai import OpenAI

from eval.config import JUDGES, JudgeConfig

logger = logging.getLogger(__name__)


@dataclass
class JudgeResult:
    """Result from a single judge evaluating a single scenario.

    ``parse_failed`` is True when the judge returned content we could not
    recover JSON from (even after one retry). Such results are kept in
    ``ConsensusResult.judge_results`` for transparency but are EXCLUDED from
    consensus/agreement math — a parse failure is not a genuine 0.0 grade.
    """

    judge_name: str
    rubric_type: str  # "task_completion" or "tool_selection"
    overall_score: float
    reasoning: str
    raw_response: dict
    latency_ms: float
    parse_failed: bool = False


@dataclass
class ConsensusResult:
    """Aggregated result across all judges for a single scenario.

    Consensus math (``consensus_score``, ``agreement_rate``,
    ``max_disagreement``) is computed ONLY from valid judges — those that both
    returned successfully and parsed. The accounting fields make the size and
    health of the panel explicit per row:

    - ``n_judges_requested``: how many judges we asked.
    - ``n_judges_valid``: how many produced a usable (scored AND parsed) score.
    - ``parse_failures``: names of judges whose output could not be parsed.
    - ``api_failures``: names of judges whose API call raised.
    - ``degraded``: True when fewer than 2 valid judges remain despite 2+ being
      requested. Consensus is still computed from what's valid, but flagged.

    ``agreement_rate`` and ``max_disagreement`` are ``None`` when fewer than 2
    valid judge scores exist (agreement is undefined for a single grader).
    """

    scenario_id: str
    rubric_type: str
    judge_results: list[JudgeResult]
    consensus_score: float
    agreement_rate: float | None
    max_disagreement: float | None
    n_judges_requested: int = 0
    n_judges_valid: int = 0
    parse_failures: list[str] = field(default_factory=list)
    api_failures: list[str] = field(default_factory=list)
    degraded: bool = False


@lru_cache(maxsize=8)
def _get_openai_client(base_url: str, api_key: str) -> OpenAI:
    """Cached OpenAI-compatible client factory."""
    return OpenAI(base_url=base_url, api_key=api_key)


@lru_cache(maxsize=1)
def _get_anthropic_client() -> anthropic.Anthropic:
    """Cached Anthropic client factory."""
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def _parse_judge_response(content: str) -> dict | None:
    """Parse JSON from judge response, handling markdown code blocks.

    Returns the parsed dict on success, or ``None`` when no JSON can be
    recovered. Returning ``None`` (rather than a synthetic ``0.0`` result)
    lets callers distinguish an unparseable response — which often means a
    truncated/malformed generation — from a genuine 0.0 grade. A fabricated
    0.0 would silently drag the consensus and crater the agreement rate.
    """
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
    return None


def _call_judge_api(
    judge: JudgeConfig,
    system_prompt: str,
    rubric_prompt: str,
) -> str:
    """Make a single judge API call and return the raw text content.

    Separated from :func:`score_with_judge` so the orchestration layer can
    retry it (a fresh API call) without re-implementing provider dispatch.
    """
    if judge.provider == "anthropic":
        client = _get_anthropic_client()
        response = client.messages.create(
            model=judge.model_id,
            system=system_prompt,
            messages=[{"role": "user", "content": rubric_prompt}],
            temperature=judge.temperature,
            max_tokens=judge.max_tokens,
        )
        return response.content[0].text
    elif judge.provider == "openrouter":
        # Open-weight judges via OpenRouter (OpenAI-compatible). Needs a real
        # key, unlike a self-hosted local endpoint.
        base_url = judge.endpoint or "https://openrouter.ai/api/v1"
        client = _get_openai_client(base_url, os.environ.get("OPENROUTER_API_KEY", ""))
        response = client.chat.completions.create(
            model=judge.model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": rubric_prompt},
            ],
            temperature=judge.temperature,
            max_tokens=judge.max_tokens,
        )
        return response.choices[0].message.content
    else:
        raise ValueError(
            f"Unknown judge provider {judge.provider!r} for {judge.name}. "
            "Expected 'anthropic' or 'openrouter'."
        )


def score_with_judge(
    judge: JudgeConfig,
    system_prompt: str,
    rubric_prompt: str,
    rubric_type: str,
) -> JudgeResult:
    """Score a scenario using a single judge.

    On a parse failure, the judge is called ONCE more (a fresh API call) —
    transient truncation/format glitches are common and a second sample often
    parses cleanly. If parsing still fails, a JudgeResult flagged
    ``parse_failed=True`` is returned (overall_score 0.0 as a placeholder),
    which downstream consensus excludes rather than treating as a real grade.

    Args:
        judge: Judge model configuration.
        system_prompt: The judge system prompt.
        rubric_prompt: The filled-in rubric with transcript and context.
        rubric_type: "task_completion" or "tool_selection".

    Returns:
        JudgeResult with score and reasoning, or a parse_failed placeholder.
    """
    start = time.perf_counter()

    parsed = None
    attempts = 2  # initial call + one retry on parse failure
    for attempt in range(attempts):
        content = _call_judge_api(judge, system_prompt, rubric_prompt)
        parsed = _parse_judge_response(content)
        if parsed is not None:
            break
        if attempt + 1 < attempts:
            logger.warning(
                "Judge %s parse failed (attempt %d/%d) — retrying",
                judge.name,
                attempt + 1,
                attempts,
            )

    latency_ms = (time.perf_counter() - start) * 1000

    if parsed is None:
        logger.error("Judge %s parse failed after %d attempts", judge.name, attempts)
        return JudgeResult(
            judge_name=judge.name,
            rubric_type=rubric_type,
            overall_score=0.0,
            reasoning="Failed to parse judge response",
            raw_response={},
            latency_ms=latency_ms,
            parse_failed=True,
        )

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
    n_requested = len(keys)
    results: list[JudgeResult] = []
    api_failures: list[str] = []

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
                    "Judge %s %s for %s (%s)",
                    result.judge_name,
                    "PARSE-FAILED" if result.parse_failed else f"scored {result.overall_score:.2f}",
                    scenario_id,
                    rubric_type,
                )
            except Exception:
                # API-level failure (raised exception). Record the judge name
                # so the panel shrinkage is visible per row rather than being
                # silently absorbed into a smaller-N mean.
                logger.exception("Judge %s failed on %s", key, scenario_id)
                api_failures.append(JUDGES[key].name)

    parse_failures = [r.judge_name for r in results if r.parse_failed]
    valid = [r for r in results if not r.parse_failed]
    n_valid = len(valid)
    # Degraded: we asked for a real panel (2+) but fewer than 2 usable scores
    # survived. Consensus is still reported from what's valid, but flagged.
    degraded = n_requested >= 2 and n_valid < 2

    if not valid:
        logger.warning(
            "No valid judge scores for %s (%s): %d parse-failed, %d api-failed",
            scenario_id,
            rubric_type,
            len(parse_failures),
            len(api_failures),
        )
        return ConsensusResult(
            scenario_id=scenario_id,
            rubric_type=rubric_type,
            judge_results=results,  # keep failed results for transparency
            consensus_score=0.0,
            agreement_rate=None,  # no valid judges = undefined, not perfect
            max_disagreement=None,
            n_judges_requested=n_requested,
            n_judges_valid=0,
            parse_failures=parse_failures,
            api_failures=api_failures,
            degraded=degraded,
        )

    scores = [r.overall_score for r in valid]
    consensus = sum(scores) / len(scores)

    # Agreement is undefined with a single grader — don't report a lone judge
    # as "perfect agreement" (the old pairs==0 -> 1.0 degeneracy).
    if len(scores) < 2:
        agreement_rate = None
        max_disagreement = None
    else:
        max_disagreement = max(scores) - min(scores)
        # Agreement rate: fraction of judge pairs within 0.2 of each other
        pairs = 0
        agreements = 0
        for i in range(len(scores)):
            for j in range(i + 1, len(scores)):
                pairs += 1
                if abs(scores[i] - scores[j]) <= 0.2:
                    agreements += 1
        agreement_rate = agreements / pairs

    return ConsensusResult(
        scenario_id=scenario_id,
        rubric_type=rubric_type,
        judge_results=results,  # includes parse-failed for transparency
        consensus_score=consensus,
        agreement_rate=agreement_rate,
        max_disagreement=max_disagreement,
        n_judges_requested=n_requested,
        n_judges_valid=n_valid,
        parse_failures=parse_failures,
        api_failures=api_failures,
        degraded=degraded,
    )
