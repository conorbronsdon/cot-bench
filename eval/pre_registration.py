"""Pre-registration of evaluation runs for COT Bench.

A pre-registration is only meaningful if it is written *before* the results are
known. This module builds and writes ``pre_registration.json`` — the artifact
that commits a run's definition (models under test, the exact scenario set, the
judge panel, and the seeds/temperatures that apply) to disk **before the first
agent or simulator call**. The companion post-run ``run_manifest.json`` (written
by ``scripts/run_eval.py`` after results are computed) links back to this file by
path and hash, so the pair is auditable: anyone can confirm the run's definition
was fixed ahead of the numbers.

What makes this a *true* pre-registration rather than the old post-hoc manifest:

- **Corpus-level scenario-set hash.** Each scenario ID already embeds an
  8-character content hash, but there was no hash over the *whole set* being run.
  :func:`scenario_set_hash` computes one deterministic sha256 over the canonical
  serialized scenario set (sorted by ID, each scenario re-serialized with sorted
  keys), so the exact corpus a run committed to is tamper-evident as a single
  value — and is independent of on-disk file ordering or whitespace.
- **Judge panel.** The configured judges (names + configured model IDs) are
  recorded. ``resolved_model`` — the model a provider *actually served* — is only
  knowable at call time and stays in the post-run artifacts (see governance §2);
  the pre-registration records the panel as *configured*, which is what the run
  commits to in advance.
- **Seeds / temperatures.** The bootstrap seed, the agent temperature (0.0), and
  the user/tool simulator temperatures are recorded. The honest caveat is kept:
  the user/tool simulators run unseeded at temperature > 0, so runs are not
  bit-for-bit reproducible. The pre-registration records that fact explicitly
  rather than implying a reproducibility it does not have.
- **Judge-prompt mode.** Whether the run uses the combined single-prompt judge
  path or the legacy separate-prompt path.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from eval.simulation.profiles import DEFAULT_SIM_PROFILE

# Filename for the pre-registration artifact, written next to the results
# parquet (alongside the post-run run_manifest.json).
PRE_REGISTRATION_FILENAME = "pre_registration.json"


def canonical_scenario_bytes(scenario_data: dict) -> bytes:
    """Canonically serialize one scenario dict to deterministic bytes.

    Sorted keys, compact separators, UTF-8. The same logical scenario produces
    the same bytes regardless of on-disk key order or whitespace, so the corpus
    hash tracks scenario *content*, not file formatting. Float repr stability is
    guaranteed on CPython (shortest-repr since 3.1), which is what the bench and
    CI run; other interpreters are not a supported audit path.
    """
    return json.dumps(scenario_data, sort_keys=True, separators=(",", ":")).encode("utf-8")


def scenario_set_hash(scenarios_by_domain: dict) -> tuple[str, list[dict]]:
    """Compute a deterministic sha256 over the full scenario set being run.

    ``scenarios_by_domain`` maps a ``Domain`` to its list of loaded ``Scenario``
    objects (the structure ``run_eval.main`` builds). For each scenario the
    canonical JSON of its content is hashed; the per-scenario digests are sorted
    by scenario ID and folded into one corpus digest. Sorting makes the result
    independent of domain iteration order and of the on-disk glob order.

    Returns ``(hex_digest, scenario_index)`` where ``scenario_index`` is a sorted
    list of ``{"domain", "scenario_id", "sha256"}`` entries — recorded in the
    manifest so an auditor can see exactly which scenarios (and which content)
    the corpus hash covers, and can recompute it.

    Any change to any scenario's content changes that scenario's digest and
    therefore the corpus hash; adding or removing a scenario changes the set and
    therefore the hash.
    """
    entries: list[dict] = []
    for domain, scenarios in scenarios_by_domain.items():
        domain_value = getattr(domain, "value", domain)
        for scenario in scenarios:
            data = _scenario_to_canonical_dict(scenario)
            digest = hashlib.sha256(canonical_scenario_bytes(data)).hexdigest()
            entries.append(
                {
                    "domain": domain_value,
                    "scenario_id": scenario.id,
                    "sha256": digest,
                }
            )

    # Sort by (scenario_id) for a stable, order-independent corpus digest.
    entries.sort(key=lambda e: e["scenario_id"])

    corpus = hashlib.sha256()
    for entry in entries:
        # Bind the id to its content digest so reordering or renaming is caught.
        corpus.update(entry["scenario_id"].encode("utf-8"))
        corpus.update(b"\x00")
        corpus.update(entry["sha256"].encode("utf-8"))
        corpus.update(b"\n")
    return corpus.hexdigest(), entries


def _canonical_template_bytes(template_data: dict) -> bytes:
    """Canonical bytes for a RAW template dict (issue #60).

    Identical serialization to :func:`canonical_scenario_bytes` — the difference
    is only what is hashed: the on-disk template JSON *including* its
    ``template_slots`` declaration, before any instantiation. Pinning the template
    corpus this way means "which templates a run used" is tamper-evident
    independently of the per-run instantiation seed.
    """
    return json.dumps(template_data, sort_keys=True, separators=(",", ":")).encode("utf-8")


def template_corpus_hash(templates_by_domain: dict) -> tuple[str, list[dict]]:
    """Deterministic sha256 over the RAW template corpus (issue #60).

    ``templates_by_domain`` maps a domain (or its string value) to a list of raw
    scenario dicts as loaded from disk — templates carry ``template_slots``;
    non-template scenarios are included verbatim, so this hash covers the WHOLE
    authored corpus (the thing in git), not just the templated subset. The fold is
    identical to :func:`scenario_set_hash` (per-scenario digest, sorted by id,
    bound id-to-digest), so the same audit story applies.

    This is the hash a run pre-registers as ``template_corpus`` — invariant to the
    instantiation seed. The companion instantiated-corpus hash (which DOES depend
    on the seed) is :func:`scenario_set_hash` over the instantiated Scenario
    objects.
    """
    entries: list[dict] = []
    for domain, templates in templates_by_domain.items():
        domain_value = getattr(domain, "value", domain)
        for data in templates:
            digest = hashlib.sha256(_canonical_template_bytes(data)).hexdigest()
            entries.append(
                {
                    "domain": domain_value,
                    "scenario_id": data.get("id", ""),
                    "sha256": digest,
                    "templated": bool(data.get("template_slots")),
                }
            )
    entries.sort(key=lambda e: e["scenario_id"])

    corpus = hashlib.sha256()
    for entry in entries:
        corpus.update(entry["scenario_id"].encode("utf-8"))
        corpus.update(b"\x00")
        corpus.update(entry["sha256"].encode("utf-8"))
        corpus.update(b"\n")
    return corpus.hexdigest(), entries


def holdout_set_hash(holdout_by_domain: dict) -> tuple[str, int]:
    """Compute the holdout corpus hash and count WITHOUT revealing its content.

    The private holdout (issue #31) is pre-registered so it is tamper-evident —
    the same set always hashes to the same value, and any change to the holdout
    (a different scenario, an added/removed one) changes the hash — but its
    *content* is never published. So, unlike :func:`scenario_set_hash`, this
    returns only ``(hex_digest, n_scenarios)``: NO scenario IDs and NO
    per-scenario index. The digest is computed identically to the public corpus
    hash (per-scenario canonical sha256, sorted by ID, folded into one corpus
    digest), so the same machinery pins the holdout — the only difference is that
    the breadcrumbs (IDs, index) that would expose the holdout are dropped on the
    floor and never written to disk.

    Hashing still binds each scenario's ID to its content digest internally (so a
    rename or reorder is caught); the ID simply never leaves this function.
    """
    corpus_hash, entries = scenario_set_hash(holdout_by_domain)
    return corpus_hash, len(entries)


def _scenario_to_canonical_dict(scenario) -> dict:
    """Extract the content-bearing fields of a Scenario for hashing.

    Mirrors the fields ``run_eval.load_scenarios`` reads from the scenario JSON,
    so the hash covers exactly the scenario content that drives a run. ``domain``
    is included as its string value. Fields that are ``None`` (e.g. legacy
    scenarios with no ground_truth) are included as ``None`` so their absence is
    itself part of the hashed content.

    ``rubric_criteria`` (issue #54) changes scoring semantics, so it MUST be
    covered by the corpus hash — but it is included CONDITIONALLY (only when
    present), unlike the always-included fields above. Rationale: adding an
    always-``None`` key would silently change every existing scenario digest
    (and therefore the corpus hash) for corpora whose content did not change.
    Conditional inclusion keeps criteria-less scenarios hashing identically to
    before this field existed, while any added/edited/removed criteria still
    change the scenario digest and the corpus hash.
    """
    data = {
        "id": scenario.id,
        "domain": getattr(scenario.domain, "value", scenario.domain),
        "persona": scenario.persona,
        "user_goals": scenario.user_goals,
        "tools": scenario.tools,
        "category": scenario.category,
        "initial_message": scenario.initial_message,
        "ground_truth": scenario.ground_truth,
        "expected_state_changes": scenario.expected_state_changes,
    }
    rubric_criteria = getattr(scenario, "rubric_criteria", None)
    if rubric_criteria:
        data["rubric_criteria"] = rubric_criteria
    # Dual control (issue #58) changes what a run DOES (the simulated user also
    # acts on the shared world), so it MUST be covered by the corpus hash when
    # present — but included CONDITIONALLY, exactly like rubric_criteria above,
    # so single-control scenarios (the entire public corpus today) hash
    # identically to before this field existed. Serialized back to the same plain
    # dict shape the scenario JSON carries (user_tools + user_actions) so the
    # canonical bytes match the on-disk content regardless of whether the
    # Scenario holds a DualControl object or the loader passed a raw dict.
    dual_control = getattr(scenario, "dual_control", None)
    if dual_control is not None:
        data["dual_control"] = _dual_control_to_dict(dual_control)
    # Recovery probe (issue #57) changes what a run DOES (it injects a scripted
    # perturbation and grades recovery), so it MUST be covered by the corpus hash
    # when present — but included CONDITIONALLY, exactly like rubric_criteria
    # above, so probe-less scenarios (the entire public corpus today) hash
    # identically to before this field existed. Serialized back to the same plain
    # dict shape the scenario JSON carries (turn/kind/injection/recovery_assertions)
    # so the canonical bytes match the on-disk content regardless of whether the
    # Scenario holds a RecoveryProbe object or the loader passed a raw dict.
    recovery_probe = getattr(scenario, "recovery_probe", None)
    if recovery_probe is not None:
        data["recovery_probe"] = _recovery_probe_to_dict(recovery_probe)
    return data


def _dual_control_to_dict(dc) -> dict:
    """Canonical plain-dict form of a dual-control block for hashing (issue #58).

    Accepts either a ``DualControl`` object (the loaded form) or an already-raw
    dict (defensive), and returns the two content lists — ``user_tools`` (each
    with name/description/parameters/scope) and ``user_actions`` (each with
    tool/trigger/trigger_value/arguments/state_delta/user_message). Empty
    optional fields are normalized (``arguments``/``state_delta`` -> ``{}``,
    ``scope``/``parameters`` -> ``[]``, ``user_message`` -> ``None``) to match the
    object's own normalization, so a block authored with an omitted optional
    hashes the same as one with the explicit empty value.
    """
    if isinstance(dc, dict):
        tools = dc.get("user_tools", [])
        actions = dc.get("user_actions", [])
        return {
            "user_tools": [
                {
                    "name": t.get("name"),
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters") or [],
                    "scope": t.get("scope") or [],
                }
                for t in tools
            ],
            "user_actions": [
                {
                    "tool": a.get("tool"),
                    "trigger": a.get("trigger"),
                    "trigger_value": a.get("trigger_value"),
                    "arguments": a.get("arguments") or {},
                    "state_delta": a.get("state_delta") or {},
                    "user_message": a.get("user_message"),
                }
                for a in actions
            ],
        }
    return {
        "user_tools": [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
                "scope": t.scope,
            }
            for t in dc.user_tools.values()
        ],
        "user_actions": [
            {
                "tool": a.tool,
                "trigger": a.trigger,
                "trigger_value": a.trigger_value,
                "arguments": a.arguments,
                "state_delta": a.state_delta,
                "user_message": a.user_message,
            }
            for a in dc.user_actions
        ],
    }


def _recovery_probe_to_dict(probe) -> dict:
    """Canonical plain-dict form of a recovery probe for hashing (issue #57).

    Accepts either a ``RecoveryProbe`` object (the loaded form) or an already-raw
    dict (defensive), and returns the four content fields. ``recovery_assertions``
    defaults to ``[]`` to match RecoveryProbe's own normalization, so a probe with
    no extra assertions hashes the same whether it was authored with an empty
    list or with the key omitted.
    """
    if isinstance(probe, dict):
        return {
            "turn": probe.get("turn"),
            "kind": probe.get("kind"),
            "injection": probe.get("injection"),
            "recovery_assertions": probe.get("recovery_assertions") or [],
        }
    return {
        "turn": probe.turn,
        "kind": probe.kind,
        "injection": probe.injection,
        "recovery_assertions": probe.recovery_assertions,
    }


def build_pre_registration(
    *,
    run_id: str,
    models,
    scenarios_by_domain: dict,
    judges: dict,
    judge_keys,
    reliability_runs: int,
    holdout_by_domain: dict | None = None,
    bootstrap_seed: int,
    agent_temperature: float,
    user_simulator_temperature: float,
    tool_simulator_temperature: float,
    separate_judge_calls: bool,
    user_simulator_model: str | None = None,
    tool_simulator_model: str | None = None,
    user_sim_profile: str = DEFAULT_SIM_PROFILE,
    templates_by_domain: dict | None = None,
    instantiation_seed: int | None = None,
    artifacts_dir=None,
    trace_dir=None,
) -> dict:
    """Assemble the pre-registration record (pure — no I/O).

    Captures the run's definition as committed *before* any model call:

    - ``run_id`` and ``timestamp`` (UTC, when the pre-registration is built).
    - ``models_under_test`` — the requested models (name + model_id + provider).
      "Requested" only: which models *complete* is a post-run fact and lives in
      run_manifest.json.
    - ``domains`` and ``scenario_set`` — domains, per-domain scenario IDs, and the
      corpus-level ``sha256`` over the canonical serialized PUBLIC scenario set.
    - ``holdout_set`` — present only when ``holdout_by_domain`` is supplied (issue
      #31). It records the holdout corpus ``sha256`` and ``n_scenarios`` ONLY —
      deliberately NO scenario IDs and NO per-scenario index, unlike
      ``scenario_set``. This pins the private holdout (tamper-evident: any change
      to the held-out scenarios changes the hash) without revealing what is in it,
      so the holdout's content never lands in a published artifact.
    - ``judge_panel`` — judges as configured (name + configured model_id +
      provider + temperature). ``resolved_model`` is intentionally absent: it is
      only knowable at call time and is recorded in the post-run artifacts.
    - ``reliability_runs`` and ``seeds_and_temperatures`` (bootstrap seed, agent
      temp, simulator temps, the requested simulator model ids — issue #50 — AND
      the behavioral user-sim profile — issue #59) with an explicit note that the
      user/tool simulators are unseeded at temp > 0, so runs are not bit-for-bit
      reproducible.
    - ``judge_prompt_mode`` — "combined" (default) or "separate".
    - ``templating`` — present only when ``templates_by_domain`` is supplied (issue
      #60). Records the raw-template corpus hash (seed-invariant), the
      ``instantiation_seed``, and the instantiated-corpus hash (== the
      ``scenario_set`` hash, since the scenarios passed in are already
      instantiated). ``None`` for runs with no templated scenarios.
    """
    corpus_hash, scenario_index = scenario_set_hash(scenarios_by_domain)

    # Templating (issue #60). ``scenarios_by_domain`` already holds INSTANTIATED
    # scenarios, so ``corpus_hash`` above IS the instantiated-corpus hash (the
    # surface the run actually used). When templates were loaded we ALSO record
    # the raw-template corpus hash (seed-invariant) and the instantiation seed, so
    # a run with templated scenarios pre-registers all three — template content,
    # the seed, and the exact instantiated surface — making the surface
    # tamper-evident AND recomputable from (templates, seed). Omitted entirely
    # when no templates were loaded, so non-templated runs are unchanged.
    templating_block = None
    if templates_by_domain is not None:
        tmpl_hash, tmpl_index = template_corpus_hash(templates_by_domain)
        n_templated = sum(1 for e in tmpl_index if e["templated"])
    else:
        n_templated = 0
    # Only emit the block when at least one scenario is actually a template, so a
    # run over the current (non-templated) corpus pre-registers EXACTLY as today —
    # no spurious templating block, no behavior change. The seed is meaningless
    # when nothing is templated.
    if templates_by_domain is not None and n_templated > 0:
        templating_block = {
            "instantiation_seed": instantiation_seed,
            "n_templated_scenarios": n_templated,
            "template_corpus": {
                "sha256": tmpl_hash,
                "n_scenarios": len(tmpl_index),
                "template_index": tmpl_index,
            },
            "instantiated_corpus_sha256": corpus_hash,
            "note": (
                "Anti-memorization templating (issue #60). 'template_corpus.sha256' "
                "pins the raw authored templates (seed-invariant). "
                "'instantiated_corpus_sha256' (== scenario_set.sha256) pins the "
                "concrete surface this run used, which is a deterministic function "
                "of the templates and 'instantiation_seed'. Re-running with the same "
                "seed reproduces the same surface byte-for-byte."
            ),
        }

    scenario_ids_by_domain: dict[str, list[str]] = {}
    for domain, scenarios in scenarios_by_domain.items():
        domain_value = getattr(domain, "value", domain)
        scenario_ids_by_domain[domain_value] = sorted(s.id for s in scenarios)

    judge_panel = [
        {
            "key": key,
            "name": judges[key].name,
            "configured_model_id": judges[key].model_id,
            "provider": judges[key].provider,
            "temperature": judges[key].temperature,
        }
        for key in judge_keys
    ]

    # Holdout set (issue #31): hash + count ONLY. No IDs, no index — the holdout
    # is pinned without being revealed. Omitted entirely when no holdout was run.
    holdout_set = None
    if holdout_by_domain:
        holdout_hash, holdout_n = holdout_set_hash(holdout_by_domain)
        holdout_set = {
            "sha256": holdout_hash,
            "n_scenarios": holdout_n,
            "privacy_note": (
                "Private holdout (issue #31). Only the corpus sha256 and count are "
                "recorded — deliberately no scenario IDs and no per-scenario index. "
                "The hash pins the held-out set (any change to it changes the hash) "
                "without revealing its content, which is authored and stored outside "
                "this repository and never published."
            ),
        }

    return {
        "artifact_type": "pre_registration",
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models_under_test": [
            {
                "name": m["name"],
                "model_id": m["model_id"],
                "provider": m["provider"],
            }
            for m in models
        ],
        "domains": [getattr(d, "value", d) for d in scenarios_by_domain],
        "scenario_set": {
            "sha256": corpus_hash,
            "n_scenarios": len(scenario_index),
            "scenario_ids_by_domain": scenario_ids_by_domain,
            "scenario_index": scenario_index,
        },
        "holdout_set": holdout_set,
        "templating": templating_block,
        "judge_panel": {
            "judges": judge_panel,
            "resolved_model_note": (
                "Only configured model IDs are pre-registered. The model a "
                "provider actually served (resolved_model) is knowable only at "
                "call time and is recorded per call in the post-run artifacts "
                "(see governance.md §2)."
            ),
        },
        "reliability_runs": reliability_runs,
        "seeds_and_temperatures": {
            "bootstrap_seed": bootstrap_seed,
            "agent_temperature": agent_temperature,
            "user_simulator_temperature": user_simulator_temperature,
            "tool_simulator_temperature": tool_simulator_temperature,
            # Simulator MODEL identity is part of the run definition (issue #50):
            # the user/tool simulators can be overridden per run for the
            # sensitivity test, so the requested model ids are pre-registered
            # alongside their temperatures. None falls back to the configured
            # defaults at the call site; they are recorded here as resolved.
            "user_simulator_model": user_simulator_model,
            "tool_simulator_model": tool_simulator_model,
            # Behavioral user-sim profile (issue #59 part 1) is part of the run
            # definition exactly like the sim model ids above: a non-cooperative
            # profile changes the agent's scoring conditions, so the run must
            # commit to it before any number is known. Rows it produces are
            # excluded from the public leaderboard aggregates.
            "user_sim_profile": user_sim_profile,
            "reproducibility_note": (
                "The agent under test runs at temperature 0.0 and the bootstrap "
                "uses a fixed seed for reproducible confidence intervals. The "
                "user and tool simulators run unseeded; the user simulator runs "
                "at temperature > 0. Runs are therefore NOT bit-for-bit "
                "reproducible."
            ),
        },
        "judge_prompt_mode": "separate" if separate_judge_calls else "combined",
        "artifacts_dir": str(artifacts_dir) if artifacts_dir is not None else None,
        "trace_dir": str(trace_dir) if trace_dir is not None else None,
    }


def write_pre_registration(results_dir, pre_registration: dict) -> Path:
    """Write the pre-registration record to disk and return its path.

    Called from ``run_eval.main`` BEFORE the evaluation loop starts (before any
    agent/simulator/judge call), which is what makes it a real pre-registration
    rather than an after-the-fact record.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / PRE_REGISTRATION_FILENAME
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pre_registration, f, indent=2)
    return path


def file_sha256(path) -> str:
    """sha256 over a file's bytes — used to link the post-run record to the
    exact pre-registration file that was written."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
