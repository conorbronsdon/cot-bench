# Scenario Schema

COT Bench scenarios are JSON files under `data/scenarios/{domain}/`. This document
describes the **v0.2** schema, which adds a ground-truth world state and
deterministic state assertions on top of the original (unversioned) schema.

## Staged migration

Older scenarios that omit `schema_version` remain valid — the validator treats
the v0.2 blocks as optional unless `schema_version == "0.2"`. New and migrated
scenarios should set `schema_version: "0.2"` and supply the new blocks. This lets
the corpus migrate file-by-file without breaking the build.

## Core fields (all schema versions)

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `id` | string | yes | Unique, format `{domain}_{category}_{number}` |
| `category` | string | yes | One of the five categories (see methodology) |
| `persona` | object | yes | `name`, `age`, `occupation`, `personality_traits`, `tone`, `detail_level`, `background` |
| `user_goals` | string[] | yes | 3–10 interconnected goals (typically 5–8) |
| `tools` | object[] | yes | ≥2 tool definitions with `name`, `description`, `parameters`, optional `response_schema`, optional `writes` (top-level state keys the tool may mutate; clamps the tool-sim delta — required on every tool of a `dual_control` scenario, read-only tools use `[]`) |
| `initial_message` | string | yes | In-character opening message; do **not** leak goal text or canonical IDs |
| `difficulty` | string | no | `easy` \| `medium` \| `hard` |
| `expected_tool_sequence` | string[] | no | Advisory ideal tool order (sequence-agnostic once state assertions exist) |

## v0.2 fields

When `schema_version == "0.2"`, the following are **required**:

| Field | Type | Required (v0.2) | Notes |
|-------|------|-----------------|-------|
| `schema_version` | string | yes | `"0.2"` |
| `authorship` | object | yes | Must include `author_model` |
| `ground_truth` | object | yes | Free-form nested world state (canonical IDs, balances, records) |
| `expected_state_changes` | object[] | yes | Assertions over the post-conversation state; may be `[]` |

### `authorship`

```jsonc
{
  "author_model": "human-handwritten",   // required; must NOT be a model under test
  "author_run": "2026-06-12-claude-batch", // optional batch id
  "human_reviewed_by": "Conor Bronsdon",   // optional
  "review_date": "2026-06-20"              // optional
}
```

`author_model` must not match any model id or display name in
`MODELS_UNDER_TEST` — a contestant never writes its own exam. The sentinel
`"human-handwritten"` is always allowed.

### `ground_truth`

A free-form nested dict describing the canonical world the tool simulator must
answer from and mutate. Keys are the **canonical IDs** (e.g. `BUS-CHK-001`); the
`initial_message` and persona stay natural-language ("my business checking"), so
resolving the friendly name to the canonical ID is part of what the agent is
graded on. Do **not** leak canonical IDs into user-visible text.

#### Tool-simulator state deltas (the `__append__` convention)

During a stateful run the tool simulator returns
`{"response": …, "state_delta": …}`. The `state_delta` maps **dotted paths** into
the world to new values, and the runner applies them deterministically
(`eval/simulation/runner.py:apply_state_delta`):

- A plain value at a path **replaces** whatever is there, e.g.
  `{"accounts.BUS-CHK-001.balance": 10920.55}`. Intermediate dicts are created on
  demand, so a delta can set a previously-absent key.
- To **append** to a list, the value must be the explicit form
  `{"__append__": <item>}`, e.g.
  `{"recurring_transfers": {"__append__": {"from": "BUS-CHK-001", "amount": 500}}}`.
  This is the only way to grow a list (creating an empty list first if the key is
  absent) — it keeps list mutation explicit rather than guessing append-vs-replace
  from the value's type.

Invalid paths (descending into a non-dict, `__append__` onto a non-list) are
logged and skipped; a malformed delta never crashes a run.

### `expected_state_changes`

A list of assertions over the world state **after** the conversation. Each
assertion is:

```jsonc
{
  "assert": "accounts.BUS-CHK-001.balance",  // dotted path into ground_truth
  "op": "increased_by",                       // equals | increased_by | decreased_by | contains
  "value": 2500.00,                           // for equals/increased_by/decreased_by
  "goal": "Transfer $2,500 ..."               // optional; must fuzzy-match a user_goal
}
```

Operators:

- `equals` — final value at `assert` path equals `value`.
- `increased_by` / `decreased_by` — final value differs from the initial
  `ground_truth` value by `value`. The dotted path must resolve in `ground_truth`.
- `contains` — the path must resolve to a **list** in `ground_truth`; after the
  conversation that list must contain an entry matching `match` (a partial dict).
  A `<key>_contains` key in `match` means "the entry's `<key>` field contains this
  substring" (e.g. `"reason_contains": "DIGITAL_SVC_LLC"`).

An **empty** `expected_state_changes: []` is meaningful: it asserts that **no
unauthorized mutation** occurred. This is the right value for refusal-heavy
`scope_management` and `adversarial_input_mitigation` scenarios — did the agent
wrongly move money under social-engineering pressure?

For judgment cases (e.g. fee waivers), grade the **action**, not the **outcome**:
assert that a `request_fee_waiver` record was *submitted*, not that the waiver was
approved (approval is the bank's decision, not the agent's).

## `rubric_criteria` (issue #54, adopted)

> **Status: adopted (PR #67).** Every public scenario carries criteria (4–6
> per scenario; the schema floor is 3), as do the private holdout scenarios.
> See `docs/atomic-rubrics.md` for the decision record.

A scenario MAY carry 3–6 atomic, checkable, instance-specific criteria that
inform the **judge** dimensions only (`task_completion` / `tool_selection` —
Cost/Latency/Reliability and the deterministic state check are measured, not
judged). When present, the judge evaluates each criterion as met/unmet with
brief evidence, and that dimension's judge score becomes the weighted fraction
of met criteria (the holistic template score is still recorded in artifacts).
Scenarios without criteria are evaluated byte-identically to today.

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `rubric_criteria` | object[] | no | 3–6 items when present |
| `rubric_criteria[].id` | string | yes | unique within the scenario; short snake_case slug |
| `rubric_criteria[].text` | string | yes | atomic + checkable; ≥15 chars; ONE behavior per criterion |
| `rubric_criteria[].dimension` | string | yes | `task_completion` \| `tool_selection` |
| `rubric_criteria[].weight` | number | no | in `(0, 10]`, default `1.0` |
| `criteria_authorship` | object | with criteria | must include `criteria_author_model` |

### `criteria_authorship`

```jsonc
{
  "criteria_author_model": "anthropic/claude-opus-4.8", // required; the model that ACTUALLY wrote the criteria
  "criteria_author_run": "2026-06-12-claude-opus-criteria-batch", // optional batch id
  "human_reviewed_by": "Conor Bronsdon",  // optional
  "review_date": "2026-06-20"             // optional
}
```

Criteria are usually authored later (and by a different model) than the
scenario, so they carry their own provenance stamp rather than overloading
`authorship`. The same contamination rule applies, family-aware: a model under
test must not write the grading criteria for its own exam. Authoring agents
should stamp via `scripts/generate_data.stamp_criteria()`, which enforces the
guard and re-validates the scenario.

## Banking example

```jsonc
{
  "id": "banking_adaptive_tool_use_0001",
  "category": "adaptive_tool_use",
  "schema_version": "0.2",
  "authorship": {
    "author_model": "human-handwritten",
    "human_reviewed_by": "Conor Bronsdon"
  },
  "persona": { /* ... */ },
  "user_goals": [ /* ... */ ],
  "tools": [ /* ... */ ],
  "initial_message": "Hello, I need to check on my business account ...",
  "difficulty": "medium",
  "expected_tool_sequence": [ /* advisory */ ],

  "ground_truth": {
    "customer": { "customer_id": "CUST-88421", "ssn_last4": "4417", "verified": false },
    "accounts": {
      "BUS-CHK-001": { "type": "checking", "balance": 8420.55, "available": 8120.55 },
      "BUS-SAV-002": { "type": "savings", "balance": 15300.00, "interest_rate_apy": 0.041 }
    },
    "transactions": {
      "BUS-CHK-001": [
        { "id": "TXN-7781", "date": "2026-06-05", "amount": -47.99, "merchant": "DIGITAL_SVC_LLC", "status": "posted" }
        /* ... 4 more so "last 5" is well-defined ... */
      ]
    },
    "recurring_transfers": [],
    "fraud_cases": [],
    "interest_rates": { "savings": { "business": 0.041, "personal": 0.032 } }
  },

  "expected_state_changes": [
    { "assert": "accounts.BUS-CHK-001.balance", "op": "increased_by", "value": 2500.00,
      "goal": "Transfer $2,500 from business savings to business checking to cover upcoming payroll" },
    { "assert": "accounts.BUS-SAV-002.balance", "op": "decreased_by", "value": 2500.00 },
    { "assert": "recurring_transfers", "op": "contains",
      "match": { "from": "BUS-CHK-001", "to": "BUS-SAV-002", "amount": 500, "frequency": "weekly" },
      "goal": "Set up a recurring weekly transfer of $500 from checking to savings starting next Monday" },
    { "assert": "fraud_cases", "op": "contains",
      "match": { "transaction_id": "TXN-7781", "reason_contains": "DIGITAL_SVC_LLC" },
      "goal": "Report a suspicious $47.99 charge from 'DIGITAL_SVC_LLC' she doesn't recognize" }
  ]
}
```

## Validation

`python -m scripts.validate_scenarios` enforces:

- v0.2 conditional requirements (above).
- `authorship.author_model` is not a model under test.
- Every `expected_state_changes` assertion path resolves against `ground_truth`
  (for `increased_by`/`decreased_by`/`equals` the path must exist; for `contains`
  it must resolve to a list).
- Each assertion `goal` (when present) fuzzy-matches (≥0.70) some `user_goal`.
- Cross-scenario dedup within a domain (near-duplicate `initial_message` +
  `user_goals`, and goal-set Jaccard overlap).
- A distribution report (category / difficulty / persona reuse / tool coverage).
  Use `--strict-distribution` to turn band violations into failures.
