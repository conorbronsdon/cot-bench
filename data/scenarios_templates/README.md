# Demonstration scenario templates (issue #60)

These are **demonstration** parameterized templates for the anti-memorization
mechanism in `eval/templating.py`. They are NOT part of the evaluated corpus —
they live OUTSIDE `data/scenarios/`, so `scripts/run_eval.py` never loads them.
They exist to (a) make the mechanism concrete and reviewable and (b) give the
test suite real, schema-valid fixtures that exercise the full coherence path
(slots that appear in goals, persona, initial message, ground-truth keys AND
values, state assertions, tool-arg matches, and rubric-criteria text).

Each file is a **converted copy** of a real public scenario, clearly marked:
the `id` carries a `_tmpl` suffix and the `authorship` notes the conversion. The
original scenarios under `data/scenarios/` are untouched. Whether to convert the
real corpus to templates is a separate decision — see
`docs/parameterized-templates.md`.

## How a template differs from a concrete scenario

A template adds a `template_slots` object declaring substitutable surface
values, and replaces those values with `{{slot}}` placeholders everywhere they
appear. Instantiation (`eval.templating.instantiate(template, seed)`) resolves
each slot once from the seed and substitutes it coherently through the whole
scenario, then strips `template_slots`, yielding an ordinary v0.2 scenario.

```
template + seed  --instantiate-->  concrete v0.2 scenario (validator-clean)
```

Same template + same seed → byte-identical scenario. The logical task — goals,
tool sequence, state assertions, criteria semantics — is invariant; only the
surface (which name, which account number, which amount/date) changes per run.

## Files

- `banking_adaptive_tool_use_tmpl.json` — converted from
  `banking_adaptive_tool_use_0000_59bb2918` (banking / adaptive_tool_use).
  Demonstrates the hard case: account IDs used as `ground_truth.accounts` KEYS
  and inside dotted `expected_state_changes` assert paths, plus a transaction ID,
  a merchant drawn from a fixed `choice` set (so the fraud-report criterion still
  matches), names, a customer ID, amounts, and dates.
- `cs_scope_management_tmpl.json` — converted from `cs_scope_management_0001`
  (customer_success / scope_management). Demonstrates a different coherence
  shape: the account ID is a ground-truth VALUE (not a dict key) that also
  appears in `expected_state_changes` `match` blocks and in criteria text; the
  persona first name is rewritten into criteria; and the out-of-scope
  `equals []` assertions (no slots) survive instantiation unchanged while the
  surrounding goal text is rewritten. Shows the mechanism is not banking-shaped.
