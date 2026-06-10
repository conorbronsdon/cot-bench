# Expert Scenario Review — v0.2 Corpus

Review protocol for the 80-scenario corpus (methodology: review **100% of hard scenarios** plus a **25% seeded random sample** of easy/medium — seed 42, reproducible via the snippet at the bottom). One reviewer per scenario; route any scenario any reviewer flags to a second reviewer. Record outcomes in each scenario's `authorship` block (`human_reviewed_by`, `review_date`).

## Reviewer checklist (≤10 items)

For each scenario, score each item pass/fail (or 1-5). A scenario ships when every item passes.

1. **Solvable** — can the goals actually be achieved with the provided `tools`? (Impossible-by-construction goals are allowed only in scope/adversarial scenarios where refusal IS the test.)
2. **Goals interconnected & realistic** — do they read like one real person's session, not a checklist of unrelated tasks?
3. **Persona consistent** — does `initial_message` match the persona's tone and `detail_level`, without quoting goal text verbatim?
4. **Ground truth complete** — every ID/balance/record the goals need exists in `ground_truth`; "last 5 transactions" actually has ≥5.
5. **State assertions correct** — does each `expected_state_changes` entry reflect what a *correct* agent would do (right direction, amount, account/ledger)?
6. **No ID leakage** — canonical IDs (`BUS-CHK-001`, `ACCT-5512`) do NOT appear in user-visible text; the agent must resolve them.
7. **Difficulty honest** — does the label match the trap density (hidden asks, contradictions, lookalike tools)?
8. **Category fit** — does it genuinely exercise its category's failure mode (adversarial = a real manipulation attempt, not just a rude user)?
9. **One genuine trap** — is there ≥1 spot where a careless agent plausibly fails?
10. **Not a near-dupe** — to your knowledge, distinct from other scenarios in the same domain/category.


## Review set (40 scenarios)

### All hard scenarios (26)

| id | category | file |
|----|----------|------|
| banking_adaptive_tool_use_0008 | adaptive_tool_use | `data/scenarios/banking/banking_adaptive_tool_use_0008.json` |
| banking_adaptive_tool_use_0009 | adaptive_tool_use | `data/scenarios/banking/banking_adaptive_tool_use_0009.json` |
| banking_adaptive_tool_use_0010 | adaptive_tool_use | `data/scenarios/banking/banking_adaptive_tool_use_0010.json` |
| banking_adversarial_input_mitigation_0006 | adversarial_input_mitigation | `data/scenarios/banking/banking_adversarial_input_mitigation_0006.json` |
| banking_adversarial_input_mitigation_0007 | adversarial_input_mitigation | `data/scenarios/banking/banking_adversarial_input_mitigation_0007.json` |
| banking_empathetic_resolution_0001 | empathetic_resolution | `data/scenarios/banking/banking_empathetic_resolution_0001.json` |
| banking_empathetic_resolution_0007 | empathetic_resolution | `data/scenarios/banking/banking_empathetic_resolution_0007.json` |
| banking_empathetic_resolution_0008 | empathetic_resolution | `data/scenarios/banking/banking_empathetic_resolution_0008.json` |
| banking_extreme_scenario_recovery_0001 | extreme_scenario_recovery | `data/scenarios/banking/banking_extreme_scenario_recovery_0001.json` |
| banking_extreme_scenario_recovery_0004 | extreme_scenario_recovery | `data/scenarios/banking/banking_extreme_scenario_recovery_0004.json` |
| banking_extreme_scenario_recovery_0005 | extreme_scenario_recovery | `data/scenarios/banking/banking_extreme_scenario_recovery_0005.json` |
| banking_extreme_scenario_recovery_0007 | extreme_scenario_recovery | `data/scenarios/banking/banking_extreme_scenario_recovery_0007.json` |
| banking_scope_management_0008 | scope_management | `data/scenarios/banking/banking_scope_management_0008.json` |
| cs_adaptive_tool_use_0001 | adaptive_tool_use | `data/scenarios/customer_success/cs_adaptive_tool_use_0001.json` |
| cs_adaptive_tool_use_0009 | adaptive_tool_use | `data/scenarios/customer_success/cs_adaptive_tool_use_0009.json` |
| cs_adaptive_tool_use_0010 | adaptive_tool_use | `data/scenarios/customer_success/cs_adaptive_tool_use_0010.json` |
| cs_adversarial_input_mitigation_0006 | adversarial_input_mitigation | `data/scenarios/customer_success/cs_adversarial_input_mitigation_0006.json` |
| cs_adversarial_input_mitigation_0007 | adversarial_input_mitigation | `data/scenarios/customer_success/cs_adversarial_input_mitigation_0007.json` |
| cs_empathetic_resolution_0001 | empathetic_resolution | `data/scenarios/customer_success/cs_empathetic_resolution_0001.json` |
| cs_empathetic_resolution_0003 | empathetic_resolution | `data/scenarios/customer_success/cs_empathetic_resolution_0003.json` |
| cs_empathetic_resolution_0005 | empathetic_resolution | `data/scenarios/customer_success/cs_empathetic_resolution_0005.json` |
| cs_extreme_scenario_recovery_0002 | extreme_scenario_recovery | `data/scenarios/customer_success/cs_extreme_scenario_recovery_0002.json` |
| cs_extreme_scenario_recovery_0003 | extreme_scenario_recovery | `data/scenarios/customer_success/cs_extreme_scenario_recovery_0003.json` |
| cs_extreme_scenario_recovery_0004 | extreme_scenario_recovery | `data/scenarios/customer_success/cs_extreme_scenario_recovery_0004.json` |
| cs_extreme_scenario_recovery_0007 | extreme_scenario_recovery | `data/scenarios/customer_success/cs_extreme_scenario_recovery_0007.json` |
| cs_scope_management_0006 | scope_management | `data/scenarios/customer_success/cs_scope_management_0006.json` |

### Sampled easy/medium (14, seed 42)

| id | category | difficulty | file |
|----|----------|------------|------|
| banking_adaptive_tool_use_0002 | adaptive_tool_use | easy | `data/scenarios/banking/banking_adaptive_tool_use_0002.json` |
| banking_adaptive_tool_use_0006 | adaptive_tool_use | medium | `data/scenarios/banking/banking_adaptive_tool_use_0006.json` |
| banking_adaptive_tool_use_0007 | adaptive_tool_use | medium | `data/scenarios/banking/banking_adaptive_tool_use_0007.json` |
| banking_adversarial_input_mitigation_0001 | adversarial_input_mitigation | easy | `data/scenarios/banking/banking_adversarial_input_mitigation_0001.json` |
| banking_adversarial_input_mitigation_0002 | adversarial_input_mitigation | easy | `data/scenarios/banking/banking_adversarial_input_mitigation_0002.json` |
| banking_empathetic_resolution_0004 | empathetic_resolution | medium | `data/scenarios/banking/banking_empathetic_resolution_0004.json` |
| banking_empathetic_resolution_0005 | empathetic_resolution | medium | `data/scenarios/banking/banking_empathetic_resolution_0005.json` |
| banking_extreme_scenario_recovery_0002 | extreme_scenario_recovery | medium | `data/scenarios/banking/banking_extreme_scenario_recovery_0002.json` |
| cs_adaptive_tool_use_0002 | adaptive_tool_use | easy | `data/scenarios/customer_success/cs_adaptive_tool_use_0002.json` |
| cs_adversarial_input_mitigation_0001 | adversarial_input_mitigation | easy | `data/scenarios/customer_success/cs_adversarial_input_mitigation_0001.json` |
| cs_adversarial_input_mitigation_0004 | adversarial_input_mitigation | medium | `data/scenarios/customer_success/cs_adversarial_input_mitigation_0004.json` |
| cs_empathetic_resolution_0004 | empathetic_resolution | medium | `data/scenarios/customer_success/cs_empathetic_resolution_0004.json` |
| cs_empathetic_resolution_0008 | empathetic_resolution | medium | `data/scenarios/customer_success/cs_empathetic_resolution_0008.json` |
| cs_scope_management_0001 | scope_management | medium | `data/scenarios/customer_success/cs_scope_management_0001.json` |

## Reproducing the sample

```python
import json, glob, random, os
rows = []
for f in sorted(glob.glob('data/scenarios/*/*.json')):
    s = json.load(open(f))
    rows.append((s['id'], s['category'], s['difficulty']))
rest = [r for r in rows if r[2] != 'hard']
sample = sorted(random.Random(42).sample(rest, round(len(rest) * 0.25)))
```

## After review

- Record `human_reviewed_by` + `review_date` in each reviewed scenario's `authorship` block.
- Apply fixes in place; rerun `python -m scripts.validate_scenarios --strict-distribution`.
- A scenario two reviewers reject gets culled and its cell backfilled to keep the distribution matrix.
