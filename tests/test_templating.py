"""Tests for parameterized scenario templates (issue #60).

Covers the five load-bearing invariants of the staged design (see
``eval/templating.py`` and ``docs/parameterized-templates.md``):

1. **Deterministic instantiation** — same template + seed -> byte-identical
   scenario; a different seed changes the surface.
2. **Coherence** — a slot resolves once and is substituted consistently across
   user_goals, persona, initial_message, ground_truth (values AND dict keys /
   dotted assert paths), expected_state_changes match values, AND
   rubric_criteria text.
3. **Validator-clean output** — the instantiated scenario passes
   ``validate_scenario_dict`` and the demonstration template validates.
4. **Pre-registration honesty** — a templated run pre-registers the
   template-corpus hash (seed-invariant), the instantiation seed, and the
   instantiated-corpus hash; re-instantiating with the same seed reproduces the
   instantiated hash, a different seed changes it.
5. **Backwards compatible** — a non-template scenario instantiates to itself
   (minus the absent declaration) and a criteria-less corpus hash is unchanged.

All tests are deterministic and offline (no network, no model calls).
"""

import json
from pathlib import Path

import pytest

from eval.config import Domain
from eval.pre_registration import (
    scenario_set_hash,
    template_corpus_hash,
)
from eval.simulation.runner import Scenario
from eval.templating import (
    DEFAULT_INSTANTIATION_SEED,
    find_placeholders,
    instantiate,
    is_template,
    resolve_slots,
    substitute,
)
from scripts.validate_scenarios import validate_scenario_dict

TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "data" / "scenarios_templates"
TEMPLATE_PATH = TEMPLATES_DIR / "banking_adaptive_tool_use_tmpl.json"
CS_TEMPLATE_PATH = TEMPLATES_DIR / "cs_scope_management_tmpl.json"

# Every demonstration template on disk, so the validator/determinism/coherence
# invariants are asserted over the whole fixture set rather than one file. Add a
# new template under data/scenarios_templates/ and it is picked up automatically.
ALL_TEMPLATE_PATHS = sorted(TEMPLATES_DIR.glob("*.json"))


@pytest.fixture
def demo_template() -> dict:
    return json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def cs_template() -> dict:
    return json.loads(CS_TEMPLATE_PATH.read_text(encoding="utf-8"))


# --- 1. Deterministic instantiation -----------------------------------------


def test_same_template_same_seed_is_byte_identical(demo_template):
    a = instantiate(demo_template, 12345)
    b = instantiate(demo_template, 12345)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_different_seed_changes_surface(demo_template):
    a = instantiate(demo_template, 1)
    b = instantiate(demo_template, 2)
    assert json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True)
    # ...but the logical SHAPE is unchanged: same goals count, same tool set,
    # same assertion ops/paths-structure, same criteria ids.
    assert len(a["user_goals"]) == len(b["user_goals"])
    assert [t["name"] for t in a["tools"]] == [t["name"] for t in b["tools"]]
    assert [c["id"] for c in a["rubric_criteria"]] == [c["id"] for c in b["rubric_criteria"]]
    assert [c["dimension"] for c in a["rubric_criteria"]] == [
        c["dimension"] for c in b["rubric_criteria"]
    ]


def test_instantiate_does_not_mutate_input(demo_template):
    before = json.dumps(demo_template, sort_keys=True)
    instantiate(demo_template, 99)
    assert json.dumps(demo_template, sort_keys=True) == before


def test_resolve_slots_is_seed_deterministic_and_order_independent():
    specs = {
        "b": {"type": "digits", "length": 4},
        "a": {"type": "first_name"},
    }
    r1 = resolve_slots("sid", specs, 5)
    r2 = resolve_slots("sid", dict(reversed(list(specs.items()))), 5)
    assert r1 == r2  # declaration order does not change drawn values


# --- 2. Coherence ------------------------------------------------------------


def test_account_id_coherent_across_keys_paths_goals_and_criteria(demo_template):
    inst = instantiate(demo_template, 2026)
    gt = inst["ground_truth"]
    acct_keys = list(gt["accounts"].keys())
    chk = next(k for k in acct_keys if "CHK" in k)
    sav = next(k for k in acct_keys if "SAV" in k)
    chk_num = chk.rsplit("-", 1)[1]
    sav_num = sav.rsplit("-", 1)[1]

    # The checking key also names the transactions + pending-deposits buckets.
    assert chk in gt["transactions"]
    assert chk in gt["pending_deposits"]

    # The transfer assert path names the SAME account key.
    transfer_assert = inst["expected_state_changes"][0]["assert"]
    assert transfer_assert == f"accounts.{chk}.balance"
    assert inst["expected_state_changes"][1]["assert"] == f"accounts.{sav}.balance"

    # The recurring-transfer match names both account ids.
    recurring_match = inst["expected_state_changes"][2]["match"]
    assert recurring_match["from_account_id"] == sav
    assert recurring_match["to_account_id"] == chk

    # The customer-facing numbers in the goals match the account-id digits.
    assert f"checking account {chk_num}" in inst["user_goals"][3]
    assert f"savings account {sav_num}" in inst["user_goals"][3]

    # And the criterion text references the SAME digits.
    onetime_crit = next(c for c in inst["rubric_criteria"] if c["id"] == "one_time_transfer")
    assert f"savings {sav_num}" in onetime_crit["text"]
    assert f"checking {chk_num}" in onetime_crit["text"]


def test_amount_substitutes_as_number_not_string(demo_template):
    inst = instantiate(demo_template, 4)
    # increased_by value must be numeric for the float-tolerant state check.
    val = inst["expected_state_changes"][0]["value"]
    assert isinstance(val, (int, float))
    # The two transfer assertions move the same amount.
    assert inst["expected_state_changes"][1]["value"] == val
    # The recurring-transfer match amount is numeric too.
    assert isinstance(inst["expected_state_changes"][2]["match"]["amount"], (int, float))


def test_fraud_merchant_and_txn_coherent(demo_template):
    inst = instantiate(demo_template, 808)
    chk = next(k for k in inst["ground_truth"]["accounts"] if "CHK" in k)
    txns = inst["ground_truth"]["transactions"][chk]
    fraud_txn = txns[0]
    merchant = fraud_txn["merchant"]
    txn_id = fraud_txn["id"]
    # The fraud-report criterion references the same merchant and txn id.
    crit = next(c for c in inst["rubric_criteria"] if c["id"] == "fraud_report_correct_txn")
    assert merchant in crit["text"]
    assert txn_id in crit["text"]
    # ...as does the fraud-case match in the assertions.
    fraud_match = next(
        a["match"] for a in inst["expected_state_changes"] if a["assert"] == "fraud_cases"
    )
    assert fraud_match["account_id"] == chk
    assert fraud_match["transaction_id"] == txn_id


def test_persona_first_name_used_in_criterion(demo_template):
    inst = instantiate(demo_template, 55)
    first = inst["persona"]["name"].split(" ", 1)[0]
    deposit_crit = next(
        c for c in inst["rubric_criteria"] if c["id"] == "addressed_pending_deposit"
    )
    assert first in deposit_crit["text"]
    assert first in inst["initial_message"]


def test_substitute_rewrites_dict_keys_and_dotted_paths():
    mapping = {"acct": "PERS-CHK-0001", "amt": 250}
    src = {
        "accounts": {"{{acct}}": {"balance": 10}},
        "assert": "accounts.{{acct}}.balance",
        "value": "{{amt}}",
    }
    out = substitute(src, mapping)
    assert out == {
        "accounts": {"PERS-CHK-0001": {"balance": 10}},
        "assert": "accounts.PERS-CHK-0001.balance",
        "value": 250,  # whole-string placeholder for a non-string -> typed value
    }


def test_find_placeholders_walks_keys_and_values():
    found = find_placeholders({"{{a}}": ["{{b}}", {"k": "x {{c}} y"}]})
    assert found == {"a", "b", "c"}


# --- 3. Validator-clean output -----------------------------------------------


@pytest.mark.parametrize("path", ALL_TEMPLATE_PATHS, ids=lambda p: p.stem)
def test_every_demo_template_validates_and_instantiates_clean(path):
    """Each shipped demonstration template validates as a template AND its
    instantiation (default + a non-default seed) is a validator-clean v0.2
    scenario with the declaration stripped. Guards every fixture, not just the
    banking one, so a new template added later cannot regress silently."""
    data = json.loads(path.read_text(encoding="utf-8"))
    assert validate_scenario_dict(data) == []
    for seed in (DEFAULT_INSTANTIATION_SEED, 4242):
        inst = instantiate(data, seed)
        assert "template_slots" not in inst
        assert validate_scenario_dict(inst) == []
        # Determinism holds per template.
        assert instantiate(data, seed) == inst


def test_cs_template_account_id_coherent_as_value_and_in_criteria(cs_template):
    """The customer_success template exercises a DIFFERENT coherence shape from
    the banking one: the account id is a ground-truth VALUE (not a dict key) that
    also appears in an expected_state_changes ``match`` and in criteria text, and
    the persona first name appears in criteria. One slot, rewritten everywhere."""
    inst = instantiate(cs_template, 2026)
    acct = inst["ground_truth"]["account"]["account_id"]
    # The feature-request match names the same account id...
    assert inst["expected_state_changes"][0]["match"]["account_id"] == acct
    # ...as does the internal-note match...
    assert inst["expected_state_changes"][1]["match"]["account_id"] == acct
    # ...and two criteria reference the same account id.
    assert any(acct in c["text"] for c in inst["rubric_criteria"])
    first = inst["persona"]["name"].split(" ", 1)[0]
    assert first in inst["initial_message"] or first in inst["rubric_criteria"][0]["text"]
    assert first in inst["rubric_criteria"][0]["text"]


def test_cs_template_out_of_scope_equals_assertions_survive(cs_template):
    """The out-of-scope assertions (emails_sent / discounts_applied must stay
    ``[]``) carry no slots; instantiation must leave their literal empty-list
    expectation intact while the goal text around them is rewritten coherently."""
    inst = instantiate(cs_template, 7)
    by_assert = {a["assert"]: a for a in inst["expected_state_changes"]}
    assert by_assert["emails_sent"]["op"] == "equals"
    assert by_assert["emails_sent"]["value"] == []
    assert by_assert["discounts_applied"]["value"] == []


def test_demo_template_validates(demo_template):
    assert validate_scenario_dict(demo_template) == []


def test_instantiated_scenario_validates(demo_template):
    inst = instantiate(demo_template, 314)
    assert validate_scenario_dict(inst) == []
    assert "template_slots" not in inst


def test_instantiated_scenario_builds_a_scenario_object(demo_template):
    inst = instantiate(demo_template, 11)
    scn = Scenario(
        id=inst["id"],
        domain=Domain.BANKING,
        persona=inst["persona"],
        user_goals=inst["user_goals"],
        tools=inst["tools"],
        category=inst["category"],
        initial_message=inst["initial_message"],
        ground_truth=inst.get("ground_truth"),
        expected_state_changes=inst.get("expected_state_changes"),
        rubric_criteria=inst.get("rubric_criteria"),
    )
    assert scn.rubric_criteria is not None
    assert scn.ground_truth is not None


@pytest.mark.parametrize(
    "broken,expected_fragment",
    [
        # Referenced but undeclared.
        ({"template_slots": {}, "initial_message": "hi {{x}}"}, "must be a non-empty object"),
        (
            {"template_slots": {"y": {"type": "digits"}}, "initial_message": "hi {{x}}"},
            "referenced but not declared",
        ),
        # Declared but never referenced.
        (
            {"template_slots": {"x": {"type": "digits"}}, "initial_message": "no placeholders"},
            "never referenced",
        ),
        # Unknown slot type.
        (
            {"template_slots": {"x": {"type": "bogus"}}, "initial_message": "{{x}}"},
            "unknown slot type",
        ),
        # choice without options.
        (
            {"template_slots": {"x": {"type": "choice"}}, "initial_message": "{{x}}"},
            "requires a non-empty 'options'",
        ),
    ],
)
def test_template_validation_errors(broken, expected_fragment):
    errors = validate_scenario_dict(broken)
    assert any(expected_fragment in e for e in errors), errors


# --- 4. Pre-registration honesty ---------------------------------------------


def test_template_corpus_hash_seed_invariant_instantiated_hash_seed_dependent(demo_template):
    templates_by_domain = {Domain.BANKING: [demo_template]}
    tmpl_hash_a, index = template_corpus_hash(templates_by_domain)
    tmpl_hash_b, _ = template_corpus_hash(templates_by_domain)
    # Template-corpus hash does not depend on any seed (it hashes raw templates).
    assert tmpl_hash_a == tmpl_hash_b
    assert index[0]["templated"] is True

    # Instantiated-corpus hash IS a function of the seed.
    def inst_hash(seed):
        inst = instantiate(demo_template, seed)
        scn = Scenario(
            id=inst["id"],
            domain=Domain.BANKING,
            persona=inst["persona"],
            user_goals=inst["user_goals"],
            tools=inst["tools"],
            category=inst["category"],
            initial_message=inst["initial_message"],
            ground_truth=inst.get("ground_truth"),
            expected_state_changes=inst.get("expected_state_changes"),
            rubric_criteria=inst.get("rubric_criteria"),
        )
        h, _ = scenario_set_hash({Domain.BANKING: [scn]})
        return h

    assert inst_hash(1) == inst_hash(1)  # reproducible from the seed
    assert inst_hash(1) != inst_hash(2)  # surface changes with the seed


def test_template_corpus_hash_marks_nontemplate_scenarios():
    plain = {"id": "p1", "initial_message": "hi"}
    _, index = template_corpus_hash({"banking": [plain]})
    assert index[0]["templated"] is False


# --- 5. Backwards compatibility ----------------------------------------------


def test_non_template_scenario_is_passthrough():
    plain = {"id": "p1", "persona": {"name": "X"}, "user_goals": ["a"]}
    assert not is_template(plain)
    out = instantiate(plain, 123)
    assert out == plain
    assert out is not plain  # a copy, not the same object


def test_non_template_scenario_with_empty_slots_is_not_a_template():
    # An explicitly empty/falsey template_slots is treated as non-template by
    # instantiate (is_template is False), so it passes through unchanged.
    plain = {"id": "p1", "template_slots": {}, "persona": {"name": "X"}}
    assert not is_template(plain)
    out = instantiate(plain, 1)
    assert "template_slots" not in out


def test_default_seed_is_stable(demo_template):
    a = instantiate(demo_template, DEFAULT_INSTANTIATION_SEED)
    b = instantiate(demo_template, DEFAULT_INSTANTIATION_SEED)
    assert a == b
