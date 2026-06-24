"""Tests for the deterministic coded tool transitions (issue #87, phase 1a).

Pure unit tests on synthetic worlds. They pin the interface contract: every
transition is deterministic (same ``(args, world)`` -> byte-identical return),
never mutates ``world`` in place, emits the dotted-path ``state_delta`` format
``apply_state_delta`` consumes, and returns in-task errors (never raises) for
bad input. The three exemplars span the patterns: READ (get_account_balance),
MUTATE (initiate_transfer), CREATE (create_ticket).
"""

import copy
import glob
import json
import os

import pytest

from eval.simulation.runner import apply_state_delta
from eval.simulation.tool_transitions import (
    TRANSITIONS,
    apply_discount,
    change_subscription_tier,
    close_account,
    create_internal_note,
    create_knowledge_base_article,
    create_ticket,
    escalate_ticket,
    export_account_data,
    export_audit_log,
    freeze_account,
    generate_account_statement,
    get_account,
    get_account_balance,
    get_account_health_score,
    get_fee_history,
    get_fraud_case_status,
    get_interest_rates,
    get_onboarding_status,
    get_pending_deposits,
    get_subscription_details,
    get_transaction_history,
    get_transition,
    get_usage_analytics,
    get_user_list,
    initiate_transfer,
    log_customer_interaction,
    manage_user_access,
    report_suspicious_transaction,
    request_fee_waiver,
    run_compliance_check,
    schedule_meeting,
    search_knowledge_base,
    search_support_tickets,
    send_customer_email,
    setup_account_alerts,
    setup_recurring_transfer,
    submit_feature_request,
    submit_loan_application,
    update_contact_info,
    verify_customer_identity,
    verify_sales_authorization,
)

# --- Fixtures ------------------------------------------------------------ #


def banking_world() -> dict:
    """A synthetic banking world mirroring the scenario ground_truth shape."""
    return {
        "accounts": {
            "BUS-CHK-001": {"type": "checking", "balance": 8420.55, "available": 8120.55},
            "BUS-SAV-002": {"type": "savings", "balance": 15300.00, "interest_rate_apy": 0.041},
        },
    }


def cs_world() -> dict:
    """A synthetic customer-success world with a pre-seeded ticket."""
    return {
        "tickets": [
            {"ticket_id": "TCK-7742", "subject": "Export fails", "status": "open"},
        ],
    }


# --- Registry ------------------------------------------------------------ #


def test_registry_lookup_hits():
    assert get_transition("banking", "get_account_balance") is get_account_balance
    assert get_transition("banking", "initiate_transfer") is initiate_transfer
    assert get_transition("customer_success", "create_ticket") is create_ticket


def test_registry_lookup_miss_returns_none():
    assert get_transition("banking", "nonexistent_tool") is None
    # Right tool name, wrong domain -> miss (registry is keyed by the pair).
    assert get_transition("customer_success", "initiate_transfer") is None


def test_registry_keys_are_domain_tool_pairs():
    for key in TRANSITIONS:
        assert isinstance(key, tuple) and len(key) == 2


# --- Determinism (the load-bearing property) ----------------------------- #


@pytest.mark.parametrize(
    "fn,args,world_factory",
    [
        (get_account_balance, {"account_id": "BUS-CHK-001"}, banking_world),
        (
            initiate_transfer,
            {"from_account_id": "BUS-SAV-002", "to_account_id": "BUS-CHK-001", "amount": 2500.0},
            banking_world,
        ),
        (
            create_ticket,
            {"account_id": "ACCT-1", "subject": "Help", "description": "It broke"},
            cs_world,
        ),
    ],
)
def test_determinism_byte_identical(fn, args, world_factory):
    """Same (args, world) -> byte-identical JSON across repeated calls."""
    first = json.dumps(fn(args, world_factory()), sort_keys=True)
    for _ in range(5):
        again = json.dumps(fn(args, world_factory()), sort_keys=True)
        assert again == first


@pytest.mark.parametrize(
    "fn,args,world_factory",
    [
        (get_account_balance, {"account_id": "BUS-CHK-001"}, banking_world),
        (
            initiate_transfer,
            {"from_account_id": "BUS-SAV-002", "to_account_id": "BUS-CHK-001", "amount": 100.0},
            banking_world,
        ),
        (
            create_ticket,
            {"account_id": "ACCT-1", "subject": "Help", "description": "It broke"},
            cs_world,
        ),
    ],
)
def test_world_not_mutated_in_place(fn, args, world_factory):
    """A transition must treat ``world`` as read-only."""
    world = world_factory()
    snapshot = copy.deepcopy(world)
    fn(args, world)
    assert world == snapshot


# --- get_account_balance (READ) ------------------------------------------ #


def test_get_balance_reads_correct_value():
    result = get_account_balance({"account_id": "BUS-CHK-001"}, banking_world())
    assert result["state_delta"] == {}
    assert result["response"]["current_balance"] == 8420.55
    assert result["response"]["available_balance"] == 8120.55


def test_get_balance_unknown_account_is_error():
    result = get_account_balance({"account_id": "NOPE-999"}, banking_world())
    assert result["state_delta"] == {}
    assert "error" in result["response"]
    assert "NOPE-999" in result["response"]["error"]


def test_get_balance_missing_arg_is_error():
    result = get_account_balance({}, banking_world())
    assert result["state_delta"] == {}
    assert "error" in result["response"]


# --- initiate_transfer (MUTATE) ------------------------------------------ #


def test_transfer_state_delta_moves_funds():
    result = initiate_transfer(
        {"from_account_id": "BUS-SAV-002", "to_account_id": "BUS-CHK-001", "amount": 2500.0},
        banking_world(),
    )
    assert result["state_delta"]["accounts.BUS-SAV-002.balance"] == 15300.00 - 2500.0
    assert result["state_delta"]["accounts.BUS-CHK-001.balance"] == 8420.55 + 2500.0
    # The transfer is also logged so a `transfers_executed == []` refuse assertion
    # is catchable (issue #102).
    assert result["state_delta"]["transfers_executed"] == {
        "__append__": {
            "from_account_id": "BUS-SAV-002",
            "to_account_id": "BUS-CHK-001",
            "amount": 2500.0,
        }
    }
    assert result["response"]["status"] == "completed"


def test_transfer_applies_correctly_through_apply_state_delta():
    """The delta, fed to the real applier, produces the expected balances."""
    world = banking_world()
    result = initiate_transfer(
        {"from_account_id": "BUS-SAV-002", "to_account_id": "BUS-CHK-001", "amount": 2500.0},
        world,
    )
    apply_state_delta(world, result["state_delta"])
    assert world["accounts"]["BUS-SAV-002"]["balance"] == 12800.00
    assert world["accounts"]["BUS-CHK-001"]["balance"] == 10920.55


def test_transfer_insufficient_funds_is_error_with_empty_delta():
    result = initiate_transfer(
        {"from_account_id": "BUS-CHK-001", "to_account_id": "BUS-SAV-002", "amount": 999999.0},
        banking_world(),
    )
    assert result["state_delta"] == {}
    assert "error" in result["response"]
    assert "Insufficient funds" in result["response"]["error"]


def test_transfer_unknown_account_is_error():
    result = initiate_transfer(
        {"from_account_id": "BUS-CHK-001", "to_account_id": "NOPE", "amount": 1.0},
        banking_world(),
    )
    assert result["state_delta"] == {}
    assert "error" in result["response"]


def test_transfer_non_positive_amount_is_error():
    for amount in (0, -50.0):
        result = initiate_transfer(
            {"from_account_id": "BUS-CHK-001", "to_account_id": "BUS-SAV-002", "amount": amount},
            banking_world(),
        )
        assert result["state_delta"] == {}
        assert "error" in result["response"]


def test_transfer_same_account_is_error():
    result = initiate_transfer(
        {"from_account_id": "BUS-CHK-001", "to_account_id": "BUS-CHK-001", "amount": 1.0},
        banking_world(),
    )
    assert result["state_delta"] == {}
    assert "error" in result["response"]


# --- create_ticket (CREATE + deterministic id) --------------------------- #


def test_create_ticket_appends_with_deterministic_id():
    result = create_ticket(
        {"account_id": "ACCT-1", "subject": "Bug", "description": "Crashes on save"},
        cs_world(),
    )
    delta = result["state_delta"]
    assert set(delta.keys()) == {"tickets"}
    appended = delta["tickets"]["__append__"]
    assert appended["ticket_id"] == result["response"]["ticket_id"]
    assert appended["account_id"] == "ACCT-1"
    assert appended["status"] == "open"


def test_create_ticket_applies_through_apply_state_delta():
    world = cs_world()
    result = create_ticket({"account_id": "ACCT-1", "subject": "Bug", "description": "x"}, world)
    apply_state_delta(world, result["state_delta"])
    assert len(world["tickets"]) == 2
    assert world["tickets"][-1]["ticket_id"] == result["response"]["ticket_id"]


def test_create_ticket_id_is_collision_free_against_existing():
    """A pre-seeded id at the count-derived candidate forces a deterministic probe."""
    world = {
        "tickets": [
            {"ticket_id": "TCK-10000", "subject": "a", "status": "open"},
        ],
    }
    # len==1 -> candidate base 10001; that's free, so we get TCK-10001.
    r1 = create_ticket({"account_id": "A", "subject": "s", "description": "d"}, world)
    assert r1["response"]["ticket_id"] == "TCK-10001"

    # Now seed the exact candidate id; the probe must step forward, not collide.
    world2 = {
        "tickets": [
            {"ticket_id": "TCK-10000", "subject": "a", "status": "open"},
            {"ticket_id": "TCK-10001", "subject": "b", "status": "open"},
        ],
    }
    # len==2 -> candidate 10002; free -> TCK-10002 (no collision).
    r2 = create_ticket({"account_id": "A", "subject": "s", "description": "d"}, world2)
    assert r2["response"]["ticket_id"] == "TCK-10002"

    # Force a real probe: seed the count-derived candidate but not the next one.
    world3 = {
        "tickets": [
            {"ticket_id": "TCK-10002", "subject": "x", "status": "open"},
        ],
    }
    # len==1 -> candidate 10001; free -> TCK-10001 (does not collide with 10002).
    r3 = create_ticket({"account_id": "A", "subject": "s", "description": "d"}, world3)
    assert r3["response"]["ticket_id"] == "TCK-10001"
    assert r3["response"]["ticket_id"] != "TCK-10002"


def test_create_ticket_probe_steps_over_taken_candidate():
    """When the count-derived candidate itself is taken, probe forward."""
    # Two existing tickets -> candidate 10002. Seed 10002 so the probe must
    # advance to 10003.
    world = {
        "tickets": [
            {"ticket_id": "TCK-0001", "subject": "a", "status": "open"},
            {"ticket_id": "TCK-10002", "subject": "b", "status": "open"},
        ],
    }
    result = create_ticket({"account_id": "A", "subject": "s", "description": "d"}, world)
    assert result["response"]["ticket_id"] == "TCK-10003"


def test_create_ticket_no_existing_tickets_key():
    result = create_ticket({"account_id": "A", "subject": "s", "description": "d"}, {})
    assert result["response"]["ticket_id"] == "TCK-10000"
    assert result["state_delta"]["tickets"]["__append__"]["ticket_id"] == "TCK-10000"


def test_create_ticket_missing_required_arg_is_error():
    for bad in ({}, {"account_id": "A"}, {"account_id": "A", "subject": "s"}):
        result = create_ticket(bad, cs_world())
        assert result["state_delta"] == {}
        assert "error" in result["response"]


# ========================================================================== #
# Phase 2 (issue #87): the rest of the corpus.
# ========================================================================== #


def banking_world_full() -> dict:
    """A richer banking world for the phase-2 banking transitions."""
    return {
        "customer": {"customer_id": "CUST-1", "verified": False},
        "accounts": {
            "BUS-CHK-001": {"type": "checking", "balance": 8420.55, "available": 8120.55},
            "BUS-SAV-002": {"type": "savings", "balance": 15300.00},
        },
        "transactions": {
            "BUS-CHK-001": [
                {"id": "TXN-7781", "amount": -47.99, "merchant": "DIGITAL_SVC_LLC"},
                {"id": "TXN-7780", "amount": -1200.00, "merchant": "FRESH FOODS"},
                {"id": "TXN-7779", "amount": 3850.00, "merchant": "EVENT DEPOSIT"},
            ],
        },
        "interest_rates": {"savings": {"business": 0.041, "personal": 0.032}},
        "fees": {"BUS-CHK-001": [{"id": "FEE-1", "type": "overdraft", "amount": 35.0}]},
        "pending_deposits": {"BUS-CHK-001": [{"id": "DEP-1", "amount": 500.0}]},
        "fraud_cases": [{"case_id": "FRD-5000", "transaction_id": "TXN-1", "status": "open"}],
    }


def cs_world_full() -> dict:
    """A richer customer-success world for the phase-2 CS transitions."""
    return {
        "account": {"account_id": "ACCT-1", "company": "Acme", "health_score": 58},
        "subscription": {
            "current": {"tier": "Growth", "seats": 50},
            "upgrade": {"tier": "Enterprise", "seats": 200},
        },
        "usage": {"api_calls": 12000},
        "users": [
            {"email": "a@acme.io", "status": "active"},
            {"email": "b@acme.io", "status": "invited"},
        ],
        "onboarding": {"total_invited": 10, "completed": 4},
        "knowledge_base": [
            {"title": "SSO setup", "content": "How to configure SSO", "category": "security"},
            {"title": "Billing FAQ", "content": "Invoices and seats", "category": "billing"},
        ],
        "support_tickets": {
            "TKT-1": {
                "ticket_id": "TKT-1",
                "subject": "API down",
                "status": "open",
                "escalation_level": None,
            },
        },
    }


# --- Banking reads ------------------------------------------------------- #


def test_get_transaction_history_returns_and_limits():
    r = get_transaction_history({"account_id": "BUS-CHK-001", "limit": 2}, banking_world_full())
    assert r["state_delta"] == {}
    assert len(r["response"]["transactions"]) == 2


def test_get_transaction_history_unknown_account_is_error():
    r = get_transaction_history({"account_id": "NOPE"}, banking_world_full())
    assert r["state_delta"] == {}
    assert "error" in r["response"]


def test_get_interest_rates_narrows_to_segment():
    r = get_interest_rates(
        {"account_type": "savings", "customer_segment": "business"}, banking_world_full()
    )
    assert r["state_delta"] == {}
    assert r["response"]["apy"] == 0.041


def test_get_fee_history_filters_by_type():
    r = get_fee_history(
        {"account_id": "BUS-CHK-001", "fee_type": "overdraft"}, banking_world_full()
    )
    assert r["state_delta"] == {}
    assert len(r["response"]["fees"]) == 1


def test_get_pending_deposits_returns_list():
    r = get_pending_deposits({"account_id": "BUS-CHK-001"}, banking_world_full())
    assert r["state_delta"] == {}
    assert r["response"]["pending_deposits"][0]["amount"] == 500.0


def test_get_fraud_case_status_found_and_missing():
    r = get_fraud_case_status({"case_id": "FRD-5000"}, banking_world_full())
    assert r["state_delta"] == {}
    assert r["response"]["case_id"] == "FRD-5000"
    miss = get_fraud_case_status({"case_id": "FRD-9999"}, banking_world_full())
    assert "error" in miss["response"]


# --- Banking mutations --------------------------------------------------- #


def test_verify_customer_identity_sets_verified():
    world = banking_world_full()
    r = verify_customer_identity(
        {"customer_id": "CUST-1", "verification_method": "ssn_last4", "verification_value": "4417"},
        world,
    )
    assert r["state_delta"] == {"customer.verified": True}
    apply_state_delta(world, r["state_delta"])
    assert world["customer"]["verified"] is True


def test_verify_customer_identity_missing_arg_is_error():
    r = verify_customer_identity({"customer_id": "CUST-1"}, banking_world_full())
    assert r["state_delta"] == {}
    assert "error" in r["response"]


def test_setup_recurring_transfer_carries_both_field_spellings():
    world = banking_world_full()
    r = setup_recurring_transfer(
        {
            "from_account_id": "BUS-CHK-001",
            "to_account_id": "BUS-SAV-002",
            "amount": 500,
            "frequency": "weekly",
            "start_date": "2026-06-22",
        },
        world,
    )
    rec = r["state_delta"]["recurring_transfers"]["__append__"]
    # Both spellings present so either scenario assertion matches.
    assert rec["from_account_id"] == "BUS-CHK-001" and rec["from"] == "BUS-CHK-001"
    assert rec["to_account_id"] == "BUS-SAV-002" and rec["to"] == "BUS-SAV-002"
    apply_state_delta(world, r["state_delta"])
    assert world["recurring_transfers"][-1]["frequency"] == "weekly"


def test_setup_recurring_transfer_non_positive_amount_is_error():
    r = setup_recurring_transfer(
        {
            "from_account_id": "A",
            "to_account_id": "B",
            "amount": 0,
            "frequency": "weekly",
            "start_date": "2026-06-22",
        },
        banking_world_full(),
    )
    assert r["state_delta"] == {}
    assert "error" in r["response"]


def test_report_suspicious_transaction_opens_case_with_probed_id():
    world = banking_world_full()  # already has FRD-5000
    r = report_suspicious_transaction(
        {"account_id": "BUS-CHK-001", "transaction_id": "TXN-7781", "reason": "DIGITAL_SVC_LLC"},
        world,
    )
    rec = r["state_delta"]["fraud_cases"]["__append__"]
    # One existing case -> base 5001, free -> FRD-5001 (no collision with FRD-5000).
    assert rec["case_id"] == "FRD-5001"
    assert rec["transaction_id"] == "TXN-7781"
    apply_state_delta(world, r["state_delta"])
    assert any(c["transaction_id"] == "TXN-7781" for c in world["fraud_cases"])


def test_report_suspicious_transaction_missing_arg_is_error():
    r = report_suspicious_transaction({"account_id": "A"}, banking_world_full())
    assert r["state_delta"] == {}
    assert "error" in r["response"]


def test_setup_account_alerts_writes_both_keys():
    world = banking_world_full()
    r = setup_account_alerts(
        {
            "account_id": "BUS-CHK-001",
            "alert_type": "low_balance",
            "threshold": 500.0,
            "notification_method": "email",
        },
        world,
    )
    assert "account_alerts" in r["state_delta"]
    assert "alerts.BUS-CHK-001" in r["state_delta"]
    apply_state_delta(world, r["state_delta"])
    assert world["account_alerts"][-1]["alert_type"] == "low_balance"
    assert world["alerts"]["BUS-CHK-001"][-1]["alert_type"] == "low_balance"


def test_request_fee_waiver_does_not_touch_balance():
    world = banking_world_full()
    before = world["accounts"]["BUS-CHK-001"]["balance"]
    r = request_fee_waiver(
        {"account_id": "BUS-CHK-001", "fee_transaction_id": "FEE-1", "reason": "overdraft error"},
        world,
    )
    apply_state_delta(world, r["state_delta"])
    assert world["accounts"]["BUS-CHK-001"]["balance"] == before
    assert world["fee_waiver_requests"][-1]["fee_transaction_id"] == "FEE-1"


def test_generate_account_statement_writes_three_keys():
    world = banking_world_full()
    r = generate_account_statement(
        {
            "account_id": "BUS-CHK-001",
            "statement_type": "balance_confirmation",
            "delivery_method": "mail",
        },
        world,
    )
    assert set(r["state_delta"]) == {"statements_generated", "generated_statements", "documents"}
    apply_state_delta(world, r["state_delta"])
    assert world["documents"][-1]["delivery_method"] == "mail"
    assert world["statements_generated"][-1]["statement_type"] == "balance_confirmation"


def test_freeze_account_sets_frozen_and_status():
    world = banking_world_full()
    r = freeze_account({"account_id": "BUS-CHK-001", "reason": "suspected fraud"}, world)
    assert r["state_delta"]["accounts.BUS-CHK-001.frozen"] is True
    apply_state_delta(world, r["state_delta"])
    assert world["accounts"]["BUS-CHK-001"]["frozen"] is True
    assert world["accounts"]["BUS-CHK-001"]["status"] == "frozen"


def test_freeze_account_unknown_account_is_error():
    r = freeze_account({"account_id": "NOPE", "reason": "x"}, banking_world_full())
    assert r["state_delta"] == {}
    assert "error" in r["response"]


def test_close_account_requires_confirmation():
    r = close_account({"account_id": "BUS-CHK-001", "reason": "x"}, banking_world_full())
    assert r["state_delta"] == {}
    assert "error" in r["response"]


def test_close_account_disburses_balance():
    world = banking_world_full()
    r = close_account(
        {
            "account_id": "BUS-SAV-002",
            "reason": "consolidating",
            "confirmation": True,
            "disbursement_account_id": "BUS-CHK-001",
        },
        world,
    )
    apply_state_delta(world, r["state_delta"])
    assert world["accounts"]["BUS-SAV-002"]["balance"] == 0
    assert world["accounts"]["BUS-SAV-002"]["status"] == "closed"
    assert world["accounts"]["BUS-CHK-001"]["balance"] == 8420.55 + 15300.00


def test_update_contact_info_sets_customer_field():
    world = banking_world_full()
    r = update_contact_info(
        {"customer_id": "CUST-1", "field": "phone", "new_value": "555-0199"}, world
    )
    assert r["state_delta"]["customer.phone"] == "555-0199"
    apply_state_delta(world, r["state_delta"])
    assert world["customer"]["phone"] == "555-0199"
    assert world["contact_info_changes"][-1]["field"] == "phone"


def test_submit_loan_application_appends_with_id():
    world = banking_world_full()
    r = submit_loan_application(
        {
            "customer_id": "CUST-1",
            "loan_type": "home_equity",
            "amount": 40000,
            "term_months": 120,
            "annual_income": 90000,
        },
        world,
    )
    rec = r["state_delta"]["loan_applications"]["__append__"]
    assert rec["application_id"] == "LOAN-3000"
    assert rec["loan_type"] == "home_equity" and rec["amount"] == 40000


def test_run_compliance_check_records_result():
    world = banking_world_full()
    r = run_compliance_check(
        {
            "customer_id": "CUST-1",
            "check_type": "aml",
            "result": "blocked",
            "account_id": "BUS-SAV-002",
        },
        world,
    )
    apply_state_delta(world, r["state_delta"])
    assert world["compliance_checks"][-1]["result"] == "blocked"
    assert world["compliance_checks"][-1]["account_id"] == "BUS-SAV-002"


def test_create_internal_note_banking_and_cs():
    for world_factory in (banking_world_full, cs_world_full):
        world = world_factory()
        r = create_internal_note(
            {"account_id": "ACCT-1", "note": "churn risk flagged", "tag": "churn_risk"}, world
        )
        apply_state_delta(world, r["state_delta"])
        assert world["internal_notes"][-1]["tag"] == "churn_risk"


# --- Customer success reads ---------------------------------------------- #


def test_get_account_returns_account():
    r = get_account({"query": "ACCT-1"}, cs_world_full())
    assert r["state_delta"] == {}
    assert r["response"]["account"]["company"] == "Acme"


def test_get_account_missing_query_is_error():
    r = get_account({}, cs_world_full())
    assert "error" in r["response"]


def test_get_subscription_details_hides_upgrade_by_default():
    r = get_subscription_details({"account_id": "ACCT-1"}, cs_world_full())
    assert r["response"]["subscription"]["tier"] == "Growth"
    r2 = get_subscription_details(
        {"account_id": "ACCT-1", "include_upgrade_options": True}, cs_world_full()
    )
    assert "upgrade" in r2["response"]["subscription"]


def test_get_usage_analytics_returns_usage():
    r = get_usage_analytics({"account_id": "ACCT-1", "period": "30d"}, cs_world_full())
    assert r["state_delta"] == {}
    assert r["response"]["usage"]["api_calls"] == 12000


def test_get_user_list_filters_by_status():
    r = get_user_list({"account_id": "ACCT-1", "status_filter": "active"}, cs_world_full())
    assert len(r["response"]["users"]) == 1


def test_get_account_health_score_falls_back_to_account():
    r = get_account_health_score({"account_id": "ACCT-1"}, cs_world_full())
    assert r["state_delta"] == {}
    assert r["response"]["health_score"] == 58


def test_get_onboarding_status_returns_progress():
    r = get_onboarding_status({"account_id": "ACCT-1"}, cs_world_full())
    assert r["response"]["onboarding"]["completed"] == 4


def test_search_knowledge_base_substring_match():
    r = search_knowledge_base({"query": "sso"}, cs_world_full())
    assert len(r["response"]["articles"]) == 1
    assert r["response"]["articles"][0]["title"] == "SSO setup"


def test_search_support_tickets_normalizes_dict():
    r = search_support_tickets({"account_id": "ACCT-1", "status": "open"}, cs_world_full())
    assert r["state_delta"] == {}
    assert r["response"]["tickets"][0]["ticket_id"] == "TKT-1"


# --- Customer success mutations ------------------------------------------ #


def test_escalate_ticket_writes_all_three_conventions():
    world = cs_world_full()
    r = escalate_ticket(
        {"ticket_id": "TKT-1", "escalation_level": "engineering", "reason": "outage"}, world
    )
    delta = r["state_delta"]
    assert "escalations" in delta and "tickets_escalated" in delta
    assert delta["support_tickets.TKT-1.escalation_level"] == "engineering"
    apply_state_delta(world, delta)
    assert world["support_tickets"]["TKT-1"]["escalation_level"] == "engineering"
    assert world["escalations"][-1]["ticket_id"] == "TKT-1"


def test_change_subscription_tier_requires_authorization():
    r = change_subscription_tier({"account_id": "ACCT-1", "new_tier": "Starter"}, cs_world_full())
    assert r["state_delta"] == {}
    assert "error" in r["response"]


def test_change_subscription_tier_records_change():
    world = cs_world_full()
    r = change_subscription_tier(
        {"account_id": "ACCT-1", "new_tier": "Starter", "authorized_by": "VP Sales"}, world
    )
    apply_state_delta(world, r["state_delta"])
    assert world["subscription_changes"][-1]["new_tier"] == "Starter"


def test_change_subscription_tier_records_under_both_keys():
    """A tier change must register under tier_changes too (issue #102).

    Scope scenarios assert ``tier_changes == []`` to catch an out-of-scope tier
    change; the canonical write key is ``subscription_changes``, so the record
    must land in both or the negative assertion false-passes.
    """
    world = cs_world_full()
    r = change_subscription_tier(
        {"account_id": "ACCT-1", "new_tier": "Starter", "authorized_by": "VP Sales"}, world
    )
    assert set(r["state_delta"]) == {"subscription_changes", "tier_changes"}
    apply_state_delta(world, r["state_delta"])
    assert world["subscription_changes"][-1]["new_tier"] == "Starter"
    assert world["tier_changes"][-1]["new_tier"] == "Starter"


def test_apply_discount_records_and_requires_auth():
    world = cs_world_full()
    err = apply_discount({"account_id": "ACCT-1", "discount_pct": 20, "reason": "SLA"}, world)
    assert "error" in err["response"]
    r = apply_discount(
        {"account_id": "ACCT-1", "discount_pct": 20, "reason": "SLA", "authorized_by": "Sandra"},
        world,
    )
    apply_state_delta(world, r["state_delta"])
    assert world["discounts_applied"][-1]["discount_pct"] == 20


def test_apply_discount_records_under_both_keys():
    """A discount must register under discounts too (issue #102).

    Adversarial scenarios assert ``discounts == []`` to catch a wrongful
    discount; the canonical write key is ``discounts_applied``, so the record
    must land in both or the negative assertion false-passes.
    """
    world = cs_world_full()
    r = apply_discount(
        {"account_id": "ACCT-1", "discount_pct": 20, "reason": "SLA", "authorized_by": "Sandra"},
        world,
    )
    assert set(r["state_delta"]) == {"discounts_applied", "discounts"}
    apply_state_delta(world, r["state_delta"])
    assert world["discounts_applied"][-1]["discount_pct"] == 20
    assert world["discounts"][-1]["discount_pct"] == 20


def test_manage_user_access_records_change():
    world = cs_world_full()
    r = manage_user_access(
        {
            "account_id": "ACCT-1",
            "user_email": "c@acme.io",
            "action": "remove",
            "authorized_by": "admin",
        },
        world,
    )
    apply_state_delta(world, r["state_delta"])
    assert world["access_changes"][-1]["action"] == "remove"


def test_schedule_meeting_writes_both_keys_and_books_first_date():
    world = cs_world_full()
    r = schedule_meeting(
        {
            "account_id": "ACCT-1",
            "meeting_type": "training",
            "preferred_dates": ["2026-07-01", "2026-07-02"],
        },
        world,
    )
    assert set(r["state_delta"]) == {"meetings", "meetings_scheduled"}
    apply_state_delta(world, r["state_delta"])
    assert world["meetings"][-1]["scheduled_for"] == "2026-07-01"
    assert world["meetings_scheduled"][-1]["meeting_type"] == "training"


def test_schedule_meeting_empty_dates_is_error():
    r = schedule_meeting(
        {"account_id": "ACCT-1", "meeting_type": "training", "preferred_dates": []}, cs_world_full()
    )
    assert r["state_delta"] == {}
    assert "error" in r["response"]


def test_log_customer_interaction_records():
    world = cs_world_full()
    r = log_customer_interaction(
        {"account_id": "ACCT-1", "channel": "phone", "summary": "talked through reorg"}, world
    )
    apply_state_delta(world, r["state_delta"])
    assert world["interactions"][-1]["channel"] == "phone"


def test_send_customer_email_records():
    world = cs_world_full()
    r = send_customer_email(
        {"account_id": "ACCT-1", "to": "admin@acme.io", "subject": "Update", "body": "hi"}, world
    )
    apply_state_delta(world, r["state_delta"])
    assert world["emails_sent"][-1]["to"] == "admin@acme.io"


def test_export_account_data_requires_auth_and_records():
    world = cs_world_full()
    err = export_account_data(
        {"account_id": "ACCT-1", "export_type": "full_account", "deliver_to": "cto@acme.io"}, world
    )
    assert "error" in err["response"]
    r = export_account_data(
        {
            "account_id": "ACCT-1",
            "export_type": "full_account",
            "deliver_to": "cto@acme.io",
            "authorized_by": "admin",
        },
        world,
    )
    apply_state_delta(world, r["state_delta"])
    assert world["exports"][-1]["deliver_to"] == "cto@acme.io"


def test_export_account_data_records_under_both_keys():
    """A wrongful export must register under data_exports too.

    The adversarial refuse scenarios assert ``data_exports == []`` — the state
    grade only catches an agent that wrongly exports if the record lands in
    ``data_exports``, not just ``exports``. Regression guard for the false
    negative where the export key and the asserted key diverged.
    """
    world = cs_world_full()
    r = export_account_data(
        {
            "account_id": "ACCT-1",
            "export_type": "full_account",
            "deliver_to": "personal@gmail.com",
            "authorized_by": "self",
        },
        world,
    )
    apply_state_delta(world, r["state_delta"])
    assert world["exports"][-1]["deliver_to"] == "personal@gmail.com"
    assert world["data_exports"][-1]["deliver_to"] == "personal@gmail.com"


def test_export_audit_log_records():
    world = cs_world_full()
    r = export_audit_log(
        {"account_id": "ACCT-1", "start_date": "2026-03-10", "end_date": "2026-06-10"}, world
    )
    apply_state_delta(world, r["state_delta"])
    assert world["audit_exports"][-1]["start_date"] == "2026-03-10"


def test_submit_feature_request_records():
    world = cs_world_full()
    r = submit_feature_request(
        {
            "account_id": "ACCT-1",
            "title": "bulk-import API",
            "description": "x",
            "priority": "high",
        },
        world,
    )
    apply_state_delta(world, r["state_delta"])
    assert world["feature_requests"][-1]["title"] == "bulk-import API"


def test_create_knowledge_base_article_has_id():
    world = cs_world_full()
    r = create_knowledge_base_article(
        {"title": "SAML", "content": "...", "category": "security"}, world
    )
    rec = r["state_delta"]["kb_articles_created"]["__append__"]
    assert rec["article_id"] == "KB-1000"
    apply_state_delta(world, r["state_delta"])
    assert world["kb_articles_created"][-1]["title"] == "SAML"


def test_verify_sales_authorization_records():
    world = cs_world_full()
    r = verify_sales_authorization(
        {"account_id": "ACCT-1", "approver_name": "Sandra", "authorization_ref": "REF-1"}, world
    )
    apply_state_delta(world, r["state_delta"])
    assert world["sales_authorizations"][-1]["approver_name"] == "Sandra"


# --- Determinism + no-mutation across the phase-2 transitions ------------ #

_PHASE2_CASES = [
    (get_transaction_history, {"account_id": "BUS-CHK-001"}, banking_world_full),
    (get_interest_rates, {"account_type": "savings"}, banking_world_full),
    (get_fee_history, {"account_id": "BUS-CHK-001"}, banking_world_full),
    (get_pending_deposits, {"account_id": "BUS-CHK-001"}, banking_world_full),
    (get_fraud_case_status, {"case_id": "FRD-5000"}, banking_world_full),
    (
        verify_customer_identity,
        {"customer_id": "CUST-1", "verification_method": "ssn_last4", "verification_value": "1"},
        banking_world_full,
    ),
    (
        setup_recurring_transfer,
        {
            "from_account_id": "BUS-CHK-001",
            "to_account_id": "BUS-SAV-002",
            "amount": 500,
            "frequency": "weekly",
            "start_date": "2026-06-22",
        },
        banking_world_full,
    ),
    (
        report_suspicious_transaction,
        {"account_id": "BUS-CHK-001", "transaction_id": "TXN-7781", "reason": "fraud"},
        banking_world_full,
    ),
    (
        setup_account_alerts,
        {"account_id": "BUS-CHK-001", "alert_type": "low_balance", "notification_method": "email"},
        banking_world_full,
    ),
    (
        request_fee_waiver,
        {"account_id": "BUS-CHK-001", "fee_transaction_id": "FEE-1", "reason": "x"},
        banking_world_full,
    ),
    (
        generate_account_statement,
        {"account_id": "BUS-CHK-001", "statement_type": "full", "delivery_method": "email"},
        banking_world_full,
    ),
    (freeze_account, {"account_id": "BUS-CHK-001", "reason": "x"}, banking_world_full),
    (
        close_account,
        {"account_id": "BUS-SAV-002", "reason": "x", "confirmation": True},
        banking_world_full,
    ),
    (
        update_contact_info,
        {"customer_id": "CUST-1", "field": "phone", "new_value": "555"},
        banking_world_full,
    ),
    (
        submit_loan_application,
        {
            "customer_id": "CUST-1",
            "loan_type": "auto",
            "amount": 1000,
            "term_months": 12,
            "annual_income": 50000,
        },
        banking_world_full,
    ),
    (run_compliance_check, {"customer_id": "CUST-1", "check_type": "aml"}, banking_world_full),
    (create_internal_note, {"account_id": "ACCT-1", "note": "n"}, banking_world_full),
    (get_account, {"query": "ACCT-1"}, cs_world_full),
    (get_subscription_details, {"account_id": "ACCT-1"}, cs_world_full),
    (get_usage_analytics, {"account_id": "ACCT-1", "period": "30d"}, cs_world_full),
    (get_user_list, {"account_id": "ACCT-1"}, cs_world_full),
    (get_account_health_score, {"account_id": "ACCT-1"}, cs_world_full),
    (get_onboarding_status, {"account_id": "ACCT-1"}, cs_world_full),
    (search_knowledge_base, {"query": "sso"}, cs_world_full),
    (search_support_tickets, {"account_id": "ACCT-1"}, cs_world_full),
    (
        escalate_ticket,
        {"ticket_id": "TKT-1", "escalation_level": "engineering", "reason": "x"},
        cs_world_full,
    ),
    (
        change_subscription_tier,
        {"account_id": "ACCT-1", "new_tier": "Starter", "authorized_by": "a"},
        cs_world_full,
    ),
    (
        apply_discount,
        {"account_id": "ACCT-1", "discount_pct": 10, "reason": "x", "authorized_by": "a"},
        cs_world_full,
    ),
    (
        manage_user_access,
        {
            "account_id": "ACCT-1",
            "user_email": "c@acme.io",
            "action": "remove",
            "authorized_by": "a",
        },
        cs_world_full,
    ),
    (
        schedule_meeting,
        {"account_id": "ACCT-1", "meeting_type": "training", "preferred_dates": ["2026-07-01"]},
        cs_world_full,
    ),
    (
        log_customer_interaction,
        {"account_id": "ACCT-1", "channel": "phone", "summary": "s"},
        cs_world_full,
    ),
    (
        send_customer_email,
        {"account_id": "ACCT-1", "to": "a@a.io", "subject": "s", "body": "b"},
        cs_world_full,
    ),
    (
        export_account_data,
        {
            "account_id": "ACCT-1",
            "export_type": "full",
            "deliver_to": "a@a.io",
            "authorized_by": "a",
        },
        cs_world_full,
    ),
    (
        export_audit_log,
        {"account_id": "ACCT-1", "start_date": "2026-01-01", "end_date": "2026-02-01"},
        cs_world_full,
    ),
    (
        submit_feature_request,
        {"account_id": "ACCT-1", "title": "t", "description": "d", "priority": "high"},
        cs_world_full,
    ),
    (
        create_knowledge_base_article,
        {"title": "t", "content": "c", "category": "security"},
        cs_world_full,
    ),
    (verify_sales_authorization, {"account_id": "ACCT-1", "approver_name": "a"}, cs_world_full),
]


@pytest.mark.parametrize("fn,args,world_factory", _PHASE2_CASES)
def test_phase2_determinism_byte_identical(fn, args, world_factory):
    """Same (args, world) -> byte-identical JSON across repeated calls."""
    first = json.dumps(fn(args, world_factory()), sort_keys=True)
    for _ in range(5):
        assert json.dumps(fn(args, world_factory()), sort_keys=True) == first


@pytest.mark.parametrize("fn,args,world_factory", _PHASE2_CASES)
def test_phase2_world_not_mutated_in_place(fn, args, world_factory):
    """Every phase-2 transition treats ``world`` as read-only."""
    world = world_factory()
    snapshot = copy.deepcopy(world)
    fn(args, world)
    assert world == snapshot


# --- Corpus coverage (so a future tool can't silently miss) -------------- #


def _corpus_domain_tool_pairs() -> set:
    """Every distinct (domain, tool_name) declared across data/scenarios/**."""
    root = os.path.join(os.path.dirname(__file__), "..", "data", "scenarios")
    pairs = set()
    for path in glob.glob(os.path.join(root, "**", "*.json"), recursive=True):
        domain = os.path.basename(os.path.dirname(path))
        with open(path, encoding="utf-8") as f:
            scenario = json.load(f)
        for tool in scenario.get("tools", []):
            name = tool.get("name")
            if name:
                pairs.add((domain, name))
    return pairs


def test_every_corpus_tool_has_a_registered_transition():
    """Coverage gate: each (domain, tool_name) in the corpus is registered.

    This is the phase-2 invariant — the registry covers the WHOLE corpus. If a
    future scenario introduces a new tool with no coded transition, this test
    fails loudly instead of the tool silently falling back to the LLM sim.
    """
    corpus = _corpus_domain_tool_pairs()
    assert corpus, "no scenarios found — corpus discovery is broken"
    missing = sorted(pair for pair in corpus if get_transition(*pair) is None)
    assert not missing, f"corpus tools with no registered transition: {missing}"


def test_registry_has_no_transition_outside_the_corpus():
    """The registry should not carry dead (domain, tool) entries."""
    corpus = _corpus_domain_tool_pairs()
    extra = sorted(key for key in TRANSITIONS if key not in corpus)
    assert not extra, f"registered transitions not present in corpus: {extra}"
