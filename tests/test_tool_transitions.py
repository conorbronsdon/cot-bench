"""Tests for the deterministic coded tool transitions (issue #87, phase 1a).

Pure unit tests on synthetic worlds. They pin the interface contract: every
transition is deterministic (same ``(args, world)`` -> byte-identical return),
never mutates ``world`` in place, emits the dotted-path ``state_delta`` format
``apply_state_delta`` consumes, and returns in-task errors (never raises) for
bad input. The three exemplars span the patterns: READ (get_account_balance),
MUTATE (initiate_transfer), CREATE (create_ticket).
"""

import copy
import json

import pytest

from eval.simulation.runner import apply_state_delta
from eval.simulation.tool_transitions import (
    TRANSITIONS,
    create_ticket,
    get_account_balance,
    get_transition,
    initiate_transfer,
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
    assert result["state_delta"] == {
        "accounts.BUS-SAV-002.balance": 15300.00 - 2500.0,
        "accounts.BUS-CHK-001.balance": 8420.55 + 2500.0,
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
