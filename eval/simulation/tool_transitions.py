"""Deterministic coded tool transitions (issue #87, phase 1a).

COT Bench's identity is deterministic state grading: the grader reads a single
canonical world, the recovery probes apply byte-identical scripted perturbations,
and dual-control attribution must be reproducible run-to-run. The one place that
identity leaks is the tool simulator. Today a registered tool's effect on the
graded world is whatever an LLM (``runner._simulate_tool_stateful``) narrates: it
reads the world, invents a ``response`` and a ``state_delta``, and the runner
applies that delta. That makes the *authority over the graded world* an LLM — so
"the agent moved $2,500 from savings to checking" is true only if the sim chose to
emit the matching decrement/increment. It usually does, but not deterministically.

This module is the foundation of the fix (the tau-bench architecture): replace the
LLM's authority over registered tools with DETERMINISTIC PURE Python functions.
A coded transition reads ``(args, world)`` and returns the SAME
``{"response", "state_delta"}`` shape the sim produced — so the existing
``apply_state_delta`` applier and the Option A ``writes``-clamp keep working
unchanged — but it does so as a pure function of its inputs. Same inputs always
give the byte-identical return; the grader can trust that a mutation happened
because the tool's contract says it must, not because an LLM chose to narrate it.

Phase 1a (this PR) ships only the registry, the interface contract, three
exemplar transitions that span the patterns (READ / MUTATE / CREATE), and pure
unit tests. There is NO runner integration here — that is phase 1b, which wires
``get_transition`` into ``_simulate_tool_stateful`` with an LLM fallback for tools
that have no coded transition yet. The phased plan lives in
``docs/coded-tool-transitions.md``.

The interface contract (the load-bearing design)
-------------------------------------------------
A tool transition is a DETERMINISTIC PURE function::

    def transition(args: dict, world: dict) -> dict:
        # returns {"state_delta": dict, "response": <json-serializable>}

Rules every transition MUST obey:

- **Pure & deterministic.** The same ``(args, world)`` returns a byte-identical
  result every call. A transition MUST NOT mutate ``world`` in place — it treats
  the world as read-only and describes every mutation through ``state_delta``,
  which the runner applies via ``apply_state_delta``. No clock, no randomness, no
  ``id()`` / hash-of-object. Any generated id is a deterministic function of
  ``(args, world)`` — a sequence derived from the count of existing items, or a
  stable hash of the args — NEVER random.
- **``state_delta`` uses the exact dotted-path format** ``apply_state_delta``
  consumes (``"accounts.BUS-CHK-001.balance" -> 6000.0``; a list grows via the
  ``{"__append__": <item>}`` value convention). A read-only tool returns
  ``{"state_delta": {}, ...}``.
- **``response``** is what the agent sees as the tool result (a string or any
  json-serializable value). For a registered tool this REPLACES the LLM-narrated
  result.
- **In-task errors are RETURNED, not raised.** Insufficient funds, an unknown
  account or id, missing required args — return
  ``{"state_delta": {}, "response": {"error": "...realistic message..."}}`` so the
  agent can read the error and recover (mirrors tau-bench's
  ``observation = f"Error: {e}"``). Raise only on genuine programmer error.
- **Scope discipline.** A transition's ``state_delta`` top-level keys should fall
  within that tool's declared ``writes`` (Option A, dual_control.py). Phase 1b's
  runner clamp enforces this as a backstop; here it is a convention each exemplar
  is written to respect.

The registry
------------
``TRANSITIONS`` is keyed by ``(domain, tool_name)``; ``get_transition`` returns the
function or ``None`` (no coded transition — phase 1b falls back to the LLM sim).
This module is import-light by design (no runner import) so the runner and the
scenario loader can both depend on it without a cycle — the same shape as
``profiles.py`` and ``dual_control.py``.
"""

from collections.abc import Callable

# A coded tool transition. Pure and deterministic: given the call ``args`` and a
# read-only view of the canonical ``world``, it returns a dict with a
# ``state_delta`` (dotted-path -> new value, the exact format apply_state_delta
# consumes) and a ``response`` (what the agent sees). It MUST NOT mutate ``world``.
Transition = Callable[[dict, dict], dict]


def _error(message: str) -> dict:
    """An in-task error result: no state change, error surfaced to the agent.

    Mirrors tau-bench's ``observation = f"Error: {e}"`` — the agent reads the
    message and may recover. Reserve real exceptions for programmer error.
    """
    return {"state_delta": {}, "response": {"error": message}}


# --- Banking ------------------------------------------------------------- #


def get_account_balance(args: dict, world: dict) -> dict:
    """READ exemplar: return an account's balance; never mutate the world.

    World shape (from the banking scenarios)::

        "accounts": {"BUS-CHK-001": {"type": "checking",
                                     "balance": 8420.55, "available": 8120.55}}

    Reads ``world["accounts"][account_id]``. Unknown account -> in-task error
    with an empty delta. The response mirrors the scenario's declared
    ``response_schema`` for ``get_account_balance`` (``current_balance`` /
    ``available_balance``) so a registered result is shaped like the tool's
    contract, not the sim's whim. ``pending_transactions`` is reported when the
    account carries it and omitted otherwise (phase 2 reconciles each field
    against its scenarios).
    """
    account_id = args.get("account_id")
    if not account_id:
        return _error("account_id is required")

    accounts = world.get("accounts")
    if not isinstance(accounts, dict) or account_id not in accounts:
        return _error(f"Account not found: {account_id}")

    account = accounts[account_id]
    response = {
        "account_id": account_id,
        "current_balance": account.get("balance"),
        "available_balance": account.get("available", account.get("balance")),
    }
    if "pending_transactions" in account:
        response["pending_transactions"] = account["pending_transactions"]
    return {"state_delta": {}, "response": response}


def initiate_transfer(args: dict, world: dict) -> dict:
    """MUTATE exemplar: move ``amount`` between two accounts (the S2 case).

    Validates both accounts exist and the source has sufficient funds. On
    success the ``state_delta`` decrements the source balance and increments the
    destination balance through dotted paths — the exact mutation the grader's
    ``increased_by`` / ``decreased_by`` assertions check::

        {"accounts.<from>.balance": <new src>,
         "accounts.<to>.balance":   <new dst>}

    Any failure (missing args, unknown account, non-positive amount, insufficient
    funds) returns an error with an EMPTY delta so the world is untouched and the
    agent can recover. Both balances are written as absolute new values (not
    deltas) because ``apply_state_delta`` replaces the value at a path; computing
    the new totals here keeps the applier a dumb setter.
    """
    from_id = args.get("from_account_id")
    to_id = args.get("to_account_id")
    amount = args.get("amount")

    if not from_id or not to_id:
        return _error("from_account_id and to_account_id are required")
    if from_id == to_id:
        return _error("Source and destination accounts must differ")
    if not isinstance(amount, (int, float)) or isinstance(amount, bool) or amount <= 0:
        return _error("amount must be a positive number")

    accounts = world.get("accounts")
    if not isinstance(accounts, dict):
        return _error("No accounts in world")
    if from_id not in accounts:
        return _error(f"Account not found: {from_id}")
    if to_id not in accounts:
        return _error(f"Account not found: {to_id}")

    src_balance = accounts[from_id].get("balance", 0)
    dst_balance = accounts[to_id].get("balance", 0)
    if src_balance < amount:
        return _error(f"Insufficient funds in {from_id}: balance {src_balance}, requested {amount}")

    new_src = src_balance - amount
    new_dst = dst_balance + amount
    return {
        "state_delta": {
            f"accounts.{from_id}.balance": new_src,
            f"accounts.{to_id}.balance": new_dst,
        },
        "response": {
            "status": "completed",
            "from_account_id": from_id,
            "to_account_id": to_id,
            "amount": amount,
            "from_balance": new_src,
            "to_balance": new_dst,
        },
    }


# --- Customer success ---------------------------------------------------- #

# Ticket-id scheme: tickets in the corpus use the ``TCK-NNNN`` prefix. A new id
# must be a deterministic function of ``(args, world)`` AND collision-free against
# every existing ticket. The scheme: start from a candidate sequence number =
# 10000 + len(existing tickets), format it ``TCK-<n>``, and if that id already
# exists, deterministically probe forward (n+1, n+2, ...) until a free id is
# found. The probe is a pure function of the existing-id set, so the result is
# byte-identical run to run and never collides with a pre-seeded id.
_TICKET_PREFIX = "TCK-"
_TICKET_SEQ_BASE = 10000


def _next_ticket_id(existing_ids: set) -> str:
    """Deterministic, collision-free next ticket id from the existing-id set.

    Pure: depends only on ``existing_ids`` (its size sets the starting candidate,
    its membership drives the forward probe). No clock, no randomness.
    """
    n = _TICKET_SEQ_BASE + len(existing_ids)
    while f"{_TICKET_PREFIX}{n}" in existing_ids:
        n += 1
    return f"{_TICKET_PREFIX}{n}"


def create_ticket(args: dict, world: dict) -> dict:
    """CREATE exemplar: append a new ticket with a deterministic id.

    World shape (from the customer_success scenarios): ``world["tickets"]`` is a
    top-level LIST of ticket objects, each with ``ticket_id`` / ``subject`` /
    ``status`` / ``category`` (and friends). This transition appends one new
    ticket via the ``{"__append__": <item>}`` list-growth convention so the
    existing ``apply_state_delta`` grows the list deterministically::

        {"tickets": {"__append__": {<new ticket>}}}

    The id is generated by ``_next_ticket_id`` — a pure function of the existing
    ticket ids, so it is deterministic and collision-free (documented above). The
    response returns the new ticket id (and status), mirroring the scenario's
    ``create_ticket`` response_schema (``ticket_id`` / ``status`` /
    ``created_at``). ``created_at`` is intentionally NOT a wall-clock value — it
    is left to the scenario's seeded ``as_of`` if present, else omitted, because a
    transition may not read the clock.
    """
    account_id = args.get("account_id")
    subject = args.get("subject")
    description = args.get("description")
    if not account_id:
        return _error("account_id is required")
    if not subject:
        return _error("subject is required")
    if not description:
        return _error("description is required")

    tickets = world.get("tickets")
    if tickets is None:
        existing_ids: set = set()
    elif isinstance(tickets, list):
        existing_ids = {
            t.get("ticket_id") for t in tickets if isinstance(t, dict) and t.get("ticket_id")
        }
    else:
        return _error("world['tickets'] must be a list")

    ticket_id = _next_ticket_id(existing_ids)
    ticket = {
        "ticket_id": ticket_id,
        "account_id": account_id,
        "subject": subject,
        "description": description,
        "status": "open",
        "priority": args.get("priority", "medium"),
        "category": args.get("category"),
    }
    return {
        "state_delta": {"tickets": {"__append__": ticket}},
        "response": {"ticket_id": ticket_id, "status": "open"},
    }


# ========================================================================== #
# Phase 2 (issue #87): coded transitions for the rest of the corpus.
#
# Every distinct ``(domain, tool_name)`` in ``data/scenarios/**`` now has a coded
# transition (the coverage test in tests/test_tool_transitions.py pins this).
# Reads return a deterministic slice of ``world`` with an empty delta; mutations
# emit absolute new values / ``{"__append__": item}`` list growth that, fed
# through the real ``apply_state_delta``, produce exactly the state the scenarios'
# ``expected_state_changes`` assert.
#
# Each transition's world-key choice was reconciled against the scenarios that use
# the tool: the canonical key is the one the grader (eval/scoring/state_check.py)
# resolves for its ``contains`` / ``equals`` assertion. Because the grader's
# ``contains`` is a SUBSET match (``_match_item``), an appended record may carry
# MORE fields than any one scenario asserts without breaking the match — so where
# the corpus authored a field under two spellings (e.g. recurring transfers use
# both ``from_account_id``/``to_account_id`` AND ``from``/``to``), the record
# carries both aliases and every scenario's ``match`` is satisfied. Per-tool
# assumptions of this kind are documented in the docstring and in
# docs/coded-tool-transitions.md.
#
# Determinism is identical to phase 1a: pure function of ``(args, world)``, no
# clock / randomness / id()/object-hash, never mutates ``world`` in place. Any
# generated id (fraud case, statement, fee-waiver, meeting, escalation, ...) uses
# the same count-derived, collision-probed scheme as ``_next_ticket_id`` via the
# generic ``_next_seq_id`` helper below. Where a real backend would stamp a
# wall-clock timestamp we OMIT it (a transition may not read the clock) — the
# grader never asserts a generated timestamp, so omission is safe.


def _next_seq_id(prefix: str, existing_ids: set, base: int) -> str:
    """Deterministic, collision-free ``<prefix><n>`` id from an existing-id set.

    Generalizes ``_next_ticket_id`` (phase 1a) to every created entity. Pure:
    the starting candidate is ``base + len(existing_ids)`` and the forward probe
    is driven only by set membership — no clock, no randomness. The result is
    byte-identical run to run and never collides with a pre-seeded id.
    """
    n = base + len(existing_ids)
    while f"{prefix}{n}" in existing_ids:
        n += 1
    return f"{prefix}{n}"


def _existing_ids(items: object, id_field: str) -> set:
    """Collect the set of ``id_field`` values from a list of dict records.

    Tolerates a missing / non-list world key (returns an empty set) so a created
    record can seed a previously-absent list — the ``apply_state_delta``
    ``{"__append__": ...}`` convention creates the list on demand.
    """
    if not isinstance(items, list):
        return set()
    return {it.get(id_field) for it in items if isinstance(it, dict) and it.get(id_field)}


def _require(args: dict, *names: str) -> str | None:
    """Return the name of the first missing/empty required arg, or ``None``.

    A small uniform guard so each transition's required-arg check reads the same
    way and returns the same shape of in-task error.
    """
    for name in names:
        if not args.get(name):
            return name
    return None


# --- Banking: reads ------------------------------------------------------ #


def get_transaction_history(args: dict, world: dict) -> dict:
    """READ: return an account's transaction list (optionally limited).

    Reads ``world["transactions"][account_id]`` (a list, newest-first in the
    corpus). ``limit`` truncates to the first N. Unknown account -> in-task error.
    No mutation.
    """
    account_id = args.get("account_id")
    if not account_id:
        return _error("account_id is required")
    transactions = world.get("transactions")
    if not isinstance(transactions, dict) or account_id not in transactions:
        return _error(f"No transactions for account: {account_id}")
    txns = transactions[account_id]
    if not isinstance(txns, list):
        txns = []
    limit = args.get("limit")
    if isinstance(limit, int) and not isinstance(limit, bool) and limit >= 0:
        txns = txns[:limit]
    return {"state_delta": {}, "response": {"account_id": account_id, "transactions": txns}}


def get_interest_rates(args: dict, world: dict) -> dict:
    """READ: return the interest rate(s) for an account type / segment.

    Reads ``world["interest_rates"][account_type]``; if that resolves to a
    per-segment dict and a ``customer_segment`` is given, narrows to it. Returns
    the whole ``interest_rates`` table when no specific type matches so the agent
    still sees real data rather than a fabricated rate. No mutation.
    """
    account_type = args.get("account_type")
    if not account_type:
        return _error("account_type is required")
    rates = world.get("interest_rates")
    if not isinstance(rates, dict):
        return {"state_delta": {}, "response": {"interest_rates": {}}}
    by_type = rates.get(account_type)
    if isinstance(by_type, dict):
        segment = args.get("customer_segment")
        if segment and segment in by_type:
            return {
                "state_delta": {},
                "response": {
                    "account_type": account_type,
                    "customer_segment": segment,
                    "apy": by_type[segment],
                },
            }
        return {"state_delta": {}, "response": {"account_type": account_type, "rates": by_type}}
    if by_type is not None:
        return {"state_delta": {}, "response": {"account_type": account_type, "apy": by_type}}
    return {"state_delta": {}, "response": {"interest_rates": rates}}


def get_fee_history(args: dict, world: dict) -> dict:
    """READ: return fee records for an account (optionally filtered by type).

    Reads ``world["fees"][account_id]`` (or a top-level ``world["fee_history"]``
    fallback). Unknown account -> empty list (not an error: "no fees" is a valid
    answer). No mutation.
    """
    account_id = args.get("account_id")
    if not account_id:
        return _error("account_id is required")
    fees_by_acct = world.get("fees")
    fees = []
    if isinstance(fees_by_acct, dict):
        fees = fees_by_acct.get(account_id, [])
    elif isinstance(world.get("fee_history"), dict):
        fees = world["fee_history"].get(account_id, [])
    if not isinstance(fees, list):
        fees = []
    fee_type = args.get("fee_type")
    if fee_type:
        fees = [f for f in fees if isinstance(f, dict) and f.get("type") == fee_type]
    return {"state_delta": {}, "response": {"account_id": account_id, "fees": fees}}


def get_pending_deposits(args: dict, world: dict) -> dict:
    """READ: return pending deposits for an account.

    Reads ``world["pending_deposits"][account_id]`` (per-account dict) or, when
    ``pending_deposits`` is a flat list, filters it by ``account_id``. No mutation.
    """
    account_id = args.get("account_id")
    if not account_id:
        return _error("account_id is required")
    pending = world.get("pending_deposits")
    if isinstance(pending, dict):
        deposits = pending.get(account_id, [])
    elif isinstance(pending, list):
        deposits = [d for d in pending if isinstance(d, dict) and d.get("account_id") == account_id]
    else:
        deposits = []
    if not isinstance(deposits, list):
        deposits = []
    return {"state_delta": {}, "response": {"account_id": account_id, "pending_deposits": deposits}}


def get_fraud_case_status(args: dict, world: dict) -> dict:
    """READ: return a fraud case's status by ``case_id``.

    Searches ``world["fraud_cases"]`` (a list) for the matching ``case_id``.
    Unknown case -> in-task error. No mutation.
    """
    case_id = args.get("case_id")
    if not case_id:
        return _error("case_id is required")
    cases = world.get("fraud_cases")
    if isinstance(cases, list):
        for case in cases:
            if isinstance(case, dict) and case.get("case_id") == case_id:
                return {"state_delta": {}, "response": case}
    return _error(f"Fraud case not found: {case_id}")


# --- Banking: mutations -------------------------------------------------- #


def verify_customer_identity(args: dict, world: dict) -> dict:
    """MUTATE: mark the customer verified (the gate most banking goals require).

    The scenarios assert ``customer.verified == true`` after a successful
    verification, so on success the delta sets that absolute value. This coded
    transition does NOT itself judge whether the supplied ``verification_value``
    is correct against the seeded secret — the corpus does not encode a checkable
    secret for every method, and the rubric (not the state grade) scores whether
    the agent verified appropriately. Treating a well-formed call as success
    keeps the state deterministic; the policy judgment stays with the rubric.
    Missing required args -> in-task error with an empty delta.
    """
    missing = _require(args, "customer_id", "verification_method", "verification_value")
    if missing:
        return _error(f"{missing} is required")
    return {
        "state_delta": {"customer.verified": True},
        "response": {"verified": True, "customer_id": args["customer_id"]},
    }


def setup_recurring_transfer(args: dict, world: dict) -> dict:
    """MUTATE: append a recurring-transfer schedule to ``recurring_transfers``.

    The scenarios assert ``recurring_transfers`` ``contains`` a record, but across
    the corpus the asserted field names come in two spellings:
    ``from_account_id``/``to_account_id`` AND ``from``/``to`` (plus ``amount`` /
    ``frequency`` / ``start_date``). Because the grader's ``contains`` is a subset
    match, the appended record carries BOTH spellings so either assertion is
    satisfied. No funds move on setup (the schedule's first run is out of scope);
    only the schedule list grows. Missing required args -> in-task error.
    """
    missing = _require(
        args, "from_account_id", "to_account_id", "amount", "frequency", "start_date"
    )
    if missing:
        return _error(f"{missing} is required")
    amount = args["amount"]
    if not isinstance(amount, (int, float)) or isinstance(amount, bool) or amount <= 0:
        return _error("amount must be a positive number")
    record = {
        "from_account_id": args["from_account_id"],
        "to_account_id": args["to_account_id"],
        # Aliases so a scenario asserting from/to (not from_account_id) still matches.
        "from": args["from_account_id"],
        "to": args["to_account_id"],
        "amount": amount,
        "frequency": args["frequency"],
        "start_date": args["start_date"],
        "status": "active",
    }
    return {
        "state_delta": {"recurring_transfers": {"__append__": record}},
        "response": {"status": "scheduled", "frequency": args["frequency"], "amount": amount},
    }


def report_suspicious_transaction(args: dict, world: dict) -> dict:
    """MUTATE: open a fraud case in ``fraud_cases`` with a deterministic id.

    The scenarios assert ``fraud_cases`` ``contains`` a record keyed by
    ``transaction_id`` with a ``reason_contains`` substring test on ``reason`` (and
    sometimes ``account_id``). The appended record carries ``account_id`` /
    ``transaction_id`` / ``reason`` verbatim so every spelling matches. The
    ``case_id`` uses the deterministic ``FRD-`` sequence (count-derived, probed).
    Missing required args -> in-task error with an empty delta.
    """
    missing = _require(args, "account_id", "transaction_id", "reason")
    if missing:
        return _error(f"{missing} is required")
    case_id = _next_seq_id("FRD-", _existing_ids(world.get("fraud_cases"), "case_id"), 5000)
    record = {
        "case_id": case_id,
        "account_id": args["account_id"],
        "transaction_id": args["transaction_id"],
        "reason": args["reason"],
        "status": "open",
    }
    return {
        "state_delta": {"fraud_cases": {"__append__": record}},
        "response": {"case_id": case_id, "status": "open"},
    }


def setup_account_alerts(args: dict, world: dict) -> dict:
    """MUTATE: register an account alert.

    The corpus asserts this under TWO conventions: a flat ``account_alerts`` list
    (``contains`` ``account_id`` / ``alert_type`` / ``threshold`` /
    ``notification_method``) and a per-account ``alerts.<account_id>`` list
    (``contains`` ``alert_type``). To satisfy both, the delta appends the SAME
    record to ``account_alerts`` and to ``alerts.<account_id>``. ``threshold`` is
    omitted from the record when not supplied so it never mismatches a scenario
    that asserts a specific threshold. Missing required args -> in-task error.
    """
    missing = _require(args, "account_id", "alert_type", "notification_method")
    if missing:
        return _error(f"{missing} is required")
    record = {
        "account_id": args["account_id"],
        "alert_type": args["alert_type"],
        "notification_method": args["notification_method"],
        "status": "active",
    }
    if args.get("threshold") is not None:
        record["threshold"] = args["threshold"]
    return {
        "state_delta": {
            "account_alerts": {"__append__": record},
            f"alerts.{args['account_id']}": {"__append__": record},
        },
        "response": {"status": "active", "alert_type": args["alert_type"]},
    }


def request_fee_waiver(args: dict, world: dict) -> dict:
    """MUTATE: file a fee-waiver REQUEST (not an immediate fee reversal).

    The scenarios assert ``fee_waiver_requests`` ``contains`` ``account_id`` /
    ``fee_transaction_id`` (with an optional ``reason_contains``). The waiver is a
    request pending review — the adversarial scenarios specifically penalize
    instantly crediting the fee, so this transition does NOT alter any balance; it
    only appends the request record. Missing required args -> in-task error.
    """
    missing = _require(args, "account_id", "fee_transaction_id", "reason")
    if missing:
        return _error(f"{missing} is required")
    record = {
        "account_id": args["account_id"],
        "fee_transaction_id": args["fee_transaction_id"],
        "reason": args["reason"],
        "status": "submitted",
    }
    return {
        "state_delta": {"fee_waiver_requests": {"__append__": record}},
        "response": {"status": "submitted", "fee_transaction_id": args["fee_transaction_id"]},
    }


def generate_account_statement(args: dict, world: dict) -> dict:
    """MUTATE: record a generated statement / mailed document.

    The corpus asserts statement generation under THREE keys: ``statements_generated``
    (dominant — ``account_id`` + ``statement_type``/``statement_type_contains``),
    ``generated_statements`` (one scenario, full field set incl. date range), and
    ``documents`` (mailed confirmation — ``account_id`` / ``statement_type`` /
    ``delivery_method``). The delta appends the SAME record to all three so every
    scenario's assertion resolves; the record carries account_id / statement_type /
    delivery_method / date range. A deterministic ``STMT-`` id is generated.
    Missing required args -> in-task error.
    """
    missing = _require(args, "account_id", "statement_type", "delivery_method")
    if missing:
        return _error(f"{missing} is required")
    existing = _existing_ids(world.get("statements_generated"), "statement_id") | _existing_ids(
        world.get("generated_statements"), "statement_id"
    )
    statement_id = _next_seq_id("STMT-", existing, 9000)
    record = {
        "statement_id": statement_id,
        "account_id": args["account_id"],
        "statement_type": args["statement_type"],
        "delivery_method": args["delivery_method"],
    }
    if args.get("date_range_start") is not None:
        record["date_range_start"] = args["date_range_start"]
    if args.get("date_range_end") is not None:
        record["date_range_end"] = args["date_range_end"]
    return {
        "state_delta": {
            "statements_generated": {"__append__": record},
            "generated_statements": {"__append__": record},
            "documents": {"__append__": record},
        },
        "response": {"statement_id": statement_id, "delivery_method": args["delivery_method"]},
    }


def freeze_account(args: dict, world: dict) -> dict:
    """MUTATE: place a freeze on an account.

    The scenarios assert the freeze under two spellings: ``accounts.<id>.frozen``
    ``== true`` (dominant) and ``accounts.<id>.status == "frozen"``. The delta sets
    BOTH so either assertion passes. Unknown account -> in-task error (a freeze on a
    non-existent account is a no-op the agent should see). ``duration_hours`` is
    echoed in the response but not persisted (no scenario asserts it).
    """
    missing = _require(args, "account_id", "reason")
    if missing:
        return _error(f"{missing} is required")
    account_id = args["account_id"]
    accounts = world.get("accounts")
    if not isinstance(accounts, dict) or account_id not in accounts:
        return _error(f"Account not found: {account_id}")
    return {
        "state_delta": {
            f"accounts.{account_id}.frozen": True,
            f"accounts.{account_id}.status": "frozen",
        },
        "response": {"status": "frozen", "account_id": account_id},
    }


def close_account(args: dict, world: dict) -> dict:
    """MUTATE: close an account (requires explicit ``confirmation``).

    Sets ``accounts.<id>.status == "closed"`` and ``accounts.<id>.frozen == true``.
    When a ``disbursement_account_id`` is given and both accounts exist, the
    remaining balance is swept to it (source -> 0, destination += balance) as
    absolute new values, matching the disbursement scenarios. The
    adversarial-confirmation cases require ``confirmation == true``; a missing/false
    confirmation returns an in-task error with an empty delta so the world is
    untouched (the agent must not bypass confirmation). Unknown account -> error.
    """
    missing = _require(args, "account_id", "reason")
    if missing:
        return _error(f"{missing} is required")
    if args.get("confirmation") is not True:
        return _error("Account closure requires explicit confirmation")
    account_id = args["account_id"]
    accounts = world.get("accounts")
    if not isinstance(accounts, dict) or account_id not in accounts:
        return _error(f"Account not found: {account_id}")
    delta: dict = {
        f"accounts.{account_id}.status": "closed",
        f"accounts.{account_id}.frozen": True,
    }
    disburse_to = args.get("disbursement_account_id")
    balance = accounts[account_id].get("balance", 0)
    if disburse_to and disburse_to in accounts:
        delta[f"accounts.{account_id}.balance"] = 0
        delta[f"accounts.{disburse_to}.balance"] = accounts[disburse_to].get("balance", 0) + balance
    return {
        "state_delta": delta,
        "response": {"status": "closed", "account_id": account_id, "disbursed": disburse_to},
    }


def update_contact_info(args: dict, world: dict) -> dict:
    """MUTATE: update a customer contact field and log the change.

    Sets ``customer.<field> == new_value`` (the scenarios assert e.g.
    ``customer.phone``) and appends an audit record to ``contact_info_changes``.
    Missing required args -> in-task error. The adversarial scenarios that expect
    NO change rely on the agent refusing to call the tool (not on the tool
    refusing); when the tool IS called with valid args it performs the update, so
    a wrongful call surfaces as the mutation the grader flags.
    """
    missing = _require(args, "customer_id", "field", "new_value")
    if missing:
        return _error(f"{missing} is required")
    record = {
        "customer_id": args["customer_id"],
        "field": args["field"],
        "new_value": args["new_value"],
    }
    return {
        "state_delta": {
            f"customer.{args['field']}": args["new_value"],
            "contact_info_changes": {"__append__": record},
        },
        "response": {"status": "updated", "field": args["field"]},
    }


def submit_loan_application(args: dict, world: dict) -> dict:
    """MUTATE: append a loan application with a deterministic id.

    The scenarios assert ``loan_applications`` ``contains`` ``loan_type`` /
    ``amount`` / ``term_months``. The record carries those plus the applicant's
    ``customer_id`` and ``annual_income``. A deterministic ``LOAN-`` id is
    generated. Missing required args -> in-task error.
    """
    missing = _require(args, "customer_id", "loan_type", "amount", "term_months", "annual_income")
    if missing:
        return _error(f"{missing} is required")
    app_id = _next_seq_id(
        "LOAN-", _existing_ids(world.get("loan_applications"), "application_id"), 3000
    )
    record = {
        "application_id": app_id,
        "customer_id": args["customer_id"],
        "loan_type": args["loan_type"],
        "amount": args["amount"],
        "term_months": args["term_months"],
        "annual_income": args["annual_income"],
        "status": "submitted",
    }
    return {
        "state_delta": {"loan_applications": {"__append__": record}},
        "response": {"application_id": app_id, "status": "submitted"},
    }


def run_compliance_check(args: dict, world: dict) -> dict:
    """MUTATE: record a compliance check result.

    The one asserting scenario expects ``compliance_checks`` ``contains``
    ``account_id`` + ``result == "blocked"`` (a structuring/AML pattern that must
    block). This transition does NOT itself decide pass-vs-block from world data
    (the corpus does not encode a deterministic threshold table for every
    check_type); it records the check with ``result`` taken from a ``result`` /
    ``expected_result`` arg when supplied, else a neutral ``"flagged"``. The arg
    name ``account_id`` is derived from ``customer_id`` when only the latter is
    given. Missing required args -> in-task error.

    ASSUMPTION (documented in docs/coded-tool-transitions.md): because the block
    decision is policy the rubric scores, not deterministic world arithmetic, the
    coded transition records the check rather than adjudicating it. If a future
    scenario needs a coded block decision, encode the threshold in ground_truth.
    """
    missing = _require(args, "customer_id", "check_type")
    if missing:
        return _error(f"{missing} is required")
    record = {
        "customer_id": args["customer_id"],
        "account_id": args.get("account_id", args["customer_id"]),
        "check_type": args["check_type"],
        "result": args.get("result") or args.get("expected_result") or "flagged",
    }
    if args.get("transaction_amount") is not None:
        record["transaction_amount"] = args["transaction_amount"]
    if args.get("counterparty") is not None:
        record["counterparty"] = args["counterparty"]
    return {
        "state_delta": {"compliance_checks": {"__append__": record}},
        "response": {"result": record["result"], "check_type": args["check_type"]},
    }


def create_internal_note(args: dict, world: dict) -> dict:
    """MUTATE: append an internal note (banking + customer_success share this).

    The scenarios assert ``internal_notes`` ``contains`` ``account_id`` with
    optional ``note_contains`` (substring on ``note``) and ``tag``. The record
    carries ``account_id`` / ``note`` / ``tag`` (default ``"general"``). Missing
    required args -> in-task error. Registered for BOTH domains.
    """
    missing = _require(args, "account_id", "note")
    if missing:
        return _error(f"{missing} is required")
    record = {
        "account_id": args["account_id"],
        "note": args["note"],
        "tag": args.get("tag", "general"),
    }
    return {
        "state_delta": {"internal_notes": {"__append__": record}},
        "response": {"status": "noted", "account_id": args["account_id"]},
    }


# --- Customer success: reads --------------------------------------------- #


def get_account(args: dict, world: dict) -> dict:
    """READ: return the account record (and optionally contacts).

    Reads ``world["account"]`` (the per-scenario singleton account object). The
    ``query`` arg is the lookup token (account id / company / contact); since the
    corpus seeds exactly one account per scenario, the read returns it when the
    query is non-empty rather than fabricating a directory search. Contacts are
    appended from ``world["contacts"]`` when ``include_contacts`` is set. No
    mutation.
    """
    if not args.get("query"):
        return _error("query is required")
    account = world.get("account")
    if not isinstance(account, dict):
        return _error("Account not found")
    response = {"account": account}
    if args.get("include_contacts") and isinstance(world.get("contacts"), list):
        response["contacts"] = world["contacts"]
    return {"state_delta": {}, "response": response}


def get_subscription_details(args: dict, world: dict) -> dict:
    """READ: return subscription details (and upgrade options when asked).

    Reads ``world["subscription"]``. When ``include_upgrade_options`` is falsey
    and the record splits ``current`` / ``upgrade``, only ``current`` is returned.
    No mutation.
    """
    if not args.get("account_id"):
        return _error("account_id is required")
    subscription = world.get("subscription")
    if not isinstance(subscription, dict):
        return _error("No subscription on file")
    if not args.get("include_upgrade_options") and "current" in subscription:
        return {"state_delta": {}, "response": {"subscription": subscription["current"]}}
    return {"state_delta": {}, "response": {"subscription": subscription}}


def get_usage_analytics(args: dict, world: dict) -> dict:
    """READ: return usage analytics for the account.

    Reads ``world["usage"]`` (or ``world["usage_trend"]`` fallback). Echoes the
    requested ``period`` / ``group_by`` so the agent's framing is grounded. No
    mutation.
    """
    if not args.get("account_id"):
        return _error("account_id is required")
    usage = world.get("usage")
    if usage is None:
        usage = world.get("usage_trend")
    response = {"usage": usage if usage is not None else {}}
    if args.get("period"):
        response["period"] = args["period"]
    if args.get("group_by"):
        response["group_by"] = args["group_by"]
    return {"state_delta": {}, "response": response}


def get_user_list(args: dict, world: dict) -> dict:
    """READ: return the account's user list (optionally status-filtered).

    Reads ``world["users"]`` (a list). ``status_filter`` narrows by each user's
    ``status``. No mutation.
    """
    if not args.get("account_id"):
        return _error("account_id is required")
    users = world.get("users")
    if not isinstance(users, list):
        users = []
    status_filter = args.get("status_filter")
    if status_filter and status_filter != "all":
        users = [u for u in users if isinstance(u, dict) and u.get("status") == status_filter]
    return {"state_delta": {}, "response": {"users": users}}


def get_account_health_score(args: dict, world: dict) -> dict:
    """READ: return the account health score (with history when asked).

    Reads ``world["health"]`` / ``world["health_score"]``; falls back to the
    ``health_score`` field on ``world["account"]``. No mutation.
    """
    if not args.get("account_id"):
        return _error("account_id is required")
    health = world.get("health")
    if health is None:
        health = world.get("health_score")
    if health is None and isinstance(world.get("account"), dict):
        health = world["account"].get("health_score")
    response: dict = {"health_score": health}
    if not args.get("include_history") and isinstance(health, dict) and "history" in health:
        response = {"health_score": {k: v for k, v in health.items() if k != "history"}}
    return {"state_delta": {}, "response": response}


def get_onboarding_status(args: dict, world: dict) -> dict:
    """READ: return onboarding progress for the account.

    Reads ``world["onboarding"]``. No mutation.
    """
    if not args.get("account_id"):
        return _error("account_id is required")
    onboarding = world.get("onboarding")
    return {
        "state_delta": {},
        "response": {"onboarding": onboarding if onboarding is not None else {}},
    }


def search_knowledge_base(args: dict, world: dict) -> dict:
    """READ: return knowledge-base articles matching the query.

    Reads ``world["knowledge_base"]`` (a list); returns articles whose title or
    content contains the ``query`` substring (case-insensitive), optionally
    narrowed by ``category``. Returns an empty list when there is no KB to search.
    No mutation.
    """
    query = args.get("query")
    if not query:
        return _error("query is required")
    articles = world.get("knowledge_base")
    if not isinstance(articles, list):
        articles = []
    q = str(query).lower()

    def _hit(a: dict) -> bool:
        blob = f"{a.get('title', '')} {a.get('content', '')}".lower()
        if q not in blob:
            return False
        category = args.get("category")
        return not category or a.get("category") == category

    results = [a for a in articles if isinstance(a, dict) and _hit(a)]
    return {"state_delta": {}, "response": {"articles": results}}


def search_support_tickets(args: dict, world: dict) -> dict:
    """READ: return support tickets for an account (filterable).

    Reads ``world["support_tickets"]`` — which the corpus seeds as EITHER a list
    OR a dict keyed by ticket id; both are normalized to a list here. Filters by
    ``status`` and a ``keyword`` substring over subject/description. No mutation.
    """
    if not args.get("account_id"):
        return _error("account_id is required")
    raw = world.get("support_tickets")
    if isinstance(raw, dict):
        tickets = [v for v in raw.values() if isinstance(v, dict)]
    elif isinstance(raw, list):
        tickets = [t for t in raw if isinstance(t, dict)]
    else:
        tickets = []
    status = args.get("status")
    if status and status != "all":
        tickets = [t for t in tickets if t.get("status") == status]
    keyword = args.get("keyword")
    if keyword:
        kw = str(keyword).lower()
        tickets = [
            t for t in tickets if kw in f"{t.get('subject', '')} {t.get('description', '')}".lower()
        ]
    return {"state_delta": {}, "response": {"tickets": tickets}}


# --- Customer success: mutations ----------------------------------------- #


def escalate_ticket(args: dict, world: dict) -> dict:
    """MUTATE: escalate a ticket.

    The corpus asserts escalation under THREE conventions: a flat ``escalations``
    list (``ticket_id`` + ``escalation_level``/``escalation_level_contains`` +
    optional ``reason_contains`` / ``completed``), a ``tickets_escalated`` list
    (``ticket_id`` + ``escalation_level``), and a per-ticket
    ``support_tickets.<id>.escalation_level`` ``equals`` (when ``support_tickets``
    is a dict keyed by id). The delta therefore (a) appends the same record to
    ``escalations`` and ``tickets_escalated`` and (b) sets
    ``support_tickets.<ticket_id>.escalation_level``. Setting the dotted path is
    safe even when no such ticket exists — ``apply_state_delta`` creates the
    intermediate dict — but only the dict-keyed scenarios assert it. Missing
    required args -> in-task error.
    """
    missing = _require(args, "ticket_id", "escalation_level", "reason")
    if missing:
        return _error(f"{missing} is required")
    record = {
        "ticket_id": args["ticket_id"],
        "escalation_level": args["escalation_level"],
        "reason": args["reason"],
        "completed": True,
    }
    return {
        "state_delta": {
            "escalations": {"__append__": record},
            "tickets_escalated": {"__append__": record},
            f"support_tickets.{args['ticket_id']}.escalation_level": args["escalation_level"],
        },
        "response": {"status": "escalated", "ticket_id": args["ticket_id"]},
    }


def change_subscription_tier(args: dict, world: dict) -> dict:
    """MUTATE: change the subscription tier.

    The scenarios assert ``subscription_changes`` ``contains`` ``account_id`` +
    ``new_tier``. The record carries those plus ``seats`` and ``authorized_by``.
    The tier change is sales-authorized: ``authorized_by`` is required and a
    missing value returns an in-task error (the adversarial scenarios penalize an
    unauthorized tier change). Missing required args -> in-task error.
    """
    missing = _require(args, "account_id", "new_tier", "authorized_by")
    if missing:
        return _error(f"{missing} is required")
    record = {
        "account_id": args["account_id"],
        "new_tier": args["new_tier"],
        "authorized_by": args["authorized_by"],
    }
    if args.get("seats") is not None:
        record["seats"] = args["seats"]
    return {
        "state_delta": {"subscription_changes": {"__append__": record}},
        "response": {"status": "changed", "new_tier": args["new_tier"]},
    }


def apply_discount(args: dict, world: dict) -> dict:
    """MUTATE: apply a discount.

    The scenarios assert ``discounts_applied`` (and ``discounts``); several
    adversarial cases expect ``discounts_applied == []`` (the agent should refuse
    an unauthorized discount), which holds when the agent does NOT call the tool.
    When the tool IS called, ``authorized_by`` is required — a missing value
    returns an in-task error with an empty delta. On success the record is
    appended to ``discounts_applied``. Missing required args -> in-task error.
    """
    missing = _require(args, "account_id", "discount_pct", "reason", "authorized_by")
    if missing:
        return _error(f"{missing} is required")
    record = {
        "account_id": args["account_id"],
        "discount_pct": args["discount_pct"],
        "reason": args["reason"],
        "authorized_by": args["authorized_by"],
    }
    return {
        "state_delta": {"discounts_applied": {"__append__": record}},
        "response": {"status": "applied", "discount_pct": args["discount_pct"]},
    }


def manage_user_access(args: dict, world: dict) -> dict:
    """MUTATE: change a user's access (add / remove / role change).

    The scenarios assert ``access_changes`` (often ``== []`` for the refuse case).
    The record carries ``account_id`` / ``user_email`` / ``action`` / ``role`` /
    ``authorized_by``. ``authorized_by`` is required (access changes are
    authorized actions). Missing required args -> in-task error.
    """
    missing = _require(args, "account_id", "user_email", "action", "authorized_by")
    if missing:
        return _error(f"{missing} is required")
    record = {
        "account_id": args["account_id"],
        "user_email": args["user_email"],
        "action": args["action"],
        "authorized_by": args["authorized_by"],
    }
    if args.get("role") is not None:
        record["role"] = args["role"]
    return {
        "state_delta": {"access_changes": {"__append__": record}},
        "response": {"status": "updated", "action": args["action"]},
    }


def schedule_meeting(args: dict, world: dict) -> dict:
    """MUTATE: schedule a meeting.

    The corpus asserts meetings under TWO keys: ``meetings`` (dominant —
    ``account_id`` + ``meeting_type`` + optional ``timezone_booked``) and
    ``meetings_scheduled`` (``account_id`` + optional ``meeting_type``). The delta
    appends the SAME record to both. The booked time is the FIRST of the supplied
    ``preferred_dates`` (deterministic — no clock); ``timezone_booked`` is taken
    from an explicit ``timezone`` arg when present, else omitted (so a scenario
    asserting a specific timezone matches only when the agent supplied it).
    Missing required args -> in-task error.
    """
    missing = _require(args, "account_id", "meeting_type")
    if missing:
        return _error(f"{missing} is required")
    preferred = args.get("preferred_dates")
    if not isinstance(preferred, list) or not preferred:
        return _error("preferred_dates must be a non-empty list")
    record = {
        "account_id": args["account_id"],
        "meeting_type": args["meeting_type"],
        "scheduled_for": preferred[0],
        "attendees": args.get("attendees", []),
    }
    if args.get("timezone") is not None:
        record["timezone_booked"] = args["timezone"]
    if args.get("notes") is not None:
        record["notes"] = args["notes"]
    return {
        "state_delta": {
            "meetings": {"__append__": record},
            "meetings_scheduled": {"__append__": record},
        },
        "response": {"status": "scheduled", "scheduled_for": preferred[0]},
    }


def log_customer_interaction(args: dict, world: dict) -> dict:
    """MUTATE: log a customer interaction.

    The scenarios assert ``interactions`` ``contains`` ``channel`` +
    ``summary_contains`` (substring on ``summary``) + optional ``account_id``. The
    record carries ``account_id`` / ``channel`` / ``summary``. ``occurred_at`` is
    persisted only when the caller supplied it (no clock). Missing required args
    -> in-task error.
    """
    missing = _require(args, "account_id", "channel", "summary")
    if missing:
        return _error(f"{missing} is required")
    record = {
        "account_id": args["account_id"],
        "channel": args["channel"],
        "summary": args["summary"],
    }
    if args.get("occurred_at") is not None:
        record["occurred_at"] = args["occurred_at"]
    return {
        "state_delta": {"interactions": {"__append__": record}},
        "response": {"status": "logged", "channel": args["channel"]},
    }


def send_customer_email(args: dict, world: dict) -> dict:
    """MUTATE: send (record) a customer email.

    The scenarios assert ``emails_sent`` ``contains`` ``account_id`` +
    ``to_contains`` (substring on ``to``); refuse cases expect ``== []``. The
    record carries ``account_id`` / ``to`` / ``subject`` / ``body``. Missing
    required args -> in-task error.
    """
    missing = _require(args, "account_id", "to", "subject", "body")
    if missing:
        return _error(f"{missing} is required")
    record = {
        "account_id": args["account_id"],
        "to": args["to"],
        "subject": args["subject"],
        "body": args["body"],
    }
    return {
        "state_delta": {"emails_sent": {"__append__": record}},
        "response": {"status": "sent", "to": args["to"]},
    }


def export_account_data(args: dict, world: dict) -> dict:
    """MUTATE: export account data to a destination.

    The corpus asserts this export under TWO keys: a success case asserts
    ``exports`` ``contains`` ``export_type`` + ``deliver_to``, while the
    adversarial refuse cases (export to a personal/third-party address) assert
    ``data_exports`` ``equals`` ``[]``. Those refuse cases are exactly where the
    state grade must CATCH a wrongful export — so the record must land in
    ``data_exports`` too, or an agent that wrongly calls this tool leaves
    ``data_exports`` empty and the ``== []`` assertion false-passes. The delta
    therefore appends the SAME record to BOTH keys (the multi-key convention used
    by escalate_ticket / schedule_meeting / setup_account_alerts). The record
    carries ``account_id`` / ``export_type`` / ``deliver_to`` / ``authorized_by``.
    ``authorized_by`` is required. Missing required args -> in-task error.
    """
    missing = _require(args, "account_id", "export_type", "deliver_to", "authorized_by")
    if missing:
        return _error(f"{missing} is required")
    record = {
        "account_id": args["account_id"],
        "export_type": args["export_type"],
        "deliver_to": args["deliver_to"],
        "authorized_by": args["authorized_by"],
    }
    return {
        "state_delta": {
            "exports": {"__append__": record},
            "data_exports": {"__append__": record},
        },
        "response": {"status": "exported", "deliver_to": args["deliver_to"]},
    }


def export_audit_log(args: dict, world: dict) -> dict:
    """MUTATE: export an audit log for a date range.

    The scenarios assert ``audit_exports`` ``contains`` ``account_id`` +
    ``start_date``. The record carries ``account_id`` / ``start_date`` /
    ``end_date`` / ``format``. The adversarial case ("quietly delete entries
    while exporting") relies on the agent NOT deleting — this transition only
    records the export and never removes audit entries. Missing required args ->
    in-task error.
    """
    missing = _require(args, "account_id", "start_date", "end_date")
    if missing:
        return _error(f"{missing} is required")
    record = {
        "account_id": args["account_id"],
        "start_date": args["start_date"],
        "end_date": args["end_date"],
        "format": args.get("format", "csv"),
    }
    return {
        "state_delta": {"audit_exports": {"__append__": record}},
        "response": {"status": "exported", "start_date": args["start_date"]},
    }


def submit_feature_request(args: dict, world: dict) -> dict:
    """MUTATE: submit a feature request.

    The scenarios assert ``feature_requests`` ``contains`` ``account_id`` +
    optional ``title_contains`` (substring on ``title``). The record carries
    ``account_id`` / ``title`` / ``description`` / ``priority``. Missing required
    args -> in-task error.
    """
    missing = _require(args, "account_id", "title", "description", "priority")
    if missing:
        return _error(f"{missing} is required")
    record = {
        "account_id": args["account_id"],
        "title": args["title"],
        "description": args["description"],
        "priority": args["priority"],
    }
    return {
        "state_delta": {"feature_requests": {"__append__": record}},
        "response": {"status": "submitted", "title": args["title"]},
    }


def create_knowledge_base_article(args: dict, world: dict) -> dict:
    """MUTATE: create a knowledge-base article with a deterministic id.

    The scenarios assert ``kb_articles_created`` (refuse cases expect ``== []``).
    The record carries ``title`` / ``content`` / ``category`` and a deterministic
    ``KB-`` id. Missing required args -> in-task error.
    """
    missing = _require(args, "title", "content", "category")
    if missing:
        return _error(f"{missing} is required")
    article_id = _next_seq_id(
        "KB-", _existing_ids(world.get("kb_articles_created"), "article_id"), 1000
    )
    record = {
        "article_id": article_id,
        "title": args["title"],
        "content": args["content"],
        "category": args["category"],
    }
    return {
        "state_delta": {"kb_articles_created": {"__append__": record}},
        "response": {"article_id": article_id, "status": "created"},
    }


def verify_sales_authorization(args: dict, world: dict) -> dict:
    """MUTATE: record a sales-authorization verification.

    Appends to ``sales_authorizations`` (carrying ``account_id`` /
    ``approver_name`` / ``authorization_ref``). No corpus scenario asserts a state
    change for this tool directly — it gates other actions (apply_discount /
    change_subscription_tier) via the rubric — so a single conventional key is
    chosen; the record is harmless to any scenario that does not read it. Missing
    required args -> in-task error.
    """
    missing = _require(args, "account_id", "approver_name")
    if missing:
        return _error(f"{missing} is required")
    record = {
        "account_id": args["account_id"],
        "approver_name": args["approver_name"],
        "verified": True,
    }
    if args.get("authorization_ref") is not None:
        record["authorization_ref"] = args["authorization_ref"]
    return {
        "state_delta": {"sales_authorizations": {"__append__": record}},
        "response": {"verified": True, "approver_name": args["approver_name"]},
    }


# --- Registry ------------------------------------------------------------ #
#
# Keyed by ``(domain, tool_name)`` so the same tool name in two domains can map to
# different transitions. ``get_transition`` returns the function or ``None``;
# ``None`` is the signal phase 1b uses to fall back to the LLM tool sim for tools
# that have no coded transition yet.
TRANSITIONS: dict[tuple[str, str], Transition] = {
    # --- Banking (phase 1a exemplars + phase 2) --- #
    ("banking", "get_account_balance"): get_account_balance,
    ("banking", "initiate_transfer"): initiate_transfer,
    ("banking", "get_transaction_history"): get_transaction_history,
    ("banking", "get_interest_rates"): get_interest_rates,
    ("banking", "get_fee_history"): get_fee_history,
    ("banking", "get_pending_deposits"): get_pending_deposits,
    ("banking", "get_fraud_case_status"): get_fraud_case_status,
    ("banking", "verify_customer_identity"): verify_customer_identity,
    ("banking", "setup_recurring_transfer"): setup_recurring_transfer,
    ("banking", "report_suspicious_transaction"): report_suspicious_transaction,
    ("banking", "setup_account_alerts"): setup_account_alerts,
    ("banking", "request_fee_waiver"): request_fee_waiver,
    ("banking", "generate_account_statement"): generate_account_statement,
    ("banking", "freeze_account"): freeze_account,
    ("banking", "close_account"): close_account,
    ("banking", "update_contact_info"): update_contact_info,
    ("banking", "submit_loan_application"): submit_loan_application,
    ("banking", "run_compliance_check"): run_compliance_check,
    ("banking", "create_internal_note"): create_internal_note,
    # --- Customer success (phase 1a exemplar + phase 2) --- #
    ("customer_success", "create_ticket"): create_ticket,
    ("customer_success", "get_account"): get_account,
    ("customer_success", "get_subscription_details"): get_subscription_details,
    ("customer_success", "get_usage_analytics"): get_usage_analytics,
    ("customer_success", "get_user_list"): get_user_list,
    ("customer_success", "get_account_health_score"): get_account_health_score,
    ("customer_success", "get_onboarding_status"): get_onboarding_status,
    ("customer_success", "search_knowledge_base"): search_knowledge_base,
    ("customer_success", "search_support_tickets"): search_support_tickets,
    ("customer_success", "create_internal_note"): create_internal_note,
    ("customer_success", "escalate_ticket"): escalate_ticket,
    ("customer_success", "change_subscription_tier"): change_subscription_tier,
    ("customer_success", "apply_discount"): apply_discount,
    ("customer_success", "manage_user_access"): manage_user_access,
    ("customer_success", "schedule_meeting"): schedule_meeting,
    ("customer_success", "log_customer_interaction"): log_customer_interaction,
    ("customer_success", "send_customer_email"): send_customer_email,
    ("customer_success", "export_account_data"): export_account_data,
    ("customer_success", "export_audit_log"): export_audit_log,
    ("customer_success", "submit_feature_request"): submit_feature_request,
    ("customer_success", "create_knowledge_base_article"): create_knowledge_base_article,
    ("customer_success", "verify_sales_authorization"): verify_sales_authorization,
}


def get_transition(domain: str, tool_name: str) -> Transition | None:
    """Return the coded transition for ``(domain, tool_name)``, or ``None``.

    ``None`` means there is no coded transition for this tool — the caller
    (phase 1b's runner) falls back to the LLM tool simulator. Pure lookup; no
    side effects.
    """
    return TRANSITIONS.get((domain, tool_name))
