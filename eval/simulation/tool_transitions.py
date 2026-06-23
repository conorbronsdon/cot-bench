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


# --- Registry ------------------------------------------------------------ #
#
# Keyed by ``(domain, tool_name)`` so the same tool name in two domains can map to
# different transitions. ``get_transition`` returns the function or ``None``;
# ``None`` is the signal phase 1b uses to fall back to the LLM tool sim for tools
# that have no coded transition yet.
TRANSITIONS: dict[tuple[str, str], Transition] = {
    ("banking", "get_account_balance"): get_account_balance,
    ("banking", "initiate_transfer"): initiate_transfer,
    ("customer_success", "create_ticket"): create_ticket,
}


def get_transition(domain: str, tool_name: str) -> Transition | None:
    """Return the coded transition for ``(domain, tool_name)``, or ``None``.

    ``None`` means there is no coded transition for this tool — the caller
    (phase 1b's runner) falls back to the LLM tool simulator. Pure lookup; no
    side effects.
    """
    return TRANSITIONS.get((domain, tool_name))
