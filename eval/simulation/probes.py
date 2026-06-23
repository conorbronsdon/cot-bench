"""Recovery probes — deterministic mid-conversation perturbations (issue #57).

From the game-benchmarks research (idea 4; Vending-Bench): long-horizon
*coherence* failures are goal-maintenance failures triggered by **unrecovered
errors**, with near-zero correlation to context saturation. An agent does not
drift because its context filled up; it drifts because something went wrong
mid-task and it never got back on track. Measuring that within COT Bench's
10-turn cap needs an *injected* perturbation, not a longer context: drop a
deliberate fault into the conversation at turn 4-5 and check — deterministically,
by state grading — whether the agent has recovered the correct end state by the
time the conversation closes.

A recovery probe is a small, OPTIONAL block on a scenario::

    "recovery_probe": {
      "turn": 4,
      "kind": "wrong_entity",
      "injection": "Actually, send it to account BUS-CHK-999 instead.",
      "recovery_assertions": [
        {"assert": "accounts.BUS-CHK-999", "op": "equals", "value": null}
      ]
    }

The probe is **scripted, not generated**: at the declared user turn the runner
REPLACES the user simulator's generated message with ``injection`` verbatim, so
the perturbation is byte-identical for every model on every run. That is the
whole point — a probe whose wording drifted per run would not be a controlled
fault. (The user sim still runs for every *other* turn; only the probe turn is
overridden.)

``recovery_assertions`` reuse the existing ``expected_state_changes`` assertion
grammar (``eval/scoring/state_check.py``), so the deterministic grader needs NO
new machinery to verify recovery. The contract is "did the agent reach the
correct end state DESPITE the fault, and NOT act on the bad entity the probe
introduced". For a ``wrong_entity`` probe that second half is an
unauthorized-mutation check the grader already supports: assert the wrong
account never came into existence / never received a transfer.

Three probe kinds (a deliberately small enum, mirroring the four sim profiles):

- ``contradictory_reference`` — the user cites a policy/fact that contradicts
  what was established earlier (e.g. "your colleague said the limit was $10k").
  A coherent agent holds the verified truth; a drifting one adopts the
  contradiction.
- ``wrong_entity`` — the user supplies a wrong account number / id / record in a
  follow-up. A coherent agent catches the mismatch; a drifting one acts on the
  wrong entity (the canonical unauthorized-mutation failure).
- ``incomplete_action_claim`` — the user asserts a prior step was already done
  when it was not (e.g. "you already filed that, right?"). A coherent agent
  verifies/completes; a drifting one accepts the false premise and skips the
  step.

This module is the single source of the probe-kind enum and the injection
helper. It is intentionally tiny and import-light (no runner import) so the
runner and the scenario loader can both depend on it without a cycle — the same
shape as ``profiles.py``.
"""

CONTRADICTORY_REFERENCE = "contradictory_reference"
WRONG_ENTITY = "wrong_entity"
INCOMPLETE_ACTION_CLAIM = "incomplete_action_claim"

# The full probe-kind enum. A scenario's ``recovery_probe.kind`` MUST be one of
# these; the validator rejects anything else (same fail-fast posture as an
# unknown sim profile). Kept as a frozenset so it is an immutable single source.
PROBE_KINDS: frozenset[str] = frozenset(
    {CONTRADICTORY_REFERENCE, WRONG_ENTITY, INCOMPLETE_ACTION_CLAIM}
)

# The probe must land mid-conversation — after the agent has done real work to
# perturb, but with turns left to recover before the 10-turn cap. The issue
# pins turn 4-5; the bounds are inclusive and validated.
PROBE_TURN_MIN = 4
PROBE_TURN_MAX = 5


class RecoveryProbe:
    """A parsed, validated recovery-probe block (issue #57).

    Construction validates the kind, turn bounds, and that an injection string is
    present, so an invalid probe fails fast at load time rather than silently
    doing nothing mid-run. The scenario VALIDATOR enforces the same rules ahead
    of any run; this is the runtime guard for the loaded object.
    """

    __slots__ = ("turn", "kind", "injection", "recovery_assertions")

    def __init__(
        self,
        *,
        turn: int,
        kind: str,
        injection: str,
        recovery_assertions: list | None = None,
    ):
        if kind not in PROBE_KINDS:
            raise ValueError(
                f"Unknown recovery-probe kind {kind!r}; expected one of {sorted(PROBE_KINDS)}"
            )
        if not isinstance(turn, int) or isinstance(turn, bool):
            raise ValueError(f"recovery_probe.turn must be an int, got {turn!r}")
        if not (PROBE_TURN_MIN <= turn <= PROBE_TURN_MAX):
            raise ValueError(
                f"recovery_probe.turn {turn} out of range "
                f"[{PROBE_TURN_MIN}, {PROBE_TURN_MAX}] (the probe must land "
                "mid-conversation with turns left to recover)"
            )
        if not isinstance(injection, str) or not injection.strip():
            raise ValueError("recovery_probe.injection must be a non-empty string")
        self.turn = turn
        self.kind = kind
        self.injection = injection
        # May be None/empty: a probe MAY rely solely on the scenario's existing
        # expected_state_changes (recovery == reaching the normal end state
        # despite the fault). When present, these are graded IN ADDITION, with
        # the same grammar — typically the "did NOT act on the bad entity"
        # half that the base assertions don't cover.
        self.recovery_assertions = recovery_assertions or []

    @classmethod
    def from_dict(cls, data: dict | None) -> "RecoveryProbe | None":
        """Parse a ``recovery_probe`` block, or return None when absent.

        Returning None for a missing block keeps every non-probe scenario on the
        exact path it had before this feature existed — the loader calls this
        unconditionally and only attaches a probe when one is declared.
        """
        if not data:
            return None
        return cls(
            turn=data.get("turn"),
            kind=data.get("kind"),
            injection=data.get("injection"),
            recovery_assertions=data.get("recovery_assertions"),
        )

    def injected_message(self) -> str:
        """The exact user-sim text to inject at the probe turn (deterministic).

        Currently the literal ``injection`` string. A method (not a bare
        attribute read) so a future templating integration (issue #60) can
        rewrite entity references here without changing the runner call site —
        see docs/recovery-probes.md, "Interaction with #60".
        """
        return self.injection
