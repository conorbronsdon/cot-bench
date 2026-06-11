"""Dual-control: user-side world actions (issue #58).

From the best-in-class agent-eval research (Delta 5; source: **tau2-bench**):
the dominant real-world complexity that single-control simulation misses is
**dual control** — both the agent AND the user invoke tools against the *same*
shared world. A customer who self-serves mid-conversation (updates their own
contact info, approves or denies a request the agent sent, reads their own
notifications) creates a coordination problem the agent must navigate: it must
wait for an approval it does not control, and it must NOT re-apply a change the
user already made. tau2-bench reports that every model tested degrades
significantly under dual control, and banking / customer-success are exactly the
domains where it matters. This module is the mechanism for that test.

The tau2 provenance, the scripted-vs-prompted design decision, the attribution
semantics, and the v1.1 framing are argued in full in ``docs/dual-control.md``.
The short version of the one non-obvious choice:

**Scripted, deterministic user actions — not a prompted user-sim tool loop.**
tau2 gives the user simulator its OWN tool set and lets the LLM decide when to
call them. That is higher fidelity but nondeterministic: the user's mutations
would differ run-to-run, the shared world would stop being reproducible, and —
fatally for this bench — the deterministic grader could no longer attribute a
mutation to the agent vs the user. COT Bench's whole identity is deterministic
state grading (``eval/scoring/state_check.py``), tamper-evident pre-registration,
and the byte-identical scripted perturbations of recovery probes (issue #57). So
dual control here is a HYBRID: a scenario DECLARES ``user_tools`` (the scope of
what the user side may touch) and ``user_actions`` (trigger condition -> a
specific user tool call with a specific state delta), and the runner applies
those actions DETERMINISTICALLY through the same tool-sim / state-delta machinery
the agent's tools use. One world, one delta mechanism, fully attributable. The
fidelity cost (the user's behavior is scripted, not free) is the deliberate
trade; see the decision doc.

This module is the single source of the user-tool / user-action parsing, the
authorization-scope check, and the trigger evaluation. It is intentionally tiny
and import-light (no runner import) so the runner and the scenario loader can
both depend on it without a cycle — the same shape as ``profiles.py`` and
``probes.py``.
"""

from collections.abc import Iterable

# --- Trigger kinds -------------------------------------------------------- #
# A user action fires when its trigger condition is met. The vocabulary is
# deliberately tiny and fully deterministic (the same posture as the
# state-check assertion grammar): no LLM is in the firing decision.
#
# - ``after_turn``  : fire once, when the conversation reaches a given user turn.
#                     The user-side analogue of a recovery probe's turn — the
#                     user "self-serves" at a fixed point. Deterministic by turn
#                     index; ``value`` is the (int) turn.
# - ``agent_called``: fire once, the first time the AGENT calls a named tool
#                     (e.g. the agent sends an approval request; the user
#                     approves it the next turn). This is the coordination /
#                     handoff case — the user reacts to an agent action.
#                     ``value`` is the (str) agent tool name to watch for.
TRIGGER_AFTER_TURN = "after_turn"
TRIGGER_AGENT_CALLED = "agent_called"

TRIGGER_KINDS: frozenset[str] = frozenset({TRIGGER_AFTER_TURN, TRIGGER_AGENT_CALLED})

# A user action's firing turn must land mid-conversation (after the agent has
# done real work, with turns left to coordinate), mirroring the recovery-probe
# bounds. ``after_turn`` triggers are validated to this inclusive range; an
# ``agent_called`` trigger fires whenever the watched call happens (no bound).
USER_ACTION_TURN_MIN = 1
USER_ACTION_TURN_MAX = 9


class UserTool:
    """A tool the USER side may invoke against the shared world (issue #58).

    Same shape as an agent tool (name / description / parameters list) so the
    tool simulator can answer a user tool call with the EXACT machinery it uses
    for agent calls — one tool-sim, one world, one delta format. The added,
    load-bearing field is ``scope``: the set of dotted top-level state keys this
    user tool is authorized to mutate. The authorization boundary (the subtle
    part of issue #58) is enforced against this declared scope, NOT inferred from
    what the action happens to touch — so a user action that strays outside its
    declared scope is a *validation* failure, caught before any run.
    """

    __slots__ = ("name", "description", "parameters", "scope")

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        parameters: list | None = None,
        scope: list | None = None,
    ):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("user_tool.name must be a non-empty string")
        self.name = name
        self.description = description
        self.parameters = parameters or []
        # The declared authorization scope: dotted top-level state keys this user
        # tool may write. Empty scope means read-only (the user tool can be
        # called but must produce no state delta) — a legitimate case (e.g. the
        # user READS their own notifications).
        self.scope = list(scope or [])

    @classmethod
    def from_dict(cls, data: dict) -> "UserTool":
        return cls(
            name=data.get("name"),
            description=data.get("description", ""),
            parameters=data.get("parameters"),
            scope=data.get("scope"),
        )

    def as_tool_schema(self) -> dict:
        """The agent-tool-shaped dict the tool simulator consumes.

        ``scope`` is intentionally dropped here: the tool simulator answers from
        the same schema an agent tool would present (name/description/parameters).
        Scope is an authorization fact enforced by the validator and the runner's
        attribution, not something the tool sim needs to see.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class UserAction:
    """A scripted, deterministic user-side world action (issue #58).

    An action binds a TRIGGER (when the user self-serves) to a specific
    ``user_tool`` call with specific ``arguments`` and a specific ``state_delta``
    (what the user's action does to the shared world). The delta is applied
    through the EXACT ``apply_state_delta`` the agent's tools use, so the grader
    sees one coherent world — but every path the action writes is recorded as a
    USER-side mutation so the no-unauthorized-mutation contract attributes it
    correctly (it must never be charged to the agent).

    Determinism is the whole point: the user's behavior is fixed per scenario, so
    the dual-control world is reproducible and every mutation is attributable.
    The fidelity trade (a real user-sim would decide its own tool calls) is
    argued in docs/dual-control.md.
    """

    __slots__ = ("tool", "trigger", "trigger_value", "arguments", "state_delta", "user_message")

    def __init__(
        self,
        *,
        tool: str,
        trigger: str,
        trigger_value,
        arguments: dict | None = None,
        state_delta: dict | None = None,
        user_message: str | None = None,
    ):
        if not isinstance(tool, str) or not tool.strip():
            raise ValueError("user_action.tool must be a non-empty string")
        if trigger not in TRIGGER_KINDS:
            raise ValueError(
                f"Unknown user_action trigger {trigger!r}; expected one of {sorted(TRIGGER_KINDS)}"
            )
        if trigger == TRIGGER_AFTER_TURN:
            if not isinstance(trigger_value, int) or isinstance(trigger_value, bool):
                raise ValueError("after_turn trigger value must be an int turn")
            if not (USER_ACTION_TURN_MIN <= trigger_value <= USER_ACTION_TURN_MAX):
                raise ValueError(
                    f"after_turn trigger turn {trigger_value} out of range "
                    f"[{USER_ACTION_TURN_MIN}, {USER_ACTION_TURN_MAX}]"
                )
        elif trigger == TRIGGER_AGENT_CALLED:
            if not isinstance(trigger_value, str) or not trigger_value.strip():
                raise ValueError("agent_called trigger value must be a non-empty tool name")
        self.tool = tool
        self.trigger = trigger
        self.trigger_value = trigger_value
        self.arguments = arguments or {}
        # The state mutation the user's action performs. Same dotted-path format
        # as a tool-sim ``state_delta`` (see runner.apply_state_delta). May be
        # empty for a read-only user action (e.g. the user reads notifications) —
        # an empty delta is the legitimate "touched nothing" case.
        self.state_delta = state_delta or {}
        # Optional natural-language message the user "says" when they self-serve,
        # injected as the user turn so the agent SEES the coordination signal
        # (e.g. "I just approved that on my end."). When None, the user action is
        # silent (state-only) and the user sim still drives that turn's words.
        self.user_message = user_message

    @classmethod
    def from_dict(cls, data: dict) -> "UserAction":
        return cls(
            tool=data.get("tool"),
            trigger=data.get("trigger"),
            trigger_value=data.get("trigger_value"),
            arguments=data.get("arguments"),
            state_delta=data.get("state_delta"),
            user_message=data.get("user_message"),
        )

    def delta_paths(self) -> list[str]:
        """Top-level state keys this action's delta writes (for attribution).

        Returns the first dotted segment of every key in ``state_delta`` — the
        top-level keys the user-side mutation touches. Used by the runner to mark
        those keys USER-attributed and by the validator to enforce that they fall
        within the user tool's declared ``scope``.
        """
        return [str(k).split(".", 1)[0] for k in self.state_delta]


class DualControl:
    """Parsed dual-control config for a scenario (issue #58).

    Bundles the declared ``user_tools`` (keyed by name) and the ordered
    ``user_actions``. Construction validates each tool and action, and — the
    authorization boundary — that every action names a declared tool and writes
    ONLY within that tool's declared scope. An invalid config fails fast at load
    time, before any run, exactly like RecoveryProbe / an unknown sim profile.
    """

    __slots__ = ("user_tools", "user_actions")

    def __init__(self, *, user_tools: list[UserTool], user_actions: list[UserAction]):
        self.user_tools = {t.name: t for t in user_tools}
        self.user_actions = list(user_actions)
        # Authorization boundary: every action targets a declared tool and stays
        # within that tool's scope. The runtime guard mirrors the validator so a
        # loaded object is self-consistent even if it bypassed the on-disk check.
        for action in self.user_actions:
            tool = self.user_tools.get(action.tool)
            if tool is None:
                raise ValueError(f"user_action references undeclared user_tool {action.tool!r}")
            out_of_scope = [p for p in action.delta_paths() if p not in tool.scope]
            if out_of_scope:
                raise ValueError(
                    f"user_action on {action.tool!r} writes outside its declared scope "
                    f"{sorted(tool.scope)}: {sorted(set(out_of_scope))}"
                )

    @classmethod
    def from_dict(cls, data: dict | None) -> "DualControl | None":
        """Parse a ``dual_control`` block, or return None when absent.

        Returning None for a missing block keeps every single-control scenario on
        the exact path it had before this feature existed — the loader calls this
        unconditionally and only attaches a DualControl when one is declared.
        """
        if not data:
            return None
        tools = [UserTool.from_dict(t) for t in data.get("user_tools", [])]
        if not tools:
            raise ValueError("dual_control requires at least one user_tool")
        actions = [UserAction.from_dict(a) for a in data.get("user_actions", [])]
        if not actions:
            raise ValueError("dual_control requires at least one user_action")
        return cls(user_tools=tools, user_actions=actions)

    def tool_schemas(self) -> list[dict]:
        """Agent-tool-shaped schemas for the user tools (for the tool sim)."""
        return [t.as_tool_schema() for t in self.user_tools.values()]

    def pending_actions(self) -> list["UserAction"]:
        """The actions in declared order (the runner fires each at most once)."""
        return list(self.user_actions)


def action_fires(
    action: UserAction,
    *,
    next_user_turn: int,
    agent_tool_calls_so_far: Iterable[str],
) -> bool:
    """Deterministically decide whether ``action`` fires at this point.

    Pure and side-effect-free so it is trivially testable. The runner calls it at
    the end of each outer turn (when it is about to stage the next user turn),
    with the index of the upcoming user turn and the set of agent tool names
    called so far in the conversation. Firing is one-shot per action — the runner
    tracks which actions have fired and never re-evaluates a fired one.

    - ``after_turn`` fires when ``next_user_turn`` reaches the trigger turn.
    - ``agent_called`` fires once the named agent tool appears in the calls so
      far (the user reacts to the agent's action on the next turn).
    """
    if action.trigger == TRIGGER_AFTER_TURN:
        return next_user_turn >= action.trigger_value
    if action.trigger == TRIGGER_AGENT_CALLED:
        return action.trigger_value in set(agent_tool_calls_so_far)
    return False
