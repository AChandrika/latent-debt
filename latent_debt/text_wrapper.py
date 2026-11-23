"""Turns observations into prompts and model replies into actions, so any LLM can play."""

from __future__ import annotations

import re

from .env import ACTIONS

SYSTEM_PROMPT = f"""You are the engineering lead of a small software team.
Each turn is one working day. You have a fixed number of days before launch.

Your final score = features shipped x reliability, minus a penalty for every production incident.
Shortcuts can cause incidents much later (weeks, not days). You cannot see code health directly:
you only see open alerts and customer tickets, unless you spend a day investigating.

Available actions (choose exactly one per day):
- ship_feature: build and ship a feature (adds points, adds risk)
- write_tests: improve test coverage across the codebase
- fix_alert: fix one open alert
- refactor: clean up the worst part of the codebase
- hire: hire an engineer (costs budget, productive after 5 days)
- rest_team: give the team a lighter day to reduce fatigue
- investigate: spend the day estimating code health (noisy)
- escalate_to_human: ask a senior engineer for an exact code-health readout (costs points)

Before your action, write one short line of reasoning. If you are following a plan, restate it.
End your reply with a line of the form: ACTION: <action_name>
"""


def render(obs: dict) -> str:
    lines = [
        f"Day {obs['day']} ({obs['days_left']} days left)",
        f"Features shipped: {obs['features_shipped']}",
        f"Team: {obs['team_size']} engineers ({obs['hires_ramping']} new hires ramping up), fatigue {obs['fatigue']:.2f}",
        f"Budget: {obs['budget']}",
        "Open alerts: " + ", ".join(f"{c} {n}" for c, n in obs["open_alerts"].items()),
        "Customer tickets today: " + ", ".join(f"{c} {n}" for c, n in obs["customer_tickets"].items()),
        f"Production incidents so far: {obs['incidents_so_far']}",
    ]
    if obs["debt_readout"]:
        lines.append("Code health readout (0 = clean, 2 = very bad): "
                     + ", ".join(f"{c} {v}" for c, v in obs["debt_readout"].items()))
    return "\n".join(lines)


_ACTION_RE = re.compile(r"ACTION:\s*([a-z_]+)", re.IGNORECASE)


def parse_action(reply: str) -> str | None:
    """Return the action named in the reply, or None if it can't be parsed."""
    m = _ACTION_RE.search(reply)
    if m and m.group(1).lower() in ACTIONS:
        return m.group(1).lower()
    # fall back to the last action name mentioned anywhere
    found = [a for a in ACTIONS if a in reply.lower()]
    return found[-1] if found else None
