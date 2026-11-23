"""Baseline agents. Each agent is a callable: agent(obs, rng) -> action name."""

from __future__ import annotations

import random

from .env import ACTIONS


def random_agent(obs: dict, rng: random.Random) -> str:
    return rng.choice(ACTIONS)


def greedy_agent(obs: dict, rng: random.Random) -> str:
    """Ships every single day. Looks great until the faults come due."""
    return "ship_feature"


def heuristic_agent(obs: dict, rng: random.Random) -> str:
    """A reasonable engineering lead: ships most days, keeps maintenance going.

    Not optimal, just a sensible human-written reference point.
    """
    alerts = sum(obs["open_alerts"].values())
    tickets = sum(obs["customer_tickets"].values())
    day, days_left = obs["day"], obs["days_left"]

    if obs["fatigue"] > 0.7:
        return "rest_team"
    if days_left <= 8:
        return "ship_feature"          # faults planted now land after the deadline
    if alerts >= 2:
        return "fix_alert"
    if tickets >= 9:
        return "refactor"
    if day % 4 == 3:
        return "write_tests"
    return "ship_feature"


BASELINES = {
    "random": random_agent,
    "greedy": greedy_agent,
    "heuristic": heuristic_agent,
}
