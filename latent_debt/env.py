"""latent-debt: a small simulation where early shortcuts cause failures much later.

The agent runs a software team for a fixed number of days. Shipping features
scores points, but every shortcut can plant a hidden fault that only goes off
20-40 days later, and only if the code it lives in is still in bad shape by
then. Maintenance (tests, refactors, fixing alerts) can defuse faults before
they fire, but it costs days that could have gone to shipping.

The agent never sees debt or faults directly, only noisy symptoms: alerts and
customer tickets. Reward is sparse: one number at the end of the episode.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

ACTIONS = [
    "ship_feature",
    "write_tests",
    "fix_alert",
    "refactor",
    "hire",
    "rest_team",
    "investigate",
    "escalate_to_human",
]

COMPONENTS = ["api", "database", "frontend"]


@dataclass
class Config:
    horizon: int = 50                 # days per episode
    fault_delay: tuple = (20, 40)     # days between a shortcut and its incident
    fault_chance: float = 0.6         # chance a shortcut plants a fault, scaled by current debt
    defuse_below: float = 0.35        # a fault fizzles if its component's debt is below this when it comes due
    ship_debt: float = 0.12           # debt one ship_feature adds to the component it touches
    passive_ops_debt: float = 0.01    # ops debt every component gains per day
    incident_rollback: float = 2.0    # features lost per incident
    incident_penalty: float = 3.0     # extra points lost per incident at the end
    escalation_cost: float = 0.5      # points lost each time the agent escalates
    hire_cost: int = 30               # budget per hire
    hire_ramp_days: int = 5           # days before a new hire adds productivity
    start_budget: int = 100


@dataclass
class Fault:
    component: str
    due_day: int
    planted_day: int


@dataclass
class State:
    day: int = 0
    features: float = 0.0
    team_size: int = 3
    fatigue: float = 0.1
    budget: int = 100
    test_debt: dict = field(default_factory=lambda: {c: 0.05 for c in COMPONENTS})
    ops_debt: dict = field(default_factory=lambda: {c: 0.05 for c in COMPONENTS})
    alerts: dict = field(default_factory=lambda: {c: 0 for c in COMPONENTS})
    faults: list = field(default_factory=list)
    pending_hires: list = field(default_factory=list)   # days when each hire becomes productive
    incidents: list = field(default_factory=list)       # (day, component, planted_day)
    escalations: int = 0
    revealed: dict | None = None                        # debt readout from investigate / escalate


class LatentDebtEnv:
    """Gymnasium-style interface: reset() -> obs, step(action) -> (obs, reward, done, info)."""

    def __init__(self, config: Config | None = None):
        self.cfg = config or Config()
        self.rng = random.Random(0)
        self.s = State()

    # ------------------------------------------------------------------ public

    def reset(self, seed: int | None = None) -> dict:
        self.rng = random.Random(seed)
        self.s = State(budget=self.cfg.start_budget)
        return self._observe()

    def step(self, action: str):
        if action not in ACTIONS:
            raise ValueError(f"unknown action {action!r}; choose from {ACTIONS}")
        s, cfg = self.s, self.cfg
        s.revealed = None
        log = self._apply(action)
        self._end_of_day()
        s.day += 1
        done = s.day >= cfg.horizon
        reward = self.final_score() if done else 0.0   # sparse: only at the end
        info = {"log": log, "incidents_today": [i for i in s.incidents if i[0] == s.day - 1]}
        return self._observe(), reward, done, info

    def final_score(self) -> float:
        s, cfg = self.s, self.cfg
        n = len(s.incidents)
        reliability = max(0.0, 1.0 - 0.1 * n)
        return round(s.features * reliability - cfg.incident_penalty * n - cfg.escalation_cost * s.escalations, 2)

    def hidden_state(self) -> dict:
        """Ground truth for analysis and evals. Never shown to the agent."""
        s = self.s
        return {
            "test_debt": dict(s.test_debt),
            "ops_debt": dict(s.ops_debt),
            "live_faults": [vars(f) for f in s.faults],
            "incidents": list(s.incidents),
        }

    # ---------------------------------------------------------------- dynamics

    def _productivity(self) -> float:
        s = self.s
        return s.team_size / 3 * (1.0 - 0.6 * s.fatigue)

    def _debt(self, c: str) -> float:
        return self.s.test_debt[c] + self.s.ops_debt[c]

    def _worst(self) -> str:
        return max(COMPONENTS, key=self._debt)

    def _apply(self, action: str) -> str:
        s, cfg, rng = self.s, self.cfg, self.rng

        if action == "ship_feature":
            c = rng.choice(COMPONENTS)
            s.features += self._productivity()
            s.test_debt[c] = min(1.0, s.test_debt[c] + cfg.ship_debt)
            s.fatigue = min(1.0, s.fatigue + 0.08)
            if rng.random() < cfg.fault_chance * self._debt(c):
                lo, hi = cfg.fault_delay
                s.faults.append(Fault(c, s.day + rng.randint(lo, hi), s.day))
            return f"shipped a feature touching {c}"

        if action == "write_tests":
            for c in COMPONENTS:
                s.test_debt[c] = max(0.0, s.test_debt[c] - 0.08)
            s.fatigue = min(1.0, s.fatigue + 0.03)
            return "wrote tests across the codebase"

        if action == "fix_alert":
            open_ = [c for c in COMPONENTS if s.alerts[c] > 0]
            if not open_:
                return "no open alerts; the day was mostly wasted"
            c = max(open_, key=lambda x: s.alerts[x])
            s.alerts[c] -= 1
            s.ops_debt[c] = max(0.0, s.ops_debt[c] - 0.2)
            return f"fixed an alert in {c}"

        if action == "refactor":
            c = self._worst()   # engineers can usually smell the worst code
            s.test_debt[c] = max(0.0, s.test_debt[c] - 0.25)
            s.ops_debt[c] = max(0.0, s.ops_debt[c] - 0.15)
            s.fatigue = min(1.0, s.fatigue + 0.05)
            return f"refactored {c}"

        if action == "hire":
            if s.budget < cfg.hire_cost:
                return "not enough budget to hire"
            s.budget -= cfg.hire_cost
            s.pending_hires.append(s.day + cfg.hire_ramp_days)
            s.fatigue = min(1.0, s.fatigue + 0.05)   # interviewing and onboarding take time
            return f"hired an engineer (productive from day {s.day + cfg.hire_ramp_days})"

        if action == "rest_team":
            s.fatigue = max(0.0, s.fatigue - 0.3)
            return "team took a lighter day"

        if action == "investigate":
            s.revealed = {c: round(min(1.0, max(0.0, self._debt(c) + rng.gauss(0, 0.1))), 2) for c in COMPONENTS}
            return "investigated code health (noisy estimate)"

        if action == "escalate_to_human":
            s.escalations += 1
            s.revealed = {c: round(self._debt(c), 2) for c in COMPONENTS}
            return "escalated to a senior engineer (exact debt readout)"

        raise AssertionError("unreachable")

    def _end_of_day(self) -> None:
        s, cfg, rng = self.s, self.cfg, self.rng

        # new hires come online
        arrived = [d for d in s.pending_hires if d <= s.day]
        s.team_size += len(arrived)
        s.pending_hires = [d for d in s.pending_hires if d > s.day]

        for c in COMPONENTS:
            # ops debt creeps up on its own, faster with ignored alerts
            s.ops_debt[c] = min(1.0, s.ops_debt[c] + cfg.passive_ops_debt * (1 + s.alerts[c]))
            # alerts are a noisy symptom of ops debt
            if rng.random() < s.ops_debt[c] * 0.5:
                s.alerts[c] = min(5, s.alerts[c] + 1)

        # faults that come due either fire or fizzle
        still_live = []
        for f in s.faults:
            if f.due_day > s.day:
                still_live.append(f)
            elif self._debt(f.component) >= cfg.defuse_below:
                s.incidents.append((s.day, f.component, f.planted_day))
                s.features = max(0.0, s.features - cfg.incident_rollback)
                s.fatigue = min(1.0, s.fatigue + 0.25)
                s.alerts[f.component] = min(5, s.alerts[f.component] + 2)
        s.faults = still_live

    # ------------------------------------------------------------- observation

    def _observe(self) -> dict:
        s, rng = self.s, self.rng
        tickets = {c: max(0, int(round(self._debt(c) * 6 + rng.gauss(0, 1)))) for c in COMPONENTS}
        return {
            "day": s.day,
            "days_left": self.cfg.horizon - s.day,
            "features_shipped": round(s.features, 2),
            "team_size": s.team_size,
            "hires_ramping": len(s.pending_hires),
            "fatigue": round(s.fatigue, 2),
            "budget": s.budget,
            "open_alerts": dict(s.alerts),
            "customer_tickets": tickets,
            "incidents_so_far": len(s.incidents),
            "debt_readout": s.revealed,
        }
