"""Replay saved episodes to see what the agent was looking at when it chose each action.

The world is deterministic given the seed and the actions, so replaying the recorded
actions rebuilds every observation exactly.

    python -m latent_debt.analyze results/llm-qwen2.5-3b.json
"""

from __future__ import annotations

import collections
import json
import sys

from .env import LatentDebtEnv


def analyze(path: str) -> dict:
    data = json.load(open(path))
    wasted_fix = fix_total = 0
    investigate_total = acted_on_readout = 0
    ship_days = collections.Counter()

    for ep in data["episodes"]:
        env = LatentDebtEnv()
        obs = env.reset(ep["seed"])
        actions = [s["action"] for s in ep["steps"]]
        for i, action in enumerate(actions):
            if action == "fix_alert":
                fix_total += 1
                if sum(obs["open_alerts"].values()) == 0:
                    wasted_fix += 1            # nothing to fix: the day was lost
            if action == "investigate":
                investigate_total += 1
                nxt = actions[i + 1] if i + 1 < len(actions) else None
                if nxt in ("refactor", "write_tests", "fix_alert", "ship_feature"):
                    acted_on_readout += 1      # did something with what it learned
            if action == "ship_feature":
                ship_days[obs["day"] // 10 * 10] += 1
            obs, _, _, _ = env.step(action)

    n = len(data["episodes"])
    return {
        "episodes": n,
        "fix_alert_with_no_open_alerts": f"{wasted_fix} of {fix_total}",
        "investigate_followed_by_a_concrete_action": f"{acted_on_readout} of {investigate_total}",
        "ships_per_episode_by_day_range": {f"days {d}-{d + 9}": round(c / n, 1)
                                           for d, c in sorted(ship_days.items())},
    }


if __name__ == "__main__":
    for path in sys.argv[1:]:
        print(path)
        for k, v in analyze(path).items():
            print(f"  {k}: {v}")
