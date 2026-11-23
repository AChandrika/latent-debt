import statistics

import pytest

from latent_debt.agents import greedy_agent, heuristic_agent
from latent_debt.env import ACTIONS, LatentDebtEnv
from latent_debt.run import run_episode
from latent_debt.text_wrapper import parse_action, render


def test_episode_length_and_sparse_reward():
    env = LatentDebtEnv()
    env.reset(seed=1)
    rewards = []
    for _ in range(env.cfg.horizon):
        _, r, done, _ = env.step("ship_feature")
        rewards.append(r)
    assert done
    assert all(r == 0.0 for r in rewards[:-1])   # nothing until the last step


def test_same_seed_same_world():
    a = run_episode(greedy_agent, seed=7)
    b = run_episode(greedy_agent, seed=7)
    assert a["score"] == b["score"] and a["incident_log"] == b["incident_log"]


def test_unknown_action_rejected():
    env = LatentDebtEnv()
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step("deploy_on_friday")


def test_hidden_state_not_in_observation():
    env = LatentDebtEnv()
    obs = env.reset(seed=0)
    assert "test_debt" not in obs and "live_faults" not in obs
    assert obs["debt_readout"] is None


def test_incidents_arrive_late():
    """Core design rule: consequences show up weeks after the shortcut."""
    delays = []
    for seed in range(30):
        r = run_episode(greedy_agent, seed)
        delays += [day - planted for day, _c, planted in r["incident_log"]]
    assert delays and min(delays) >= 20


def test_greedy_leads_early_then_collapses():
    greedy = [run_episode(greedy_agent, s) for s in range(50)]
    heur = [run_episode(heuristic_agent, s) for s in range(50)]
    at = lambda rs, d: statistics.mean(r["steps"][d - 1]["features"] for r in rs)
    assert at(greedy, 20) > at(heur, 20)                        # shortcuts pay off early
    assert statistics.mean(r["score"] for r in greedy) < statistics.mean(r["score"] for r in heur)


def test_doing_nothing_useful_scores_zero():
    """Reward-hacking check: pure maintenance or stalling should not score well."""
    for action in ["investigate", "rest_team", "write_tests"]:
        r = run_episode(lambda o, rng, a=action: a, seed=0)
        assert r["score"] <= 0


def test_text_wrapper_round_trip():
    env = LatentDebtEnv()
    prompt = render(env.reset(seed=0))
    assert "Day 0" in prompt
    assert parse_action("Alerts are piling up.\nACTION: fix_alert") == "fix_alert"
    assert parse_action("no idea") is None
    assert all(parse_action(f"ACTION: {a}") == a for a in ACTIONS)
