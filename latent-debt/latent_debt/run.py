"""Run an agent over many seeds and save results.

    python -m latent_debt.run --agent greedy --seeds 100
    python -m latent_debt.run --agent all --seeds 100
    python -m latent_debt.run --agent llm --seeds 10
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from .agents import BASELINES
from .env import Config, LatentDebtEnv


def run_episode(agent, seed: int, config: Config | None = None) -> dict:
    env = LatentDebtEnv(config)
    obs = env.reset(seed)
    rng = random.Random(seed + 10_000)   # agent's own randomness, separate from the world's
    if hasattr(agent, "reset"):
        agent.reset()
    steps, done, reward = [], False, 0.0
    while not done:
        action = agent(obs, rng)
        obs, reward, done, info = env.step(action)
        steps.append({"day": obs["day"] - 1, "action": action,
                      "features": obs["features_shipped"],
                      "incidents": obs["incidents_so_far"]})
    result = {"seed": seed, "score": reward, "features": obs["features_shipped"],
              "incidents": obs["incidents_so_far"], "steps": steps,
              "incident_log": env.hidden_state()["incidents"]}
    if hasattr(agent, "transcript"):
        result["transcript"] = agent.transcript
    return result


def summarize(name: str, results: list[dict]) -> dict:
    scores = [r["score"] for r in results]
    incidents = [r["incidents"] for r in results]
    delays = [day - planted for r in results for (day, _c, planted) in r["incident_log"]]
    return {
        "agent": name,
        "episodes": len(results),
        "score_mean": round(statistics.mean(scores), 2),
        "score_std": round(statistics.pstdev(scores), 2),
        "features_mean": round(statistics.mean(r["features"] for r in results), 2),
        "incidents_mean": round(statistics.mean(incidents), 2),
        "mean_days_from_shortcut_to_incident": round(statistics.mean(delays), 1) if delays else None,
    }


def features_by_day(results: list[dict], days: tuple = (10, 20, 30, 40, 50)) -> dict:
    out = {}
    for d in days:
        vals = [r["steps"][d - 1]["features"] for r in results if len(r["steps"]) >= d]
        out[d] = round(statistics.mean(vals), 2)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--agent", default="all", help="random | greedy | heuristic | llm | all")
    p.add_argument("--seeds", type=int, default=100)
    p.add_argument("--out", default="results")
    args = p.parse_args()

    if args.agent == "all":
        agents = dict(BASELINES)
    elif args.agent == "llm":
        from .llm_agent import LLMAgent
        agents = {"llm": LLMAgent()}
    else:
        agents = {args.agent: BASELINES[args.agent]}

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    print(f"{'agent':<10}{'score':>14}{'features':>10}{'incidents':>11}   features at day 10/20/30/40/50")
    for name, agent in agents.items():
        results = [run_episode(agent, seed) for seed in range(args.seeds)]
        summary = summarize(name, results)
        curve = features_by_day(results)
        summary["features_by_day"] = curve
        (out / f"{name}.json").write_text(json.dumps({"summary": summary, "episodes": results}, indent=1))
        print(f"{name:<10}{summary['score_mean']:>8} ± {summary['score_std']:<4}"
              f"{summary['features_mean']:>10}{summary['incidents_mean']:>11}   "
              + " / ".join(str(v) for v in curve.values()))


if __name__ == "__main__":
    main()
