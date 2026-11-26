"""Run an agent over many seeds and save results.

    python -m latent_debt.run --agent greedy --seeds 100
    python -m latent_debt.run --agent all --seeds 100
    python -m latent_debt.run --agent llm --seeds 10    # resumable: Ctrl+C, rerun to continue
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import time
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


def run_llm(args, out: Path) -> tuple[str, list[dict]]:
    """LLM runs are slow, so save each episode as it finishes and skip seeds already done.

    Stop any time with Ctrl+C and run the same command again to pick up where you left off.
    """
    from .llm_agent import LLMAgent

    agent = LLMAgent(history_days=args.history, max_tokens=args.max_tokens)
    name = "llm-" + re.sub(r"[^a-zA-Z0-9.]+", "-", agent.model)
    ep_dir = out / name
    ep_dir.mkdir(parents=True, exist_ok=True)
    for seed in range(args.seeds):
        path = ep_dir / f"seed_{seed:03d}.json"
        if path.exists():
            print(f"  seed {seed}: already done, skipping")
            continue
        start = time.time()
        result = run_episode(agent, seed)
        path.write_text(json.dumps(result, indent=1))
        print(f"  seed {seed}: score {result['score']:>6}  incidents {result['incidents']}  "
              f"({(time.time() - start) / 60:.1f} min)", flush=True)
    results = [json.loads(p.read_text()) for p in sorted(ep_dir.glob("seed_*.json"))]
    return name, results


def print_row(name: str, summary: dict) -> None:
    print(f"{name:<18}{summary['score_mean']:>8} ± {summary['score_std']:<6}"
          f"{summary['features_mean']:>9}{summary['incidents_mean']:>11}   "
          + " / ".join(str(v) for v in summary["features_by_day"].values()))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--agent", default="all", help="random | greedy | heuristic | llm | all")
    p.add_argument("--seeds", type=int, default=100)
    p.add_argument("--out", default="results")
    p.add_argument("--history", type=int, default=4, help="LLM: days of conversation the model sees")
    p.add_argument("--max-tokens", type=int, default=120, help="LLM: max length of each reply")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    header = f"{'agent':<18}{'score':>16}{'features':>9}{'incidents':>11}   features at day 10/20/30/40/50"

    if args.agent == "llm":
        name, results = run_llm(args, out)
        runs = {name: results}
    else:
        agents = dict(BASELINES) if args.agent == "all" else {args.agent: BASELINES[args.agent]}
        runs = {n: [run_episode(a, seed) for seed in range(args.seeds)] for n, a in agents.items()}

    print(header)
    for name, results in runs.items():
        summary = summarize(name, results)
        summary["features_by_day"] = features_by_day(results)
        (out / f"{name}.json").write_text(json.dumps({"summary": summary, "episodes": results}, indent=1))
        print_row(name, summary)


if __name__ == "__main__":
    main()
