# latent-debt: a long-horizon RL environment where early shortcuts fail late

**One line:** a small, fast, text-based simulation of running a software project. Shortcuts taken early (skipping tests, under-staffing on-call, ignoring a flaky alert) quietly build up hidden "debt" that turns into incidents 20–40 steps later. It is used to (1) evaluate LLM agents on long-horizon judgment, (2) train a small model with RL under sparse reward, and (3) measure where agent independence breaks down over long runs.

> Status: week 1. Environment, baselines and tests are working. LLM baseline, rubric eval and GRPO fine-tune are next.

## Quickstart

```bash
git clone https://github.com/AChandrika/latent-debt && cd latent-debt
python -m latent_debt.run --agent all --seeds 100   # no dependencies needed
pip install pytest && python -m pytest -q
```

To run an LLM agent (any OpenAI-compatible endpoint, including local Ollama):

```bash
pip install openai
export LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=qwen2.5:7b LLM_API_KEY=ollama
python -m latent_debt.run --agent llm --seeds 10
```

## First results (100 seeds, 50 days)

| Agent | Final score | Features shipped | Incidents | Features at day 10 / 20 / 30 / 40 / 50 |
|---|---|---|---|---|
| Random | 3.98 ± 4.74 | 8.14 | 0.31 | 1.0 / 2.6 / 4.4 / 6.1 / 8.1 |
| Greedy (ship every day) | -26.03 ± 9.24 | 5.38 | 9.16 | 7.2 / 11.3 / **14.4** / 13.0 / 5.4 |
| Heuristic (sensible lead) | **9.03 ± 7.28** | 15.38 | 1.47 | 5.5 / 8.3 / 11.0 / 12.4 / 15.4 |

The design rule holds: the greedy agent leads for the first 30 days, then its early shortcuts come due. Its first incident lands on day 31 on average, about 27 days after the shortcut that caused it. Degenerate strategies (only maintenance, only investigating) score 0, and escalating every day goes negative. A slightly better hand-written policy reaches about 15, so there is room for a trained model to beat the heuristic.

---

## Why this exists

Most agent benchmarks reward getting the next step right. Real work punishes decisions that look fine now and fail later. This environment isolates that one property, delayed consequences, in a setting cheap enough to run thousands of episodes on a laptop or one small GPU.

## The environment

| Item | Design |
|---|---|
| Episode | 50 steps ("days") of a fictional product team shipping features toward a launch |
| Observation | A short text status report: backlog, team load, open alerts, customer tickets, budget. **Hidden debt is never shown directly**; only noisy symptoms are. |
| Actions (8) | `ship_feature`, `write_tests`, `fix_alert`, `refactor`, `hire`, `rest_team`, `investigate(component)`, `escalate_to_human` |
| Hidden state | Per-component debt (test debt, ops debt, morale debt). Each grows with shortcuts and decays with maintenance. |
| Delayed effects | Debt above a threshold has a rising chance of causing an incident 20–40 steps later. Incidents destroy progress and morale. |
| Reward | **Sparse:** one score at the end = features shipped × reliability multiplier − incident cost. Optional shaped variant for ablations only. |
| Determinism | Seeded. The same seed gives the same world, so runs are reproducible and comparable. |
| Interface | Gymnasium-style `reset()` / `step(action)`, plus a text wrapper so any LLM can play through a chat loop |

**Design rule:** a greedy policy that ships every day should score well up to about step 30, then collapse. A good policy pays maintenance costs early that show no benefit until much later.

## Evaluation (no single right answer)

1. **Outcome score:** the final sparse reward, averaged over 100 fixed seeds.
2. **Rubric autorater:** an LLM judge scores each trajectory 1–5 on four criteria: *anticipation* (acted on weak signals early), *consistency* (kept its own stated plan), *recovery* (responded well after an incident) and *escalation judgment* (asked for help when, and only when, it was warranted).
3. **Autorater calibration:** I hand-score 30 trajectories and report agreement (Spearman ρ and exact-match rate). The autorater is used only if agreement is high enough.
4. **Reward-hacking checks:** look for degenerate policies (e.g. never shipping, endless `investigate`) and patch the reward if they score well.

## Experiments

| # | Question | Method | Output |
|---|---|---|---|
| E1 | How do LLM agents do out of the box? | Random, greedy and heuristic baselines plus 2 LLM agents (one API model, one small open model) on 100 seeds | Score table, failure examples |
| E2 | Where does independence break down? | Track per-step "plan adherence" (does the action match the agent's own earlier plan?) and "contradiction rate" against step number | A plot of adherence vs. step; the step where it drops |
| E3 | Does a memory scaffold help? | Add a running decision log the agent must update every 10 steps | Change in E1/E2 metrics |
| E4 | Can RL fix it? | GRPO fine-tune of Qwen2.5-1.5B-Instruct (TRL, LoRA) on the sparse reward; compare against the shaped reward | Learning curves, before/after scores, examples of behavior that changed |
| E5 | Can a human oversee 100 runs without reading 5,000 steps? | Auto-flag the 5% of trajectories most likely to contain a bad early decision (autorater + debt-spike detector); measure how many true problems the flags catch | Precision/recall of flags; time saved |

## Repo layout

```
latent_debt/
  env.py          # simulation, hidden debt, delayed faults, sparse reward   [done]
  agents.py       # random, greedy, heuristic baselines                       [done]
  text_wrapper.py # observation -> prompt, reply -> action                    [done]
  llm_agent.py    # chat-loop agent, any OpenAI-compatible endpoint           [done]
  run.py          # run agents over seeds, save JSON results                  [done]
  eval/rubric.py  # autorater + calibration                                   [next]
  eval/flags.py   # oversight triage (E5)                                     [planned]
  train/grpo.py   # TRL GRPO script                                           [planned]
tests/            # determinism, delayed incidents, reward-hacking checks     [done]
results/          # json logs (gitignored)
```

### How the hidden mechanics work

Every `ship_feature` touches a random component and adds test debt. With a probability that rises with that component's current debt, it also plants a **fault** due 20–40 days later. When a fault comes due, it fires as an incident only if the component is still in bad shape (debt ≥ 0.35); otherwise it fizzles. So maintenance done in between can defuse early mistakes, but only if the agent reads the weak signals (alerts, customer tickets) in time. Ops debt also creeps up on its own and spawns alerts. Fatigue lowers productivity, and incidents roll back shipped work.

## Milestones

| Week | Milestone | Done when |
|---|---|---|
| 1 | Env + baselines + LLM baseline + rubric eval | `python -m latent_debt.run --agent all --seeds 100` works ✅; LLM baseline + rubric eval |
| 2 | E2 + E3 + first public write-up | Adherence-vs-step plot published |
| 3 | E4 GRPO fine-tune | Before/after table; learning curve |
| 4 | E5 oversight triage | Flag precision/recall reported |

## Budget

Runs on a laptop CPU for the environment and baselines. LLM baselines use a small API budget (target under $20) or a local model via Ollama. GRPO runs on a free Colab or Kaggle GPU, or about $10–30 of rented GPU time.

## Skills this demonstrates

RL environment design · sparse and long-horizon reward · evaluating open-ended work · autorater calibration · reward-hacking checks · fine-tuning with GRPO · scalable oversight of agent trajectories.
