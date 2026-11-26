1. State, actions and reward in latent-debt

State: everything that's true about the world. In latent-debt that's the day, features shipped, team size, fatigue and budget. It also includes things the agent can't see:

the test debt and ops debt for each of the 3 components
the hidden faults, each with the day it's due to go off
new hires still ramping up

Observation: what the agent actually gets to see. The day and days left, features shipped, team size, fatigue, budget, open alerts, customer tickets (with random noise added), and incident count. It only gets a debt readout on a day it spends investigating or escalating.

So the agent never sees the debt or the faults directly. It has to guess at them from symptoms: lots of tickets for database probably means the database has debt. The RL term for this is a partially observable environment (a POMDP). The course's examples, like a game where you see the whole board, are fully observable. Yours is harder on purpose, because real work is like this: you never see technical debt directly, only its symptoms.

Actions: 8 choices, one per day: ship_feature, write_tests, fix_alert, refactor, hire, rest_team, investigate and escalate_to_human.

Reward: 0 on every day except the last. On day 50 the agent gets one number:

features shipped × reliability − 3 per incident − 0.5 per escalation

2. Why a reward only at the end is harder to learn from

Picture the agent finishing an episode with a score of −10. It made 50 decisions. Which ones caused the bad score?

Maybe a ship_feature on day 5 planted a fault that went off on day 31. The agent didn't get a −3 on day 5. It just gets one bad number on day 50, with no clue that day 5 was the mistake. This is called the credit assignment problem: figuring out which earlier actions deserve the credit or blame for a later result.

That makes learning slow, for two reasons:

The signal is weak. One number has to be shared out across 50 decisions, so the agent needs many episodes before patterns show up.
The signal is noisy. The same choices can score differently depending on the random seed, so a single episode can mislead.

The trap that makes your environment interesting: suppose you made rewards frequent, say +1 every time a feature ships. Learning gets easier, but the agent learns the wrong thing. It would learn to ship every day, which is exactly your greedy agent, which scores −26. Frequent rewards that measure the wrong thing teach bad habits. That's why latent-debt uses a sparse reward, and it's a good point to bring up in interviews.

3. What exploration means in latent-debt

Exploitation is doing what you already think works best. Exploration is trying something else to find out whether it's better.

Early in training, an agent notices that ship_feature is the only action that ever adds points. If it only exploits, it ships every day, never learns why scores collapse, and gets stuck being the greedy agent.

To get better, it has to explore: try refactor or fix_alert on days when shipping looks better, even though they give no immediate points. Only after many episodes like that can it discover that maintenance on day 10 prevents an incident on day 35.

A bonus point: latent-debt has an action that's literally about gathering information: investigate. It costs a day and gives no points, but shows the hidden debt. The agent has to learn when it's worth giving up a day to see more clearly. That's a real research question in itself.

State vs observation: "The agent sees symptoms like alerts and tickets, never the debt itself, so it's partially observable."
Sparse reward: "One score at the end, so the agent has to figure out which of 50 decisions mattered. That's the credit assignment problem."
Exploration: "To learn that maintenance pays off, the agent has to try actions that give nothing now and only pay off weeks later."
**Setup:** random, greedy (always `ship_feature`) and heuristic agents, 100 seeds, 50 days, default config.
**Command:** `python -m latent_debt.run --agent all --seeds 1`
**Result:**
It works. Even with one episode, you can already see the pattern your environment was built to catch:

	Day 10	Day 20	Day 30	Day 40	Day 50	Score
LLM (3b), 1 seed	3.4	7.1	11.1	9.5	3.5	−10.4
Greedy, 100 seeds	7.2	11.3	14.4	13.0	5.4	−26.0
Heuristic, 100 seeds	5.5	8.3	11.0	12.4	15.4	+9.0



## 2026-11-26 — Why the LLM wastes alert fixes (reading its reasoning)

**Question:** why does qwen2.5:3b choose fix_alert when no alerts are open?
**Setup:** read the reasoning in 5 transcript replies on days it chose fix_alert, from `results/llm-qwen2.5-3b.json`.
**Result:** three separate failure modes:
1. **Inverted scale.** It treats the lowest code-health score as the worst ("frontend has the lowest score ... most problematic"), but the readout says 0 = clean, 2 = very bad.
2. **The word "fix" pulls it to the wrong action.** It reasons "focus on the database" and then picks fix_alert, which can't target a component and does nothing with no open alerts. refactor and write_tests are the right tools.
3. **Stale or wrong information.** It cites a code-health readout days after the investigation that produced it, treats customer tickets like alerts, and once called database alerts "critical" (need to check whether any were open).
**What I think it means:** the failure isn't caution. It's misunderstanding what the tools do and losing track of the current state.
**Next:** prompt v2 with clearer action descriptions ("fix_alert does nothing with no open alerts", "higher = worse health"). Change only that, rerun 10 seeds, and compare wasted fixes and score.

## 2026-11-26 — Replay analysis: what the LLM saw when it acted

**Question:** were the LLM's fix_alert days wasted, and did it act on what it investigated?
**Setup:** replayed every saved episode with the recorded actions (the environment is deterministic) to rebuild each observation.
**Command:** `python -m latent_debt.analyze results/llm-qwen2.5-3b.json results/heuristic.json`
**Result:**

| | LLM (qwen2.5:3b, 10 seeds) | Heuristic (100 seeds) |
|---|---|---|
| Score | 3.63 | 9.03 |
| fix_alert with no open alerts | 153 of 232 (66%) | 0 of 774 |
| Investigations followed by a concrete action | 66 of 140 (47%) | never investigates |
| Ships per episode, days 0-9 | 1.5 | 7.0 |
| Ships per episode, days 40-49 | 1.8 | 6.9 |

**What surprised me:** 31% of all the LLM's days went to fixing alerts that weren't there.
**What I think it means:** the heuristic ships early, eases off while faults come due, and pushes at the end because late faults can't go off before day 50. The LLM's shipping is flat, so it never uses the deadline.
**Reproducibility:** my baseline numbers match the original run to the decimal on a different machine (Windows).

## 2026-11-26 — First LLM baseline (10 seeds)

**Question:** how does a small LLM do out of the box?
**Setup:** qwen2.5:3b via Ollama on CPU, 10 seeds, history 4 days, max 120 tokens per reply, temperature 0.2. About 10 minutes per episode.
**Command:** `python -m latent_debt.run --agent llm --seeds 10`
**Result:**

| Agent | Score | Features | Incidents |
|---|---|---|---|
| Greedy | -26.03 ± 9.24 | 5.38 | 9.16 |
| Random | 3.98 ± 4.74 | 8.14 | 0.31 |
| **LLM (qwen2.5:3b)** | **3.63 ± 5.1** | **4.38** | **0.2** |
| Heuristic | 9.03 ± 7.28 | 15.38 | 1.47 |

Action mix: fix_alert 46%, investigate 28%, ship_feature 14%, refactor 7%, other 5%. 3 of 10 episodes shipped nothing (scores 0, 0, -0.5).
**Checked:** 500 of 500 replies had a clear ACTION line, so this isn't a parsing artifact.
**What surprised me:** I expected it to act like greedy. Instead it almost never caused incidents but barely shipped. It scores like random, for a different reason.

## 2026-11-25 — First LLM run (1 seed, old settings)

**Setup:** qwen2.5:3b, seed 0, history 10 days, no reply cap.
**Result:** score -10.4, 4 incidents, features 3.4 / 7.1 / 11.1 / 9.5 / 3.5 at days 10-50. Same rise-and-collapse shape as greedy.
**Note:** with the new settings (history 4, cap 120), seed 0 scored +8.69 with 0 incidents. Same world, very different behavior, so settings matter. One seed proves nothing. Possible experiment: does more history help or hurt?

## 2026-11-25 — RL concepts applied to latent-debt (Hugging Face Deep RL Course, Unit 1)

- **State vs observation:** the true state includes hidden debt per component and the live faults with their due dates. The agent only sees symptoms (alerts, customer tickets, incidents), plus a debt readout the day after it investigates. So the environment is partially observable (a POMDP).
- **Actions:** 8 discrete choices, one per day. **Reward:** 0 every day, then one score at day 50.
- **Sparse reward and credit assignment:** with one score at the end, the agent has to work out which of its 50 decisions mattered. A shortcut on day 5 can cause an incident on day 31, and the reward never says so. A frequent reward like +1 per feature would be easier to learn from but would teach greedy behavior, which is why I kept it sparse.
- **Exploration:** to learn that maintenance pays off, the agent has to try actions that give nothing now (refactor, write_tests) and only pay off weeks later. `investigate` is a direct information-gathering action: it gives up a day to see the hidden state.

# Research notes

A running log of every experiment: what I tried, what happened, and what I think it means.
Newest entries at the top. Failed experiments go in too; they are often the most useful.

---

## Entry template

```
## YYYY-MM-DD — short title

**Question:** what I wanted to find out
**Setup:** what I changed (config, agent, seeds, commit hash)
**Command:** exact command to reproduce
**Result:** numbers, with seeds and spread (mean ± std)
**What surprised me:**
**What I think it means:**
**Next:**
```

---

## Open questions

- At what step do LLM agents stop following their own earlier plan? Does it depend on model size?
- Does a written decision log every 10 steps help, or just add noise?
- Will GRPO on the sparse reward learn anything in 50-step episodes, or is shaped reward needed first?
- Can an autorater spot the bad early decision in a trajectory without seeing the hidden state?
- Is the heuristic agent's score (about 9) a good bar, or is the environment too easy to beat?

## Ideas parked for later

- Smaller incidents instead of all-or-nothing faults (partial defusing).
- Let `investigate` target one component instead of all three.
- Make hiring pay off in the long run, so it becomes a real long-horizon trade-off.

---

## 2026-11-23 — Reward-hacking checks

**Question:** can an agent score well without doing the real work?
**Setup:** ran each single-action policy (always rest, always investigate, always write tests, and so on) for 50 seeds.
**Command:** see `tests/test_env.py::test_doing_nothing_useful_scores_zero`
**Result:** every maintenance-only policy scored 0. Always escalating scored negative (each escalation costs 0.5).
**What I think it means:** stalling isn't a loophole. Points only come from shipping, and shipping carries risk, which is the trade-off I want.
**Next:** check this again whenever the reward changes.

## 2026-11-23 — Baselines and the design rule

**Question:** does an always-ship agent win early and lose late?
**Setup:** random, greedy (always `ship_feature`) and heuristic agents, 100 seeds, 50 days, default config.
**Command:** `python -m latent_debt.run --agent all --seeds 100`
**Result:**

| Agent | Score | Features | Incidents | Features at day 10 / 20 / 30 / 40 / 50 |
|---|---|---|---|---|
| Random | 3.98 ± 4.74 | 8.14 | 0.31 | 1.0 / 2.6 / 4.4 / 6.1 / 8.1 |
| Greedy | -26.03 ± 9.24 | 5.38 | 9.16 | 7.2 / 11.3 / 14.4 / 13.0 / 5.4 |
| Heuristic | 9.03 ± 7.28 | 15.38 | 1.47 | 5.5 / 8.3 / 11.0 / 12.4 / 15.4 |

Greedy's first incident lands on day 31 on average. Average gap between a shortcut and its incident is about 27 days.

**What surprised me:** how hard greedy collapses: it loses more than half its features between day 30 and day 50. Part of that is fatigue (it never rests), not just delayed faults. I should separate the two effects.
**What I think it means:** the core design rule holds. A slightly better hand-written policy (fix alerts as soon as they appear, rest when fatigue is over 0.6) scored about 15, so there's room above the heuristic for a trained model.
**Next:** run with fatigue switched off to measure how much of the collapse comes from delayed faults alone. Then run the first LLM baseline.

## 2026-11-23 — Environment built

**Setup:** text-based software-team simulation. 8 actions, 3 components, hidden test and ops debt per component. Shipping plants faults with a probability that rises with current debt. A fault comes due 20–40 days later and fires only if the component's debt is still at least 0.35; otherwise it fizzles. Sparse reward at the end: features × reliability, minus 3 per incident, minus 0.5 per escalation.
**Why faults fizzle:** so maintenance done in between can save an early mistake, but only if the agent notices the weak signals (alerts, tickets) in time. Without that, the environment would just punish shipping, and there'd be nothing to learn.
**Tests:** 8 passing (determinism, delayed incidents, greedy collapse, reward hacking, text wrapper).
