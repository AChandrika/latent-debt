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
