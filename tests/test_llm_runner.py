"""Checks the LLM runner end to end with a fake model, so no server or API key is needed."""

import sys
import types
from argparse import Namespace


class _FakeCompletions:
    def create(self, model, messages, temperature, max_tokens):
        assert max_tokens == 50
        day = messages[-1]["content"].split()[1]
        action = "refactor" if int(day) % 5 == 0 else "ship_feature"
        msg = types.SimpleNamespace(content=f"Plan: ship, refactor every 5 days.\nACTION: {action}")
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


class _FakeOpenAI:
    def __init__(self, base_url=None, api_key=None):
        self.chat = types.SimpleNamespace(completions=_FakeCompletions())


def test_llm_run_saves_each_episode_and_resumes(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_FakeOpenAI))
    monkeypatch.setenv("LLM_MODEL", "fake:1b")
    from latent_debt.run import run_llm

    args = Namespace(seeds=2, history=4, max_tokens=50)
    name, results = run_llm(args, tmp_path)
    assert name == "llm-fake-1b"
    assert len(results) == 2 and len(list((tmp_path / name).glob("seed_*.json"))) == 2
    assert all(t["parsed"] for t in results[0]["transcript"])

    # running again with more seeds skips the finished ones
    args.seeds = 3
    _, results = run_llm(args, tmp_path)
    assert len(results) == 3
    assert capsys.readouterr().out.count("already done") == 2
