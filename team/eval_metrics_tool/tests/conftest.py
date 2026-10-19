import sys
from types import SimpleNamespace

import pytest


class FakeLiteLLM:
    """Stands in for the litellm module: records calls, returns a canned reply."""

    def __init__(self):
        self.calls, self.reply, self.error = [], "[]", None

    def completion(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        msg = SimpleNamespace(content=self.reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


@pytest.fixture
def fake_litellm(monkeypatch):
    fake = FakeLiteLLM()
    monkeypatch.setitem(sys.modules, "litellm", fake)
    return fake
