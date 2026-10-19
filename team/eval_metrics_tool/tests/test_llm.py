import pytest

from eval_metrics.llm import LLMError, _parse, ask_llm


def test_parse_plain_array():
    assert _parse('["F1", "Accuracy"]') == ["F1", "Accuracy"]


def test_parse_strips_think_and_prose():
    reply = '<think>maybe ["wrong"]</think>Sure! Here you go:\n```json\n["BLEU", "chrF"]\n```'
    assert _parse(reply) == ["BLEU", "chrF"]


def test_parse_dedupes_and_drops_non_strings():
    assert _parse('["F1", "f1", " ", 3, "Equalized  Odds"]') == ["F1", "Equalized Odds"]


def test_parse_errors():
    with pytest.raises(LLMError):
        _parse("I could not find any metrics.")
    with pytest.raises(LLMError):
        _parse("[F1, BLEU]")


def test_ask_llm_truncates_and_passes_options(fake_litellm):
    fake_litellm.reply = '["a", "b", "c"]'
    assert ask_llm("text", "ollama/x", top_k=2, api_key="k") == ["a", "b"]
    call = fake_litellm.calls[0]
    assert call["temperature"] == 0 and call["api_key"] == "k"
    assert "at most 2 metrics" in call["messages"][1]["content"]


def test_ask_llm_wraps_provider_errors(fake_litellm):
    fake_litellm.error = ValueError("bad key")
    with pytest.raises(LLMError, match="bad key"):
        ask_llm("text", "gpt-x")
