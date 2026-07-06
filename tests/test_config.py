import pytest

from rlm_kit.config import RLMConfig

ENV_VARS = [
    "RLM_MAIN_MODEL",
    "RLM_SUB_MODEL",
    "AI_MODEL_NAME",
    "SUB_AI_MODEL_NAME",
    "RLM_API_KEY",
    "AI_API_KEY",
    "RLM_BASE_URL",
    "AI_BASE_URL",
    "RLM_INTERPRETER",
    "RLM_ADAPTER",
    "RLM_MAX_TOKENS",
    "RLM_ALLOW_INSECURE_SANDBOX",
    "RLM_MAX_ITERATIONS",
    "RLM_MAX_LLM_CALLS",
    "RLM_MAX_OUTPUT_CHARS",
    "RLM_MAX_RETRIES",
    "RLM_OBSERVE",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_defaults():
    cfg = RLMConfig.from_env()
    assert cfg.main_model == "openai/gpt-4o"
    assert cfg.sub_model == "openai/gpt-4o"  # mirrors main when unset
    assert cfg.interpreter == "pyodide"
    assert cfg.adapter == "json"  # default: schema-guided structured output
    assert cfg.max_tokens == 8192  # generous default (not None → no tiny server cap)
    assert cfg.allow_insecure_sandbox is False
    assert cfg.observe is False
    assert cfg.max_retries == 1   # default: no whole-RLM retry (it would multiply the iteration budget)
    assert cfg.max_iterations == 10
    assert cfg.max_llm_calls == 30
    assert cfg.max_output_chars == 10_000  # matches dspy.RLM's own default


def test_sub_model_defaults_to_main(monkeypatch):
    monkeypatch.setenv("RLM_MAIN_MODEL", "openai/gpt-5")
    cfg = RLMConfig.from_env()
    assert cfg.sub_model == "openai/gpt-5"


def test_ai_model_name_fallback(monkeypatch):
    # Drop-in for projects already using AI_MODEL_NAME / SUB_AI_MODEL_NAME.
    monkeypatch.setenv("AI_MODEL_NAME", "openai/gpt-4o-mini")
    monkeypatch.setenv("SUB_AI_MODEL_NAME", "openai/gpt-4o")
    cfg = RLMConfig.from_env()
    assert cfg.main_model == "openai/gpt-4o-mini"
    assert cfg.sub_model == "openai/gpt-4o"


def test_rlm_vars_win_over_ai_vars(monkeypatch):
    monkeypatch.setenv("AI_MODEL_NAME", "openai/legacy")
    monkeypatch.setenv("RLM_MAIN_MODEL", "openai/new")
    assert RLMConfig.from_env().main_model == "openai/new"


def test_sub_falls_back_to_main_when_only_main_ai_set(monkeypatch):
    monkeypatch.setenv("AI_MODEL_NAME", "openai/solo")
    cfg = RLMConfig.from_env()
    assert cfg.main_model == "openai/solo"
    assert cfg.sub_model == "openai/solo"


def test_api_key_fallback_to_ai_api_key(monkeypatch):
    monkeypatch.setenv("AI_API_KEY", "fallback-key")
    assert RLMConfig.from_env().api_key == "fallback-key"
    monkeypatch.setenv("RLM_API_KEY", "primary-key")
    assert RLMConfig.from_env().api_key == "primary-key"


@pytest.mark.parametrize(
    "raw,expected",
    [("1", True), ("true", True), ("YES", True), ("on", True),
     ("0", False), ("false", False), ("", False), ("nope", False)],
)
def test_bool_parsing(monkeypatch, raw, expected):
    monkeypatch.setenv("RLM_OBSERVE", raw)
    assert RLMConfig.from_env().observe is expected


def test_int_parsing(monkeypatch):
    monkeypatch.setenv("RLM_MAX_RETRIES", "7")
    monkeypatch.setenv("RLM_MAX_LLM_CALLS", "99")
    cfg = RLMConfig.from_env()
    assert cfg.max_retries == 7
    assert cfg.max_llm_calls == 99


def test_adapter_from_env(monkeypatch):
    monkeypatch.setenv("RLM_ADAPTER", "json")
    assert RLMConfig.from_env().adapter == "json"


def test_max_tokens_from_env(monkeypatch):
    assert RLMConfig.from_env().max_tokens == 8192       # unset → generous default
    monkeypatch.setenv("RLM_MAX_TOKENS", "4096")
    assert RLMConfig.from_env().max_tokens == 4096       # explicit override


def test_max_output_chars_from_env(monkeypatch):
    assert RLMConfig.from_env().max_output_chars == 10_000   # unset → dspy's default
    monkeypatch.setenv("RLM_MAX_OUTPUT_CHARS", "50000")
    assert RLMConfig.from_env().max_output_chars == 50_000   # explicit override


def test_unknown_adapter_rejected():
    with pytest.raises(ValueError):
        RLMConfig(main_model="m", sub_model="m", adapter="bogus")


def test_unknown_interpreter_rejected():
    with pytest.raises(ValueError):
        RLMConfig(main_model="m", sub_model="m", interpreter="bogus")


def test_zero_retries_rejected():
    with pytest.raises(ValueError):
        RLMConfig(main_model="m", sub_model="m", max_retries=0)
