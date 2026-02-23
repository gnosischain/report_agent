# report_agent/connectors/llm/__init__.py
"""
LLM provider factory.

Creates the appropriate LLM analyzer and chat client based on configuration.
Centralizes provider switching so callers don't need to know about specific
implementations.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from report_agent.config import LLMConfig
    from report_agent.pipeline.stages.llm_analyzer import LLMAnalyzer


def create_analyzer(config: LLMConfig) -> LLMAnalyzer:
    """
    Create the right LLMAnalyzer implementation based on provider config.

    Args:
        config: LLMConfig with provider, api keys, and model names.

    Returns:
        An LLMAnalyzer instance (OpenAIAnalyzer or ClaudeAnalyzer).

    Raises:
        ValueError: If the provider is not supported.
    """
    provider = config.provider.lower()

    if provider == "anthropic":
        from report_agent.pipeline.stages.claude_analyzer import ClaudeAnalyzer
        return ClaudeAnalyzer(
            api_key=config.anthropic_api_key,
            model_name=config.anthropic_model,
        )
    elif provider == "openai":
        from report_agent.pipeline.stages.openai_analyzer import OpenAIAnalyzer
        return OpenAIAnalyzer(
            api_key=config.api_key,
            model_name=config.model,
        )
    else:
        raise ValueError(
            f"Unsupported LLM provider: '{provider}'. "
            f"Set LLM_PROVIDER to 'openai' or 'anthropic'."
        )


def create_chat_client(config: LLMConfig):
    """
    Create a lightweight chat client for text-only LLM calls
    (no code execution). Used by summary_service and cross_metric_service
    for calls that don't need a sandboxed environment.

    Returns a (client, model_name) tuple. The client exposes a
    provider-specific API -- callers use create_chat_completion() below.
    """
    provider = config.provider.lower()

    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(
            api_key=config.anthropic_api_key,
            max_retries=0,
        )
        return client, config.anthropic_model, provider
    elif provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=config.api_key)
        return client, config.model, provider
    else:
        raise ValueError(f"Unsupported LLM provider: '{provider}'.")


def create_chat_completion(
    client,
    model: str,
    provider: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
) -> tuple[str, int, int]:
    """
    Provider-agnostic text chat completion (no code execution).

    Returns:
        (response_text, input_tokens, output_tokens)
    """
    if provider == "anthropic":
        resp = client.messages.create(
            model=model,
            max_tokens=8192,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=temperature,
        )
        text = ""
        for block in (resp.content or []):
            if getattr(block, "type", None) == "text":
                text += getattr(block, "text", "")

        usage = getattr(resp, "usage", None)
        input_tokens = getattr(usage, "input_tokens", 0) or 0 if usage else 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0 if usage else 0
        return text, input_tokens, output_tokens
    else:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0 if usage else 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0 if usage else 0
        return text, input_tokens, output_tokens


def create_code_execution_client(config: LLMConfig):
    """
    Create a client + config for code-execution LLM calls.

    Used by cross_metric_service which needs code execution but doesn't
    go through the per-metric pipeline.

    Returns (client, model_name, provider) tuple.
    """
    provider = config.provider.lower()

    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(
            api_key=config.anthropic_api_key,
            max_retries=0,
        )
        return client, config.anthropic_model, provider
    elif provider == "openai":
        import httpx
        from openai import OpenAI
        client = OpenAI(
            api_key=config.api_key,
            max_retries=0,
            http_client=httpx.Client(
                timeout=httpx.Timeout(300.0, connect=10.0),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            ),
        )
        return client, config.model, provider
    else:
        raise ValueError(f"Unsupported LLM provider: '{provider}'.")
