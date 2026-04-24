"""LLM client with OpenAI-compatible API support."""

import os
from typing import Optional


def get_llm_response(
    prompt: str,
    stop: Optional[list] = None,
    model: str = "deepseek-v3-0324",
    base_url: Optional[str] = None,
    echo_stream: bool = False,
    temperature: float = 0.7,
    max_tokens: int = 32000,
) -> str:
    """Call an OpenAI-compatible LLM API.

    Supports streaming with stop-tag detection.
    Default model and base URL can be overridden via environment
    variables WEBEXPERT_MODEL and WEBEXPERT_BASE_URL.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package is required: pip install openai")

    url = base_url or os.environ.get(
        "WEBEXPERT_BASE_URL", "https://api.openai.com/v1"
    )
    model = model or os.environ.get("WEBEXPERT_MODEL", "deepseek-v3-0324")
    api_key = os.environ.get("OPENAI_API_KEY", "dummy")

    client = OpenAI(api_key=api_key, base_url=url)

    if echo_stream:
        # Streaming mode with stop-tag detection
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stop=stop or [],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        collected = []
        buffer = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            collected.append(delta)
            buffer += delta
            # Check for stop tags
            if stop:
                for tag in stop:
                    if tag in buffer:
                        return "".join(collected)
            print(delta, end="", flush=True)
        return "".join(collected)
    else:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stop=stop or [],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
