"""
LLM Provider abstraction — plug any model into the SQL AI Agent or SQLAS eval framework.

Supported providers
───────────────────
Key format                         What it uses
"azure"                            Azure OpenAI   (reads from .env settings)
"openai:gpt-4o"                    OpenAI direct  (requires OPENAI_API_KEY)
"openai:gpt-4o-mini"               OpenAI mini
"anthropic:claude-opus-4-7"        Anthropic      (requires ANTHROPIC_API_KEY)
"anthropic:claude-sonnet-4-6"      Anthropic Sonnet
"ollama:llama3"                    Ollama local   (ollama serve must be running)
"ollama:sqlcoder"                  SQLCoder local (7B fine-tuned for SQL)
"ollama:deepseek-coder"            DeepSeek local
"compat:model@https://..."         Any OpenAI-compatible endpoint
                                   (vLLM, LM Studio, Together AI, Groq, etc.)

Usage in eval
─────────────
    python eval_runner.py --provider anthropic:claude-opus-4-7
    python eval_runner.py --compare azure,anthropic:claude-opus-4-7,ollama:sqlcoder
    python eval_runner.py --provider ollama:sqlcoder --judge azure
"""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """
    Minimal interface every provider must implement.
    complete() is intentionally synchronous — it wraps the blocking SDK calls
    that are already used throughout the codebase.
    """

    @abstractmethod
    def complete(self, messages: list[dict], max_tokens: int = 2000) -> str:
        """Send a chat message list, return the assistant text reply."""
        ...

    def complete_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int = 2000,
    ) -> tuple[dict, list[dict]]:
        """
        Send messages with tool definitions. Returns:
          assistant_msg  — dict to append to conversation history
          tool_calls     — list of {id, name, arguments: dict}

        Default: no tool support. Override in providers that support function calling.
        """
        raise NotImplementedError(
            f"{self.name} does not support tool calling. "
            "Use pipeline mode or switch to azure/openai/anthropic."
        )

    @property
    def supports_tool_calling(self) -> bool:
        return False

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique human-readable ID shown in eval reports and MLflow tags."""
        ...

    @property
    def model_id(self) -> str:
        """Short model name (everything after the first '/')."""
        return self.name.split("/", 1)[-1]


# ── Azure OpenAI ──────────────────────────────────────────────────────────────

class AzureOpenAIProvider(LLMProvider):
    def __init__(self, endpoint: str, api_key: str, deployment: str, api_version: str):
        from openai import AzureOpenAI
        self._client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )
        self._deployment = deployment

    def complete(self, messages: list[dict], max_tokens: int = 2000) -> str:
        resp = self._client.chat.completions.create(
            model=self._deployment,
            messages=messages,
            max_completion_tokens=max_tokens,
        )
        return resp.choices[0].message.content

    def complete_with_tools(self, messages, tools, max_tokens=2000):
        import json
        resp = self._client.chat.completions.create(
            model=self._deployment,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_completion_tokens=max_tokens,
        )
        msg = resp.choices[0].message
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments or "{}"),
                })
        assistant_msg = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
        return assistant_msg, tool_calls

    @property
    def supports_tool_calling(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return f"azure/{self._deployment}"


# ── OpenAI ────────────────────────────────────────────────────────────────────

class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def complete(self, messages: list[dict], max_tokens: int = 2000) -> str:
        resp = self._client.chat.completions.create(
            model=self._model, messages=messages, max_tokens=max_tokens,
        )
        return resp.choices[0].message.content

    def complete_with_tools(self, messages, tools, max_tokens=2000):
        import json
        resp = self._client.chat.completions.create(
            model=self._model, messages=messages, tools=tools,
            tool_choice="auto", max_tokens=max_tokens,
        )
        msg = resp.choices[0].message
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append({"id": tc.id, "name": tc.function.name,
                                   "arguments": json.loads(tc.function.arguments or "{}")})
        assistant_msg = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
        return assistant_msg, tool_calls

    @property
    def supports_tool_calling(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return f"openai/{self._model}"


# ── Anthropic Claude ──────────────────────────────────────────────────────────

class AnthropicProvider(LLMProvider):
    """
    Claude separates the system prompt from the message list.
    This adapter extracts the system role automatically.
    """
    def __init__(self, api_key: str, model: str = "claude-opus-4-7"):
        try:
            import anthropic
        except ImportError:
            raise ImportError("Install anthropic SDK: pip install anthropic")
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def complete(self, messages: list[dict], max_tokens: int = 2000) -> str:
        system = next((m["content"] for m in messages if m["role"] == "system"), None)
        user_msgs = [m for m in messages if m["role"] != "system"]
        kwargs: dict = {"model": self._model, "max_tokens": max_tokens, "messages": user_msgs}
        if system:
            kwargs["system"] = system
        resp = self._client.messages.create(**kwargs)
        return resp.content[0].text

    def complete_with_tools(self, messages, tools, max_tokens=2000):
        """Converts OpenAI-format messages/tools to Anthropic format internally."""
        import json
        system = next((m["content"] for m in messages if m["role"] == "system"), None)
        anthropic_tools = [
            {"name": t["function"]["name"],
             "description": t["function"]["description"],
             "input_schema": t["function"]["parameters"]}
            for t in tools
        ]
        anthropic_msgs = _openai_msgs_to_anthropic(
            [m for m in messages if m["role"] != "system"]
        )
        kwargs: dict = {
            "model": self._model, "max_tokens": max_tokens,
            "messages": anthropic_msgs, "tools": anthropic_tools,
        }
        if system:
            kwargs["system"] = system
        resp = self._client.messages.create(**kwargs)

        tool_calls, text_content = [], ""
        for block in resp.content:
            if block.type == "tool_use":
                tool_calls.append({"id": block.id, "name": block.name, "arguments": block.input})
            elif block.type == "text":
                text_content += block.text

        assistant_msg: dict = {"role": "assistant", "content": text_content}
        if tool_calls:
            assistant_msg["tool_calls"] = [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])}}
                for tc in tool_calls
            ]
        return assistant_msg, tool_calls

    @property
    def supports_tool_calling(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return f"anthropic/{self._model}"


# ── Ollama (local models) ─────────────────────────────────────────────────────

class OllamaProvider(LLMProvider):
    """
    Runs any Ollama model locally — no API key needed.
    Models worth testing for SQL generation:
      ollama pull sqlcoder      (7B, fine-tuned for SQL)
      ollama pull deepseek-coder
      ollama pull llama3
      ollama pull mistral
    """
    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434"):
        self._model = model
        self._base_url = base_url.rstrip("/")

    def complete(self, messages: list[dict], max_tokens: int = 2000) -> str:
        try:
            import httpx
        except ImportError:
            raise ImportError("Install httpx: pip install httpx")
        resp = httpx.post(
            f"{self._base_url}/api/chat",
            json={
                "model": self._model,
                "messages": messages,
                "stream": False,
                "options": {"num_predict": max_tokens},
            },
            timeout=180.0,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    @property
    def name(self) -> str:
        return f"ollama/{self._model}"


# ── OpenAI-compatible (vLLM, LM Studio, Together AI, Groq, Fireworks…) ────────

class OpenAICompatibleProvider(LLMProvider):
    """
    Any endpoint that implements the OpenAI chat completions spec.
    Key format: "compat:model-name@https://api.endpoint.com/v1"
    """
    def __init__(self, base_url: str, model: str, api_key: str = "not-needed"):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._base_url = base_url

    def complete(self, messages: list[dict], max_tokens: int = 2000) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content

    @property
    def name(self) -> str:
        domain = self._base_url.split("//", 1)[-1].split("/")[0]
        return f"compat/{domain}/{self._model}"


# ── Factory ───────────────────────────────────────────────────────────────────

def _openai_msgs_to_anthropic(messages: list[dict]) -> list[dict]:
    """
    Convert OpenAI-format conversation history to Anthropic format.
    Handles tool call messages and tool result messages.
    """
    import json
    result = []
    for msg in messages:
        role = msg["role"]
        if role == "assistant":
            if msg.get("tool_calls"):
                content = []
                if msg.get("content"):
                    content.append({"type": "text", "text": msg["content"]})
                for tc in msg["tool_calls"]:
                    content.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "input": json.loads(tc["function"].get("arguments", "{}")),
                    })
                result.append({"role": "assistant", "content": content})
            else:
                result.append({"role": "assistant", "content": msg.get("content", "")})
        elif role == "tool":
            # Anthropic expects tool results as user messages
            result.append({
                "role": "user",
                "content": [{"type": "tool_result",
                              "tool_use_id": msg.get("tool_call_id", ""),
                              "content": str(msg.get("content", ""))}],
            })
        else:
            result.append({"role": role, "content": msg.get("content", "")})
    return result


def get_provider(key: str, settings) -> LLMProvider:
    """
    Instantiate a provider from a key string and settings object.

    Examples
    --------
    get_provider("azure", settings)
    get_provider("openai:gpt-4o-mini", settings)
    get_provider("anthropic:claude-sonnet-4-6", settings)
    get_provider("ollama:sqlcoder", settings)
    get_provider("compat:mistral-7b@http://localhost:1234/v1", settings)
    """
    key = (key or "azure").strip()

    if key == "azure":
        return AzureOpenAIProvider(
            endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            deployment=settings.azure_openai_deployment_name,
            api_version=settings.azure_openai_api_version,
        )

    if key.startswith("openai:"):
        model = key.split(":", 1)[1]
        if not getattr(settings, "openai_api_key", ""):
            raise ValueError("Set OPENAI_API_KEY in .env to use the openai provider.")
        return OpenAIProvider(api_key=settings.openai_api_key, model=model)

    if key.startswith("anthropic:"):
        model = key.split(":", 1)[1]
        if not getattr(settings, "anthropic_api_key", ""):
            raise ValueError("Set ANTHROPIC_API_KEY in .env to use the anthropic provider.")
        return AnthropicProvider(api_key=settings.anthropic_api_key, model=model)

    if key.startswith("ollama:"):
        model = key.split(":", 1)[1]
        base_url = getattr(settings, "ollama_base_url", "http://localhost:11434")
        return OllamaProvider(model=model, base_url=base_url)

    if key.startswith("compat:"):
        rest = key[len("compat:"):]
        if "@" not in rest:
            raise ValueError("compat provider format: compat:model-name@https://endpoint/v1")
        model, base_url = rest.rsplit("@", 1)
        api_key = getattr(settings, "compat_api_key", "not-needed")
        return OpenAICompatibleProvider(base_url=base_url, model=model, api_key=api_key)

    raise ValueError(
        f"Unknown provider key: '{key}'.\n"
        "Valid formats:\n"
        "  azure\n"
        "  openai:gpt-4o\n"
        "  anthropic:claude-opus-4-7\n"
        "  ollama:sqlcoder\n"
        "  compat:model@https://endpoint/v1"
    )
