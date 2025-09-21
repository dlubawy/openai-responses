"""Unit tests for :mod:`openai_responses.api.api_server`.

These tests exercise a *subset* of the complex logic inside the
``generate_response`` helper, but they are written in a way that
requires minimal external dependencies.  The real implementation
relies heavily on the :class:`HarmonyEncoding` type from the
``openai_harmony`` package.  For the purposes of the tests we
mock an encoding that can return controlled results while avoiding
any heavy-side effects such as actual token decoding or OpenAI
inference.

The tests cover the following paths:

* Plain assistant text output (``final`` channel).
* Function call output.
* Web‑search tool call output (happy path).
* Function call except with an exception during token parsing –
  verifies that an error is attached to the response.
* Debug mode rendering – ensures debug payload is included.

Each test uses :class:`fastapi.testclient.TestClient` to hit the
``/v1/responses`` endpoint, which internally uses the mocked
``generate_response`` logic.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi.testclient import TestClient
from openai_harmony import HarmonyEncoding, HarmonyEncodingName, load_harmony_encoding

from openai_responses.api.api_server import create_api_server
from openai_responses.api.types import ModelConnection

# ---------------------------------------------------------------------------
# Helper data structures -----------------------------------------------------
# ---------------------------------------------------------------------------


class DummyMessage:
    """Represents the minimal interface required by the server.

    The server calls ``to_dict`` to retrieve a dictionary with the
    keys ``recipient``, ``channel`` and ``content``.  ``content`` is a
    list of dictionaries matching the shape expected by the
    original implementation.
    """

    def __init__(self, recipient: str, channel: str, content: List[Dict[str, Any]]):
        self._data: Dict[str, Any] = {
            "recipient": recipient,
            "channel": channel,
            "content": content,
        }

    def to_dict(self) -> Dict[str, Any]:
        return self._data


class DummyEncoding(HarmonyEncoding):
    """Mock of :class:`HarmonyEncoding`.

    Only the methods used by the tests are implemented.  They are
    intentionally simple: ``decode_utf8`` just joins the token
    integer list into a string of their ``chr`` values, while
    ``parse_messages_from_completion_tokens`` simply returns the
    pre‑configured list of :class:`DummyMessage` objects.
    """

    def __init__(
        self, messages: List[DummyMessage] | None = None, raise_for: bool = False
    ):
        self._messages = messages or []
        self._raise_for = raise_for
        self._encoder = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self._inner = self._encoder._inner

    # The tests trigger this method with a deterministic list of
    # ``input_tokens``/``output_tokens``.  In normal operation the
    # behaviour is irrelevant as we have already set what the
    # server should parse.
    def parse_messages_from_completion_tokens(self, tokens: List[int], role):  # type: ignore[override]
        if self._raise_for:
            raise RuntimeError("Simulated parsing failure")
        return self._messages

    def decode_utf8(self, tokens: List[int]):  # type: ignore[override]
        return "".join(self._encoder.decode_utf8([t]) for t in tokens if t)

    def encode(
        self, text: str, *, allowed_special="all"
    ) -> List[int]:  # pragma: no cover
        return [ord(c) for c in text]


class DummyModelConnection(ModelConnection):
    """A minimal stub that satisfies the ``model_connection`` argument.

    The API server does not call any methods on this object for the
    routes exercised in these tests, so only a dummy ``close`` method
    is provided.
    """

    def close(self, session_id: Optional[str] = None):
        pass


def client_for_messages(messages: List[DummyMessage], raise_parse: bool = False):
    """Return a :class:`TestClient` configured to return ``messages``.

    ``raise_parse`` forces ``parse_messages_from_completion_tokens``
    to raise an exception for debugging‑mode error handling tests.
    """

    encoding = DummyEncoding(messages, raise_for=raise_parse)
    model = DummyModelConnection()
    app = create_api_server(model_connection=model, encoding=encoding)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


def test_final_output_returns_text_content():
    """A response that contains only plain assistant text.

    The server should return an ``Item`` with a single
    :class:`TextContentItem`.
    """

    final_msg = DummyMessage(
        recipient="assistant",
        channel="final",
        content=[{"text": "Hello world!"}],
    )
    client = client_for_messages([final_msg])

    body = {
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "text", "text": "Hi"}],
                "status": None,
            }
        ],
        "reasoning": {"effort": "medium"},
        "model": "gpt-oss-120b",
        "stream": False,
        "previous_response_id": None,
    }
    resp = client.post("/v1/responses", json=body)
    assert resp.status_code == 200
    data = resp.json()
    output = data.get("output", [])
    assert len(output) == 1
    item = output[0]
    assert item["type"] == "message"
    content = item["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "output_text"
    assert content[0]["text"] == "Hello world!"


def test_function_call_output():
    """Function calls should be translated into a ``FunctionCallItem``."""

    fc_msg = DummyMessage(
        recipient="functions.add",  # simulated tool prefix
        channel="analysis",
        content=[{"text": '{"a":1, "b":2}'}],
    )
    client = client_for_messages([fc_msg])

    body = {
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "text", "text": "Add numbers"}],
                "status": None,
            }
        ],
        "reasoning": {"effort": "medium"},
        "model": "gpt-oss-120b",
        "stream": False,
        "tools": [
            {"type": "function", "name": "add", "parameters": {}, "description": ""}
        ],
    }
    resp = client.post("/v1/responses", json=body)
    assert resp.status_code == 200
    data = resp.json()
    output = data.get("output", [])
    assert len(output) == 1
    fc = output[0]
    assert fc["type"] == "function_call"
    assert fc["name"] == "add"
    assert fc["arguments"] == '{"a":1, "b":2}'


def test_web_search_call_and_citations():
    """Verify the code path for a web‑search tool call.

    The test uses a mocked :class:`SimpleWebSearchTool` that
    pretends to parse arguments and normalise citations.
    """

    # Prepare a message that triggers a web_search tool.
    web_msg = DummyMessage(
        recipient="web_search.search",
        channel="analysis",
        content=[{"text": '{"query":"python"}'}],
    )
    final_msg = DummyMessage(
        recipient="",
        channel="final",
        content=[{"text": "Result is X"}],
    )
    client = client_for_messages([web_msg, final_msg])

    # Patch the SimpleWebSearchTool used by the server.
    from openai_responses.tools.simple_web_search import simple_web_search_tool

    orig_tool = simple_web_search_tool.SimpleWebSearchTool

    class TestTool(orig_tool):
        def process_arguments(self, msg):  # pragma: no cover - trivial
            return {"query": "python"}

        def normalize_citations(self, text):  # pragma: no cover - trivial
            return text, [], False

    simple_web_search_tool.SimpleWebSearchTool = TestTool
    try:
        body = {
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "text", "text": "Search"}],
                    "status": None,
                }
            ],
            "tools": [
                {"type": "web_search"},
            ],
            "reasoning": {"effort": "medium"},
            "model": "gpt-oss-120b",
            "stream": False,
        }
        resp = client.post("/v1/responses", json=body)
        assert resp.status_code == 200
        data = resp.json()
        output = data["output"]
        # Search call item should appear followed by the final message
        assert any(item["type"] == "web_search_call" for item in output)
        # Final message should have text
        assert any(item["type"] == "message" for item in output)
    finally:
        simple_web_search_tool.SimpleWebSearchTool = orig_tool


def test_parse_error_sets_error_and_returns_debug_tokens(monkeypatch):
    """When token parsing fails in debug mode an error object is placed on the response.

    The error message must contain the exception raised by parsing.
    """

    # The server will call ``generate_response`` with debug_mode=True.
    client = client_for_messages([], raise_parse=True)

    body = {
        "input": [],
        "reasoning": {"effort": "medium"},
        "model": "gpt-oss-120b",
        "stream": False,
        "metadata": {"__debug": True},
    }
    resp = client.post("/v1/responses", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["error"] is not None
    assert data["error"]["code"] == "invalid_function_call"
    assert "Simulated parsing failure" in data["error"]["message"]
    # Debug information should be present
    assert "__debug" in data.get("metadata", {})


def test_debug_mode_includes_payload(monkeypatch):
    """Debug mode should attach decoded token strings to the response metadata."""

    # Simple final message
    final_msg = DummyMessage(recipient="", channel="final", content=[{"text": "hi"}])
    client = client_for_messages([final_msg])

    body = {
        "input": [],
        "reasoning": {"effort": "medium"},
        "model": "gpt-oss-120b",
        "stream": False,
        "metadata": {"__debug": True},
    }
    resp = client.post("/v1/responses", json=body)
    assert resp.status_code == 200
    data = resp.json()
    meta = data.get("metadata", {})
    print(meta)
    assert "__debug_input" in meta
    assert "__debug_output" in meta
    assert meta["__debug"] is not None
