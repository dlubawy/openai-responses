"""Unit tests for :mod:`openai_responses.api.inference.ollama`.

These tests exercise the public :func:`setup_model` helper which constructs
an :class:`OllamaModelConnection`.  Interaction with the real Ollama/OpenAI
dependency stack is replaced with lightweight mocks so that the tests run in
isolation and remain fast.

The mock objects intentionally mimic the minimal interface required by the
implementation – namely ``ollama.Client`` and its ``generate`` method, plus the
encoding used by :mod:`openai_harmony`.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Iterator

import pytest
from openai_harmony import HarmonyEncoding, HarmonyEncodingName, load_harmony_encoding

from openai_responses.api.inference import ollama as ollama_mod

# Helper mocks ---------------------------------------------------------------


class DummyEncoding(HarmonyEncoding):
    """Very small stub of an encoding that simply maps characters to
    integer token ids using their ordinal value.
    """

    def __init__(self):
        self._encoder = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self._inner = self._encoder._inner

    def encode(self, text: str, *, allowed_special: str | None = None):  # type: ignore[override]
        return self._encoder.encode(text)

    def decode(self, ids: list[int]):  # pragma: no cover - trivial
        return self._encoder.decode(ids)


class DummyChunk:
    """Simple object mimicking the ``ollama`` chunk API.

    Every instance holds ``response`` (a string) and ``done`` flag.
    """

    def __init__(self, response: str, done: bool):
        self.response = response
        self.done = done


class DummyClient:
    """Mock of :class:`ollama.Client`.

    The :py:meth:`generate` method yields a fixed sequence of chunks defined by
    the ``chunk_seq`` attribute. The attribute is consumed each time ``generate``
    is called.
    """

    def __init__(self, chunk_seq: Iterator[DummyChunk]):
        self.chunk_seq = chunk_seq

    def generate(self, *_, **kwargs):  # pragma: no cover - interface only
        return self.chunk_seq


def _setup_model_with_mocks(chunk_seq: Iterator[DummyChunk]):
    """Convenience wrapper to patch the environment and call
    :func:`ollama_mod.setup_model`.
    """

    def fake_load_harmony_encoding(name):  # pragma: no cover - trivial
        return DummyEncoding()

    def fake_client_factory(**kwargs):  # pragma: no cover - trivial
        return DummyClient(chunk_seq)

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(ollama_mod, "load_harmony_encoding", fake_load_harmony_encoding)
        mp.setattr(ollama_mod, "ollama", SimpleNamespace(Client=fake_client_factory))
        return ollama_mod.setup_model("dummy_checkpoint")


# Tests ---------------------------------------------------------------------


def test_infer_sequence_of_tokens_and_eos(tmp_path, monkeypatch):
    # Sequence: "Hi", then " there", then done.
    chunks = iter(
        [
            DummyChunk("Hi", False),
            DummyChunk(" there", False),
            DummyChunk("", True),
        ]
    )
    conn = _setup_model_with_mocks(chunks)
    session_id = "testsid"
    token1 = conn.infer_next_token(
        [], temperature=0.0, new_request=True, session_id=session_id
    )
    token2 = conn.infer_next_token(
        [], temperature=0.0, new_request=False, session_id=session_id
    )
    assert token1 == 12194
    assert token2 == 1354
    eos = conn.infer_next_token(
        [], temperature=0.0, new_request=False, session_id=session_id
    )
    assert eos == ollama_mod.EOS_TOKEN


def test_stream_lifetime_and_close(tmp_path, monkeypatch):
    def never_yield():
        for _ in range(100):
            # Never set done=True
            yield DummyChunk("", False)

    conn = _setup_model_with_mocks(iter(never_yield()))
    session_id = "mytest"
    conn.infer_next_token([], new_request=True, session_id=session_id)
    # Close call should return EOS_TOKEN
    conn.close_session(session_id)
    tok = conn.infer_next_token([], new_request=True, session_id=session_id)
    assert tok == ollama_mod.EOS_TOKEN


def test_error_handling_on_generate_exception(tmp_path, monkeypatch):
    def raise_error():  # pragma: no cover
        for i in range(1):
            if i == 0:
                raise RuntimeError("boom")
            yield DummyChunk(chr(i), False)

    conn = _setup_model_with_mocks(iter(raise_error()))
    with pytest.raises(RuntimeError, match="boom"):
        conn.infer_next_token([], new_request=True, session_id="sid")
