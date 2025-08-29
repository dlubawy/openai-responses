"""
NOTE: this is a stitched together implementation that uses Ollama for inference. It's primarily used
for testing and development. It does not leverage any prompt caching or other optimizations and
can therefore be slow between turns.
"""

import os
import threading
import time
from typing import Optional

import ollama
from openai_harmony import HarmonyEncodingName, load_harmony_encoding

from openai_responses.api.types import ModelConnection

EOS_TOKEN = 200002  # only used on hard timeout

# Tunables
POLL_INTERVAL_S = 0.01  # 10ms between buffer checks
CALL_MAX_WAIT_S = 0.250  # max time to block inside a single infer call
NO_TOKEN_TIMEOUT_S = 15.0  # overall inactivity timeout before emitting EOS
FIRST_BYTE_TIMEOUT_S = float(
    os.getenv("OLLAMA_TIMEOUT", "30.0")
)  # time to wait for first token before EOS


def lcp(cache: list[int], inp: list[int]) -> list[int]:
    i = 0
    max_len = min(len(cache), len(inp))
    while i < max_len and cache[i] == inp[i]:
        i += 1
    return cache[:i]


def _now():
    return time.monotonic()


def setup_model(checkpoint: str) -> ModelConnection:
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    model_name = checkpoint
    client = ollama.Client(host="localhost")

    class OllamaModelConnection(ModelConnection):
        # Shared state
        _token_buffer: list[int] = []
        _buffer_lock = threading.Lock()
        _stream_thread: Optional[threading.Thread] = None
        _stream_done = threading.Event()
        _stream_error: Optional[Exception] = None
        _last_progress_ts: float = 0.0  # updated whenever we enqueue or dequeue tokens
        _previous_request_tokens: list[int] = []
        _close_connection = threading.Event()

        def _touch_progress(self):
            global _last_progress_ts
            _last_progress_ts = _now()

        def _reset_stream_state(self):
            with self._buffer_lock:
                _token_buffer = []
            self._stream_done.clear()
            self._stream_thread = None
            self._stream_error = None
            self._close_connection.clear()
            self._touch_progress()

        def _start_stream(self, token_ids: list[int], temperature: float):
            prompt_text = encoding.decode(token_ids)

            def run():
                accum_text = []
                last_len = 0  # number of tokens already emitted
                toks = None

                try:
                    for chunk in client.generate(
                        model=model_name,
                        prompt=prompt_text,
                        stream=True,
                        options={"temperature": temperature},
                        raw=True,
                    ):
                        if self._close_connection.is_set():
                            with self._buffer_lock:
                                self._token_buffer.append(EOS_TOKEN)
                            self._touch_progress
                            break

                        if isinstance(chunk.response, str):
                            accum_text.append(chunk.response)
                            toks = encoding.encode(
                                "".join(accum_text), allowed_special="all"
                            )
                            if len(toks) > last_len:
                                new_toks = toks[last_len:]
                                with self._buffer_lock:
                                    self._token_buffer.extend(new_toks)
                                last_len = len(toks)
                                self._touch_progress()

                        if chunk.done:
                            with self._buffer_lock:
                                self._token_buffer.append(EOS_TOKEN)
                            self._touch_progress()

                    self._stream_done.set()

                except Exception as e:
                    self._stream_error = e
                    self._stream_done.set()

            t = threading.Thread(target=run, name="ollama-stream", daemon=True)
            t.start()
            return t

        def close(self):
            self._close_connection.set()

        def infer_next_token(
            self, tokens: list[int], temperature: float = 0.0, new_request: bool = False
        ) -> int:
            """
            - Starts a new Ollama stream on new_request.
            - Forwards tokens as they arrive.
            - Only emits EOS_TOKEN if we exceed an inactivity timeout.
            """
            global _stream_thread

            if new_request:
                self._reset_stream_state()
                _stream_thread = self._start_stream(
                    token_ids=tokens, temperature=temperature
                )

            if self._stream_error is not None:
                raise RuntimeError(f"Ollama stream error: {self._stream_error!r}")

            received_token = False
            while not self._stream_done.is_set() and not received_token:
                if self._buffer_lock.acquire(blocking=False):
                    if self._token_buffer:
                        received_token = True
                    self._buffer_lock.release()

            with self._buffer_lock:
                if self._token_buffer:
                    tok = self._token_buffer.pop(0)
                    self._touch_progress()
                    return tok

            # If we reach here, we still haven't got a token—ask the caller to call again soon.
            # Return a harmless token that the server will replace/ignore if your interface supports it.
            # If your interface does NOT allow a sentinel, keep the short-blocking behavior above.
            return (
                EOS_TOKEN if False else 0
            )  # replace `0` with a PAD/NOOP token your server ignores

    return OllamaModelConnection()
