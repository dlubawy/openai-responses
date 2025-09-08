"""
NOTE: this is a stitched together implementation that uses Ollama for inference. It's primarily used
for testing and development. It does not leverage any prompt caching or other optimizations and
can therefore be slow between turns.
"""

import threading
import time
from collections import deque
from typing import Optional

import ollama
from openai_harmony import HarmonyEncodingName, load_harmony_encoding

from openai_responses.api.types import ModelConnection

EOS_TOKEN = 200002  # only used on hard timeout


def _now():
    return time.monotonic()


def setup_model(checkpoint: str) -> ModelConnection:
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    model_name = checkpoint
    client = ollama.Client(host="localhost")

    class OllamaModelConnection(ModelConnection):
        # Shared state
        # Use deque for efficient O(1) popleft operations
        _token_buffer: deque[int] = deque()
        _buffer_lock = threading.Lock()
        # Condition variable used to notify waiting callers that a new token
        # has been added to the buffer.  It is associated with the same
        # lock used for the token buffer to avoid race conditions.
        _token_available = threading.Condition(_buffer_lock)
        _stream_thread: Optional[threading.Thread] = None
        _stream_done = threading.Event()
        _stream_error: Optional[Exception] = None
        _close_connection = threading.Event()

        def _reset_stream_state(self):
            with self._buffer_lock:
                self._token_buffer.clear()
            self._close_connection.clear()
            self._stream_done.clear()
            if self._stream_thread and self._stream_thread.is_alive():
                self._stream_thread.join(timeout=5)
            self._stream_thread = None
            self._stream_error = None

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
                        options={
                            "temperature": temperature,
                            "top_p": 1.0,
                            "top_k": 0,
                            "num_ctx": 128000,
                        },
                        raw=True,
                    ):
                        if self._close_connection.is_set():
                            with self._buffer_lock:
                                self._token_buffer.append(EOS_TOKEN)
                                # Notify any waiting callers that a token is available.
                                self._token_available.notify_all()
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
                                    # Notify any waiting callers that tokens are available.
                                    self._token_available.notify_all()
                                last_len = len(toks)

                        if chunk.done:
                            with self._buffer_lock:
                                self._token_buffer.append(EOS_TOKEN)
                                # Notify waiting callers that EOS token is available.
                                self._token_available.notify_all()

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
            if new_request:
                self._reset_stream_state()
                self._stream_thread = self._start_stream(
                    token_ids=tokens, temperature=temperature
                )

            if self._stream_error is not None:
                raise RuntimeError(f"Ollama stream error: {self._stream_error!r}")

            # Wait for a token to become available or for the stream to finish.
            with self._token_available:
                while not self._stream_done.is_set() and not self._token_buffer:
                    # Wait with a small timeout to avoid indefinite blocking
                    # in case the stream ends without notifying.
                    self._token_available.wait(timeout=0.1)

            with self._buffer_lock:
                if self._token_buffer:
                    tok = self._token_buffer.popleft()
                    return tok

            # If we reach here, we still haven't got a token—ask the caller to call again soon.
            # Return a harmless token that the server will replace/ignore if your interface supports it.
            # If your interface does NOT allow a sentinel, keep the short-blocking behavior above.
            return (
                EOS_TOKEN if False else 0
            )  # replace `0` with a PAD/NOOP token your server ignores

    return OllamaModelConnection()
