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

    class OlmSession:
        def __init__(self, session_id: str, model_name: str, client):
            self.session_id = session_id
            self.model_name = model_name
            self.client = client
            self._token_buffer: deque[int] = deque()
            self._buffer_lock = threading.Lock()
            self._token_available = threading.Condition(self._buffer_lock)
            self._stream_thread: Optional[threading.Thread] = None
            self._stream_done = threading.Event()
            self._stream_error: Optional[Exception] = None
            self._close_connection = threading.Event()

        def _reset_stream_state(self):
            with self._buffer_lock:
                self._token_buffer.clear()
            self._close_connection.clear()
            self._stream_done.clear()
            self._stream_thread = None
            self._stream_error = None

        def _start_stream(self, token_ids: list[int], temperature: float):
            prompt_text = encoding.decode(token_ids)

            def run():
                accum_text = []
                last_len = 0
                toks = None
                try:
                    for chunk in self.client.generate(
                        model=self.model_name,
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
                                self._token_available.notify_all()
                            break
                        if isinstance(chunk.response, str):
                            accum_text.append(chunk.response)
                            toks = encoding.encode(
                                "".join(accum_text), allowed_special="all"
                            )
                            if toks and len(toks) > last_len:
                                new_toks = toks[last_len:]
                                with self._buffer_lock:
                                    self._token_buffer.extend(new_toks)
                                    self._token_available.notify_all()
                                last_len = len(toks)
                        if chunk.done:
                            with self._buffer_lock:
                                self._token_buffer.append(EOS_TOKEN)
                                self._token_available.notify_all()
                    self._stream_done.set()
                except Exception as e:
                    self._stream_error = e
                    self._stream_done.set()

            t = threading.Thread(
                target=run, name=f"ollama-stream-{self.session_id}", daemon=True
            )
            t.start()
            self._stream_thread = t

        def close(self):
            while self._stream_thread and not self._stream_done:
                self._close_connection.set()
                self._stream_thread.join(timeout=0.5)
            self._reset_stream_state()

        def infer_next_token(
            self, tokens: list[int], temperature: float = 0.0, new_request: bool = False
        ) -> int:
            if new_request:
                self.close()
                self._start_stream(tokens, temperature)
            if self._stream_error is not None:
                raise RuntimeError(f"Ollama stream error: {self._stream_error!r}")
            with self._token_available:
                while not self._stream_done.is_set() and not self._token_buffer:
                    self._token_available.wait(timeout=0.1)
            with self._buffer_lock:
                if self._token_buffer:
                    return self._token_buffer.popleft()
            return EOS_TOKEN

    class OllamaModelConnection(ModelConnection):
        def __init__(self):
            self._sessions: dict[str, OlmSession] = {}
            self._sessions_lock = threading.Lock()

        def _get_or_create_session(self, session_id: str) -> OlmSession:
            with self._sessions_lock:
                if session_id not in self._sessions:
                    self._sessions[session_id] = OlmSession(
                        session_id, model_name, client
                    )
                return self._sessions[session_id]

        def close_session(self, session_id: str):
            with self._sessions_lock:
                sess = self._sessions.pop(session_id, None)
                if sess:
                    sess.close()

        def infer_next_token(
            self,
            tokens: list[int],
            temperature: float = 0.0,
            new_request: bool = False,
            session_id: Optional[str] = None,
        ) -> int:
            if session_id is None:
                raise ValueError("session_id required for OllamaModelConnection.")
            sess = self._get_or_create_session(session_id)
            return sess.infer_next_token(tokens, temperature, new_request)

        def close(self, session_id: Optional[str] = None):
            with self._sessions_lock:
                sess = self._sessions.pop(session_id, None)
                if sess:
                    sess.close()
                else:
                    for sess in list(self._sessions.values()):
                        sess.close()
                    self._sessions.clear()

    return OllamaModelConnection()
