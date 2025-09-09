"""
NOTE: this is not the most efficient way to use transformers. It's a simple implementation that infers
one token at a time to mimic the behavior of the Triton implementation.
"""

import os
from typing import Callable, List, Optional

import torch

# Transformers imports
from transformers import AutoModelForCausalLM

from openai_responses.api.types import ModelConnection

DEFAULT_TEMPERATURE = 1.0
TP = os.environ.get("TP", 2)


def load_model(checkpoint: str):
    """
    Serve the model directly with the Auto API.
    """

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    return model


def setup_model(checkpoint: str) -> Callable[[List[int], float, bool], int]:
    model = load_model(checkpoint)

    class TransformersConnection(ModelConnection):
        def infer_next_token(
            tokens: List[int],
            temperature: float = DEFAULT_TEMPERATURE,
            new_request: bool = False,  # kept for interface compatibility; unused here
            session_id: Optional[str] = None,
        ) -> int:
            tokens = torch.tensor([tokens], dtype=torch.int64, device=model.device)
            output = model.generate(
                tokens,
                max_new_tokens=1,
                do_sample=temperature != 0,
                temperature=temperature,
            )
            return output[0, -1].tolist()

    return TransformersConnection()
