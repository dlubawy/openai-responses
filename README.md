# OpenAI Responses

A rework of the reference implementation of OpenAI's Responses API server. Primarily reworks
the Ollama inference backend to utilize Ollama's Python SDK instead of a custom streaming
client which was found to be rather unstable.

## Installation

```nix
nix shell .
```
