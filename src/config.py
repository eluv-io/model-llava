from dataclasses import dataclass

@dataclass(frozen=True)
class LLavaRuntimeConfig:
    llama_endpoint: str
    fps: float
    model: str
    temperature: float
    prompt: str