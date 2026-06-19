from dataclasses import dataclass

@dataclass(frozen=True)
class LLavaRuntimeConfig:
    llama_endpoint: str
    fps: float
    allow_single_frame: bool
    model: str
    temperature: float
    prompt: str