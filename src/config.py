from dataclasses import dataclass

@dataclass(frozen=True)
class RuntimeConfig:
    llama_endpoint: str
    models: list[str]
    fps: int
    allow_single_frame: bool
    model: str
    temperature: float
    prompt: str