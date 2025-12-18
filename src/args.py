from dataclasses import dataclass

@dataclass
class RuntimeArgs:
    llama_endpoint: str
    models: list[str] | None
    fps: int
    allow_single_frame: bool
    model: str
    temperature: float
    prompt: str