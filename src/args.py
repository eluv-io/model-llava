from dataclasses import dataclass

@dataclass
class RuntimeArgs:
    llama_endpoint: str
    models: list[str]
    fps: float
    model: str
    temperature: float
    prompt: str
    continue_on_error: bool