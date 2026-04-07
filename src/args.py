from dataclasses import dataclass, field

@dataclass
class RuntimeArgs:
    llama_endpoint: str = "http://localhost:11434"
    models: list[str] = field(default_factory=lambda: ["llava:13b"])
    fps: float = 0.25
    temperature: float = 0.3
    prompt: str = "Describe the contents of this image."
    continue_on_error: bool = False