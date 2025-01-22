from dataclasses import dataclass, asdict
from typing import Optional, List
from ollama import Client
import numpy as np
import PIL
import tempfile

from common_ml.types import Data
from common_ml.model import FrameModel, FrameTag

from config import config

@dataclass
class RuntimeConfig(Data):
    fps: int
    allow_single_frame: bool
    model: str

    @staticmethod
    def from_dict(data: dict) -> 'RuntimeConfig':
        return RuntimeConfig(**data)

class LLava(FrameModel):
    def __init__(self, runtime_config: dict):
        self.client = Client(config["llama_endpoint"])
        self.config = RuntimeConfig.from_dict(runtime_config)
        self.tmp = config["storage"]["tmp"]

    def get_config(self) -> dict:
        return asdict(self.config)
    
    def set_config(self, cfg: dict) -> None:
        self.config = RuntimeConfig.from_dict(cfg)

    def tag(self, img: np.ndarray) -> List[FrameTag]:
        # save the image to a file
        tmpfile = tempfile.NamedTemporaryFile(delete=True, dir=self.tmp)
        image_path = tmpfile.name + ".jpg"
        PIL.Image.fromarray(img).save(image_path)

        with open(image_path, "rb") as image_file:
            image_data = image_file.read()

        tmpfile.close()

        # Query the LLaVA model with the image and a prompt
        response = self.client.generate(
            model=self.config.model,
            prompt="Describe the contents of this image.",
            images=[image_data]
        )

        return [FrameTag.from_dict({"text": response["response"], "confidence": 1.0, "box": {"x1": 0.05, "y1": 0.05, "x2": 0.95, "y2": 0.95}})]
