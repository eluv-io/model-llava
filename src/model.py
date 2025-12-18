from dataclasses import asdict
from ollama import Client
import numpy as np
from PIL import Image
import tempfile
from dacite import from_dict

from common_ml.model import FrameModel, FrameTag

from src.config import LLavaRuntimeConfig
from config import config

class LLava(FrameModel):
    def __init__(self, runtime_config: LLavaRuntimeConfig):
        self.tmp = config["storage"]["tmp"]
        self.config = runtime_config
        self.client = Client(self.config.llama_endpoint)

    def get_config(self) -> dict:
        return asdict(self.config)
    
    def set_config(self, config: dict) -> None:
        self.config = from_dict(data_class=LLavaRuntimeConfig, data=config)

    def tag(self, img: np.ndarray) -> list[FrameTag]:
        # save the image to a file
        tmpfile = tempfile.NamedTemporaryFile(delete=True, dir=self.tmp)
        image_path = tmpfile.name + ".jpg"
        Image.fromarray(img).save(image_path)

        with open(image_path, "rb") as image_file:
            image_data = image_file.read()

        tmpfile.close()

        # Query the LLaVA model with the image and a prompt
        response = self.client.generate(
            model=self.config.model,
            prompt=self.config.prompt,
            images=[image_data],
            options={"temperature": self.config.temperature}
        )

        return [FrameTag.from_dict({"text": response["response"], "confidence": 1.0, "box": {"x1": 0.05, "y1": 0.05, "x2": 0.95, "y2": 0.95}})]
