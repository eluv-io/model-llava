import yaml
import os
from typing import Any

def load_config() -> Any:   
    path = os.getenv('CONFIG_PATH', 'config.yml')
    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    root_path = os.path.dirname(__file__)

    # If the path is not absolute, we assume it is relative to the root directory
    for p in config["storage"].keys():
        if not config["storage"][p].startswith('/'):
            config["storage"][p] = os.path.join(root_path, config["storage"][p])
        if not os.path.exists(config["storage"][p]):
            os.makedirs(config["storage"][p])

    return config

config = load_config()