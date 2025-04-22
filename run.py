import argparse
import os
import json
from typing import List
from common_ml.utils import nested_update
from common_ml.model import default_tag

from src.model import LLava
from config import config
import setproctitle

from multiprocessing import Process

def sub_run(file_paths: List[str], cfg: dict, model_name: str):
    cfg["model"] = model_name
    model = LLava(runtime_config=cfg)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tags')
    default_tag(model, file_paths, out_path)

def run(file_paths: List[str], runtime_config: str=None):
    if runtime_config is None:
        cfg = config["runtime"]["default"]
    else:
        cfg = json.loads(runtime_config)
        cfg = nested_update(config["runtime"]["default"], cfg)

    model_list = cfg.get("models", None)
    if model_list is None: model_list = [cfg["model"]]

    ## map model -> list of files
    sub_file_lists = { m : [] for m in model_list }

    ## distribute the work
    while len(file_paths) > 0:
        for model_name in model_list:
             if len(file_paths) == 0: break
             sub_file_lists[model_name].append(file_paths.pop())

    processes = []
    for model_name in model_list:
        p = Process(target=sub_run, args=(sub_file_lists[model_name], cfg, model_name))
        p.start()
        processes.append(p)

    errors = 0
    for process in processes:
        p.join()
        if p.exitcode != 0:
            errors = errors + 1

    if errors != 0:
        raise Exception(f"{errors} sub worker(s) got a nonzero exit code")

if __name__ == '__main__':
    setproctitle.setproctitle("model-llava")
    parser = argparse.ArgumentParser()
    parser.add_argument('file_paths', nargs='+', type=str)
    parser.add_argument('--config', type=str, required=False)
    args = parser.parse_args()
    run(args.file_paths, args.config)
