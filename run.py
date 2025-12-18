import argparse
import os
import json
from dacite import from_dict
import setproctitle
from multiprocessing import Process

from common_ml.utils import nested_update
from common_ml.model import default_tag, run_live_mode

from src.model import LLava
from src.config import LLavaRuntimeConfig
from src.args import RuntimeArgs
from config import config

def sub_run(file_paths: list[str], cfg: RuntimeArgs, model_name: str):
    run_cfg = LLavaRuntimeConfig(
        llama_endpoint=cfg.llama_endpoint,
        fps=cfg.fps,
        allow_single_frame=cfg.allow_single_frame,
        model=model_name,
        temperature=cfg.temperature,
        prompt=cfg.prompt,
    )
    model = LLava(runtime_config=run_cfg)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tags')
    default_tag(model, file_paths, out_path)

def get_runtime_config(runtime_config: str | None) -> RuntimeArgs:
    """Get the runtime configuration, merging with defaults if provided"""
    ipt_cfg = {}
    if runtime_config is not None:
        ipt_cfg = json.loads(runtime_config)

    updated_cfg = nested_update(config["runtime"]["default"], ipt_cfg)

    runtime_cfg = from_dict(RuntimeArgs, updated_cfg)

    # fix models field
    if runtime_cfg.models is None:
        runtime_cfg.models = [runtime_cfg.model]

    return runtime_cfg

def run(file_paths: list[str], cfg: RuntimeArgs):
    model_list = cfg.models
    assert model_list is not None and len(model_list) > 0

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
    for p in processes:
        p.join()
        if p.exitcode != 0:
            errors = errors + 1

    if errors != 0:
        raise Exception(f"{errors} sub worker(s) got a nonzero exit code")

def get_tag_fn(runtime_config: str | None):
    """Create a tag function with the specified configuration"""
    cfg = get_runtime_config(runtime_config)
    
    def tag_fn(file_paths: list[str]):
        run(file_paths, cfg)
    
    return tag_fn

if __name__ == '__main__':
    setproctitle.setproctitle("model-llava")
    parser = argparse.ArgumentParser()
    parser.add_argument('file_paths', nargs='*', type=str, help='Input file paths', default=[])
    parser.add_argument('--config', type=str, required=False, help='Runtime configuration JSON')
    parser.add_argument('--live', action='store_true', help='Run in live mode (read files from stdin)')
    args = parser.parse_args()

    tag_fn = get_tag_fn(args.config)
    
    if args.live:
        run_live_mode(tag_fn)
    else:
        tag_fn(args.file_paths)