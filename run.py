
import queue
from typing import Iterator, List
from dacite import from_dict
import setproctitle
from threading import Thread

from common_ml.tagging.producer import Message, TagMessageProducer
from common_ml.tagging.messages import Error
from common_ml.tagging.run_helpers import catch_errors, get_params, run_default

from src.model import LLava
from src.config import LLavaRuntimeConfig
from src.args import RuntimeArgs

class DistributedProducer(TagMessageProducer):
    def __init__(self, child_producers: list[TagMessageProducer]):
        self.child_producers = child_producers

    def produce(self, files: List[str]) -> Iterator[Message]:
        file_paths = files[:]
        
        batches = [[] for _ in self.child_producers]

        ## distribute the work
        while len(file_paths) > 0:
            for batch in batches:
                if len(file_paths) == 0: 
                    break
                batch.append(file_paths.pop())

        # each sub-producer writes to the queue
        out = queue.Queue()
        threads = []
        for i, producer in enumerate(self.child_producers):
            if len(batches[i]) == 0:
                continue
            t = Thread(target=self._run_producer, args=(producer, batches[i], out))
            t.start()
            threads.append(t)

        while any(t.is_alive() for t in threads):
            try:
                msg = out.get(timeout=1)
                yield msg
            except queue.Empty:
                pass

        for t in threads:
            t.join()

        # yield any remaining messages
        while True:
            try:
                yield out.get_nowait()
            except queue.Empty:
                break

    def _run_producer(self, producer: TagMessageProducer, files: list[str], output: queue.Queue):
        try:
            for msg in producer.produce(files):
                output.put(msg)
        except Exception as e:
            output.put(Error(message=str(e)))


if __name__ == '__main__':
    setproctitle.setproctitle("model-llava")
    catch_errors()

    params = get_params()

    params = from_dict(RuntimeArgs, params)

    model_list = params.models

    producers = []
    for model in model_list:
        args = LLavaRuntimeConfig(
            llama_endpoint=params.llama_endpoint,
            fps=params.fps,
            model=model,
            temperature=params.temperature,
            prompt=params.prompt,
        )
        fm = LLava(runtime_config=args)
        producers.append(TagMessageProducer.from_model(fm, fps=args.fps))

    distributed_producer = DistributedProducer(producers)
    run_default(distributed_producer)