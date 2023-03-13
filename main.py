from typing import List
import torch
import torch.distributed.rpc as rpc
import os
import time
import torch.multiprocessing as mp
import argparse

from src.fake_experience_maker import FakeExperienceMaker
from src.fake_trainer import FakeTrainer
from src.fake_experience_buffer import FakeExperienceBuffer

# m = master, p = producer = exp maker, c = consumer = trainer
rank_to_role = ['m','p', 'c', 'c', 'c', 'p', 'p', 'c', 'p']

def call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)

def remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.rpc_sync(rref.owner(), call_method, args=args, kwargs=kwargs)

def run_master(world_size, maker_time = 5.0, trainer_time = 2.0):
    trainer_ranks = []
    for i in range(1, world_size):
        if rank_to_role[i] == 'c':
            trainer_ranks.append(i)
    maker_ranks = []
    for i in range(1, world_size):
        if rank_to_role[i] == 'p':
            maker_ranks.append(i)

    # step 1: initialize trainers. take their model rref as handles
    trainer_rrefs = []
    for i in trainer_ranks:
        trainer_rrefs.append(rpc.remote(f"worker{i}", add_trainer, args = (i, trainer_time)))

    # step 2: initialize exp makers with trainer handles (so makers can interact with trainers)
    for i in maker_ranks:
        rpc.rpc_async(f"worker{i}", run_producer, args = (i, trainer_rrefs,maker_time))

    # step 3: trainer start working
    for rref in trainer_rrefs:
        rpc.rpc_async(f"worker{i}", rref.remote().fit, args=None)

    # TODO: support hot update
    pass

def run_producer(rank, trainer_rrefs: List[rpc.RRef[FakeTrainer]], maker_time):
    maker = FakeExperienceMaker(produce_time=maker_time, data_shape=(4,4))
    
    # make another rref to the trainer
    # for rref in trainer_rrefs:
    #     rref = rpc.RRef(rref.remote())

    for _ in range(100):
    # try:
        exp = maker.make_experience()
        # choose a target buffer
        chosen_trainer = trainer_rrefs[0]
        # buffer_length = trainer_rrefs[0].remote().get_buffer_length().local_value()
        # for ref in trainer_rrefs[1:]:
        #     temp = ref.remote().get_buffer_length().local_value()
        #     if buffer_length > temp:
        #         chosen_trainer = ref
        #         buffer_length = temp
        print(f"                        maker at {rank}: make an exp to {chosen_trainer.owner_name()}")
        chosen_trainer.remote().put_experience(exp)
    # except:
    #     # trainer out
    #     print(f"maker at {rank} out")
    #     break

def add_trainer(rank, trainer_time):
    buffer = FakeExperienceBuffer()
    return FakeTrainer(buffer, name=f"trainer at {rank}", train_time=trainer_time)

def run_worker(rank, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=32, rpc_timeout=0)
    
    if rank_to_role[rank] == 'm':
        rpc.init_rpc(
            "master",
            rank=rank,
            world_size=args.world_size,
            rpc_backend_options=options
        )
        run_master(args.world_size, args.maker_time, args.trainer_time)
        pass
    else:
        for i in range(1, args.world_size):
            if i == rank:
                continue
            options.set_device_map(f"worker{i}",{0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7})
        rpc.init_rpc(
            f"worker{rank}",
            rank=rank,
            world_size=args.world_size,
            rpc_backend_options=options
        )

    rpc.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--world_size",
        type=int,
        default=3
    )
    parser.add_argument("--maker_time",
                        type=float,
                        default=5.0)
    parser.add_argument("--trainer_time",
                        type=float,
                        default=2.0)
    parser.add_argument("--port",
                        type=str,
                        default="29674")
    args = parser.parse_args()
    world_size = args.world_size
    mp.spawn(
        run_worker,
        args=[args],
        nprocs=world_size,
        join=True
    )