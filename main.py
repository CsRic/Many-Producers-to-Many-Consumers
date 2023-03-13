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

# p = producer = exp maker, c = consumer = trainer
rank_to_role = ['p', 'c', 'p', 'c', 'p', 'p', 'c', 'p']

# expose operations suit for Torch RPC style
trainer: FakeTrainer = None
maker: FakeExperienceMaker = None
def get_buffer_length():
    global trainer
    if trainer:
        return trainer.get_buffer_length()
    return None

def put_experience(exp: FakeExperienceBuffer):
    global trainer
    trainer.put_experience(exp)
    
def run_worker(rank, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port
    world_size = args.world_size
    trainer_ranks = []
    maker_ranks = []
    for i in range(world_size):
        if rank_to_role[i] == 'c':
            trainer_ranks.append(i)
        elif rank_to_role[i] == 'p':
            maker_ranks.append(i)

    def run_maker(maker_time):
        global maker
        maker = FakeExperienceMaker(produce_time=maker_time, 
                                    data_shape=(4, 4), 
                                    device= f'cuda:{torch.cuda.current_device()}'
                                    )
        print(f"maker at {rank}: start making")
        for _ in range(100):
            exp = maker.make_experience()
            min_buffer_length = None
            chosen_rank = None
            # choose a target buffer (of a trainer) that lacks exp the most
            while chosen_rank is None:
                for i in trainer_ranks:
                    buffer_length = rpc.rpc_sync(f"worker{i}", get_buffer_length, args=None)
                    if not (buffer_length is None):
                        if (min_buffer_length is None) or (buffer_length < min_buffer_length):
                            min_buffer_length = buffer_length
                            chosen_rank = i
            rpc.remote(f"worker{chosen_rank}", put_experience, args=(exp,))
            print(f"                        maker at {rank}: make an exp to trainer at {chosen_rank}")

    def run_trainer(trainer_time):
        buffer = FakeExperienceBuffer(
            cpu_offload=True,
            )
        global trainer
        name = f"trainer at {rank}"
        trainer = FakeTrainer(buffer, 
                              name=name, 
                              train_time=trainer_time, 
                              max_epoch=10)
        print(f"{name}: start training")
        trainer.fit()

    # run
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=32, rpc_timeout=0)
    for i in range(args.world_size):
        if i == rank:
            continue
        options.set_device_map(f"worker{i}",{0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7})

    rpc.init_rpc(
        f"worker{rank}",
        rank=rank,
        world_size=args.world_size,
        rpc_backend_options=options
    )

    if rank_to_role[rank] == 'p':
        run_maker(args.maker_time)
    elif rank_to_role[rank] == 'c':
        run_trainer(args.trainer_time)
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