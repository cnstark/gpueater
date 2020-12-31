import os
import time
import random
from argparse import ArgumentParser

import torch
import torchvision
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

from gpu_eater import occupy_gpu_mem_for_ddp


def init_process(rank, world_size, port):
    dist.init_process_group(
        'nccl',
        init_method='tcp://127.0.0.1:{}'.format(port),
        rank=rank,
        world_size=world_size
    )


def train(rank, world_size, port):
    init_process(rank, world_size, port)
    torch.cuda.set_device(rank)

    occupy_gpu_mem_for_ddp(rank)

    model = torchvision.models.resnet.resnet50()
    model.cuda()
    model = DDP(model, device_ids=[rank])
    l1 = nn.L1Loss()
    l1.cuda()
    ot = optim.SGD(model.parameters(), 0.0001)

    modelc = torchvision.models.resnet.resnet18()


    while True:
        b = model(torch.randn([10, 3, 256, 256]).cuda())
        c = b.detach() + 0.5
        time.sleep(random.randint(1, 5) / 2.0)
        loss = l1(b, c)
        ot.zero_grad()
        loss.backward()
        time.sleep(random.randint(1, 5) / 2.0)
        ot.step()
        if rank % 2 == 0:
            modelc(torch.randn([10, 3, 256, 256]))


if __name__=="__main__":
    # parse agrs
    parser = ArgumentParser(description='GPU Eater')
    parser.add_argument('--gpus', help='visible gpus', type=str)
    args = parser.parse_args()

    # set gpus
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    world_size = torch.cuda.device_count()
    nccl_port = random.randint(50000, 65000)

    mp.spawn(
        train,
        args=(world_size, nccl_port),
        nprocs=world_size,
        join=True
    )
