import os
import time

import torch


def check_devices_mem():
    devices_info = os.popen(
        '"/usr/bin/nvidia-smi"' + 
        ' --query-gpu=memory.total,memory.used' +
        ' --format=csv,nounits,noheader'
    ).read().strip().split("\n")
    divices_mem_info = [x.split(',') for x in devices_info]
    divices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if divices is None:
        return divices_mem_info
    else:
        device_list = []
        for i in [int(x) for x in divices.split(',')]:
            device_list.append(divices_mem_info[i])
        return device_list


def occupy_device_mem(cuda_device, mem_info, free=1024):
    total, used = int(mem_info[0]), int(mem_info[1])
    block_mem = total - used - free
    if block_mem > 0:
        print('Occupy device_{}\'s mem ...'.format(cuda_device))
        x = torch.zeros(
            (256, 1024, block_mem),
            dtype=torch.float32,
            device='cuda:{}'.format(cuda_device)
        )
        del x
        print('Occupy device_{}\'s mem finished'.format(cuda_device))
    else:
        print('Device_{}\'s out of memory'.format(cuda_device))


def occupy_gpus_mem(free=1536):
    for i, mem_info in enumerate(check_devices_mem()):
        occupy_device_mem(i, mem_info, free)
    print('Occupy all device\'s mem finished')


def occupy_gpu_mem_for_ddp(rank, free=1536):
    mem_info = check_devices_mem()[rank]
    occupy_device_mem(rank, mem_info, free)


def fake_train(shape=(1024, 1024, 1024)):
    gpu_num = torch.cuda.device_count()
    print('Start training')
    while True:
        for i in range(gpu_num):
            x = torch.randn(shape).cuda(i)
            y = torch.randn(shape).cuda(i)
            z = x * y
            time.sleep(1)
