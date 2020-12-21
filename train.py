import os
import time
from argparse import ArgumentParser

from gpu_eater import occupy_gpus_mem, fake_train


if __name__ == '__main__':
    # parse agrs
    parser = ArgumentParser(description='GPU Eater')
    parser.add_argument('--gpus', help='visible gpus', type=str)
    args = parser.parse_args()

    # set gpus
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # occury gpus mem
    occupy_gpus_mem()

    # fake train
    fake_train()
