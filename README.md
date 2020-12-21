# GPUEater

Eat your GPUs

## 介绍

Pytorch使用GPU训练时只会使用所需要的显存，显存有大量空余情况下会给其他人带来可乘之机。多人共用GPU不仅会导致自己的程序运行变慢，还可能会在自己的程序运行中需要的显存变大时出现'Out of memory'错误。

Tensorflow在`allow_growth = False`时（也就是默认情况下）在程序运行前会自动分配所有显存，在pytorch代码中使用GPUEater可以在训练前提前分配所有显存，避免被人挤占GPU。

除此之外，单独运行本程序也可以用于**应急情况下**占用GPU（不建议这么做）。

建议配合[GPUTasker](https://github.com/cnstark/gputasker.git)使用。

## 使用方式

### 预分配显存

进入project目录，clone本项目

```shell
cd /path/to/your_project
git clone https://github.com/cnstark/gpueater.git
```

在训练开始前，设置`CUDA_VISIBLE_DEVICES`环境变量后加入

```python
# import your package
from gpu_eater import occupy_gpus_mem


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

    # your train code
```

### 单独使用

```shell
cd gpueater
python train.py --gpus 0
```
