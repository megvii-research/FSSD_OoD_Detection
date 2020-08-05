# Train CIFAR10 for OoD detection
The code is modified from https://github.com/kuangliu/pytorch-cifar .



```bash
python main.py
# CUDA_VISIBLE_DEVICES=2 python main.py  # to specify GPU id to ex. 2
```

Resume the training with `python main.py --resume --lr=0.01`
