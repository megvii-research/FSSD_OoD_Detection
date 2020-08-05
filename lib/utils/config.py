import os
import sys

shift_types = [
    "brightness",
    "contrast",
    "defocus_blur",
    "elastic_transform",
    "fog",
    "frost",
    "gaussian_blur",
    "gaussian_noise",
    "glass_blur",
    "impulse_noise",
    "jpeg_compression",
    "motion_blur",
     "pixelate",
     "saturate",
     "shot_noise",
     "snow", 
     "spatter",
     "speckle_noise",
     "zoom_blur",
]

cifar10_shift_root = "s3://yejinxing-ood-bmks/repos/cifar10-val/cifar10_test"
cifar10_shift_paths = {}
for t in shift_types:
    for intensity in [1,2,3,4,5]:
        _path = cifar10_shift_root + "_" + t + "_" + str(intensity)
        d = cifar10_shift_paths.get(t, {})
        d[intensity] = _path
        cifar10_shift_paths[t] = d


imagenet_shift_root = "s3://yejinxing-ood-bmks/repos/image-net/imagenet"
imagenet_shift_paths = {}
for t in shift_types:
    for intensity in [1,2,3,4,5]:
        _path = imagenet_shift_root + "_" + t + "_" + str(intensity)
        d = imagenet_shift_paths.get(t, {})
        d[intensity] = _path
        imagenet_shift_paths[t] = d

dogs50B_shift_root = "s3://yejinxing-ood-bmks/repos/dogs50B-val/dogs50B"
dogs50B_shift_paths = {}
for t in shift_types:
    for intensity in [1,2,3,4,5]:
        _path = dogs50B_shift_root + "_" + t + "_" + str(intensity)
        d = dogs50B_shift_paths.get(t, {})
        d[intensity] = _path
        dogs50B_shift_paths[t] = d
