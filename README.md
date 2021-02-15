# Feature Space Singularity for Out-of-Distribution Detection


This repository is the implementation of the paper [Feature Space Singularity for Out-of-Distribution Detection](https://arxiv.org/abs/2011.14654). 


![Alt text](pics/fssd_img1.jpg?raw=true "Title")

## FSSD algorithm
The central idea of FSSD algorithm is to use the following FSSD score to tell OoD samples from in-distribution ones:

<!-- ```math
FSSD(x) := ||F_{\theta}(x) - F^\ast|| ,
``` -->
<img src="https://latex.codecogs.com/svg.latex?FSSD(x)&space;:=&space;||F_{\theta}(x)&space;-&space;F^\ast||&space;," title="FSSD(x) := ||F_{\theta}(x) - F^\ast|| ," />

where <img src="https://latex.codecogs.com/svg.latex?F_{\theta}" title="F_{\theta}" /> is the feature extractor and <img src="https://latex.codecogs.com/svg.latex?F^\ast" title="F^\ast" /> is the Feature Space Singularity (FSS). 
We approximate FSS simply by calculting the mean of uniform noise features: <img src="https://latex.codecogs.com/svg.latex?F^\ast&space;=&space;\frac{1}{n}&space;\sum_{i=1}^n&space;F_{\theta}(x_{noise_i})" title="F^\ast = \frac{1}{n} \sum_{i=1}^n F_{\theta}(x_{noise_i})" />.

## Implemented Algorithms
In this repository, we implement the following algorithms. 

| Algorithm | Paper | Implementation |
| --------- | ---------- | -------------- |
| FSSD      |[Feature Space Singularity for Out-of-Distribution Detection](https://arxiv.org/abs/2011.14654)  |   [test_fss.py](test_fss.py)  |
| Baseline  | [A BASELINE FOR DETECTING MISCLASSIFIED AND OUT-OF-DISTRIBUTION EXAMPLES IN NEURAL NETWORKS](https://arxiv.org/pdf/1610.02136.pdf) |  [test_baseline.py](test_baseline.py)   |
| ODIN      | [Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks](https://arxiv.org/abs/1706.02690) |  [test_odin.py](test_odin.py)  |
| Maha      |  [A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks](https://arxiv.org/abs/1807.03888)          |  [test_maha.py](test_maha.py)  |
| DE        | [Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles](https://arxiv.org/abs/1612.01474)  | [test_de.py](test_de.py) |
| OE        |[Deep Anomaly Detection with Outlier Exposure](https://arxiv.org/abs/1812.04606)  |  [test_baseline.py](test_baseline.py)  |

Note that OE shares the same implementation with Baseline. To test these methods, see the [Evaluation](#evaluation) section. We also welcome contributions to this repository for more SOTA OoD detection methods.



## Requirements
Python: 3.6+

To install required packages:

```setup
pip install -r requirements.txt
```

## Datasets
You can download FashionMNIST and CIFAR10 datasets directly from links offered in torchvision. For ImageNet-dogs-non-dogs dataset, please download the dataset from [this link](https://www.dropbox.com/sh/yrfmp7hwa2w9gxz/AAATMrfWNLctPq1vnRa3mtZPa?dl=0) and find the dataset description in this [issue](https://github.com/megvii-research/FSSD_OoD_Detection/issues/1).  (We will add more datasets and corresponding pre-trained models in the future.)

The default dataset root is `/data/datasets/`.  

## Training

To train the model(s) in the paper, see `lib/training/`. We provide the training codes of FashionMNIST and CIFAR10 datasets.


<!-- >📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters. -->

## Evaluation

To evaluate OoD detection performance of FSSD, run:

```eval
python test_fss.py --ind fmnist --ood mnist --model_arch lenet --inp_process
```

You can also evaluate OoD performance of other methods:

Baseline:
```eval
python test_baseline.py --ind fmnist --ood mnist --model_arch lenet
```

ODIN:
```eval
python test_odin.py --ind fmnist --ood mnist --model_arch lenet
```
Mahanalobis distance:
```eval
python test_maha.py --ind fmnist --ood mnist --model_arch lenet
```
Deep Ensemble:
```eval
python test_de.py --ind fmnist --ood mnist --model_arch lenet
```
Outlier Exposure:
```eval
python test_baseline.py --ind fmnist --ood mnist --model_arch lenet --test_oe
```

## Pre-trained Models

You can download pretrained models here:

- [Google cloud links](https://drive.google.com/drive/folders/1S-xv1xnvMrYFtCNomKT8w4k3bPSDwucx?usp=sharing) for models trained on FashionMNIST, CIFAR10 , ImageNet-dogs-non-dogs-dataset,using parameters specified in our supplements.  More pre-trained models will be added in the future.

Please download the pre-trained models and put them in `pre_trained` directory


## Results
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/feature-space-singularity-for-out-of/out-of-distribution-detection-on-fashion)](https://paperswithcode.com/sota/out-of-distribution-detection-on-fashion?p=feature-space-singularity-for-out-of)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/feature-space-singularity-for-out-of/out-of-distribution-detection-on-cifar-10)](https://paperswithcode.com/sota/out-of-distribution-detection-on-cifar-10?p=feature-space-singularity-for-out-of)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/feature-space-singularity-for-out-of/out-of-distribution-detection-on-imagenet)](https://paperswithcode.com/sota/out-of-distribution-detection-on-imagenet?p=feature-space-singularity-for-out-of)


Our model achieves the following performance on :

### [OoD detection benchmarks](https://paperswithcode.com/task/out-of-distribution-detection)


FashionMNIST vs. MNIST:

|       | Base | ODIN | Maha | DE   | MCD  | OE   | FSSD |
| ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| AUROC | 77.3 | 87.9 | 99.6 | 83.9 | 76.5 | 99.6 | 99.6 |
| AUPRC | 79.2 | 88.0 | 99.7 | 83.3 | 79.3 | 99.6 | 99.7 |
| FPR80 | 42.5 | 20.0 | 0.0  | 27.5 | 42.3 | 0.0  | 0.0  |
<!-- modified: ODIN, DE-->

CIFAR10 vs. SVHN:

|       | Base | ODIN | Maha | DE   | MCD  | OE   | FSSD |
| ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| AUROC | 89.9 | 96.6 | 99.1 | 96.0 | 96.7 | 90.4 | 99.5 |
| AUPRC | 85.4 | 96.7 | 98.1 | 93.9 | 93.9 | 89.8 | 99.5 |
| FPR80 | 10.1 | 4.8  | 0.3  | 1.2  | 2.4  | 12.5 | 0.4  |

<!-- modified: ODIN, DE -->


## Citation
If you find this repository helpful, please cite:
```
@inproceedings{huang2021feature,
  title={Feature Space Singularity for Out-of-Distribution Detection},
  author={Huang, Haiwen and Li, Zhihan and Wang, Lulu and Chen, Sishuo and Dong, Bin and Zhou, Xinyu},
  booktitle={Proceedings of the Workshop on Artificial Intelligence Safety 2021 (SafeAI 2021)},
  year={2021}
}
```

## Contributing

The usage and contribution to this repository is under MIT licence.

