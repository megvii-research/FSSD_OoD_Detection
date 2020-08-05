import os
import torch
import scipy
from scipy import special
import torchvision
import numpy as np
import torch.nn.functional as F

from torch import optim
# from .private_dataloader import ImageNet
from torchvision import transforms
from torch.utils.data import Dataset
from torch.nn import Conv2d, Linear, CrossEntropyLoss
from lib.model.dropout_resnet import ResNet34
from lib.model.dropout_lenet import LeNet
from torch.optim.lr_scheduler import StepLR

# from lib.utils.exp import get_transform, get_dataloader

INPUT_SIZE = {"cifar10": 32, "image-net": 224, "mnist": 28, "fmnist": 28, "ms1m": 112}
DEFAULT_TRAIN_CONFIG = {
    "cifar10": {
        "base_lr": 1e-1,
        "base_momentum": 0.9,
        "max_epochs": 350,
        "lr_adjust_epochs": [150, 250],
        "warmup_epochs": 0,
        "batch_size": 128,
    },
    "mnist": {"gamma": 0.7, "base_lr": 1.0, "max_epochs": 14, "batch_size": 64},
    "fmnist": {"gamma": 0.7, "base_lr": 1.0, "max_epochs": 14, "batch_size": 64},
    "image-net": {},
}

test_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)


# def construct_dataloader(dataset_name, batch_size=128, input_size=32):
#     train_transform = transforms.Compose(
#         [
#             transforms.RandomCrop(input_size, padding=4),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#         ]
#     )
#
#     if dataset_name == "image-net":
#         dataset = ImageNet(transform=train_transform)
#         train_loader = torch.utils.data.DataLoader(
#             dataset, batch_size=batch_size, shuffle=True
#         )
#         return train_loader, dataset.class_num
#     elif dataset_name == "cifar10":
#         dataset = torchvision.datasets.CIFAR10(
#             root="/data/cifar-10/", train=True, download=True, transform=train_transform
#         )
#         train_loader = torch.utils.data.DataLoader(
#             dataset, batch_size=batch_size, shuffle=True
#         )
#         return train_loader, len(set(dataset.targets))
#     elif dataset_name == "ms1m":
#         # TODO: fix ms1m
#         pass
#     elif dataset_name == "mnist":
#         transform = get_transform(dataset_name)
#         train_loader = get_dataloader(dataset_name, transform, "train")
#         return train_loader, 10
#     elif dataset_name == "fmnist":
#         transform = get_transform(dataset_name)
#         train_loader = get_dataloader(dataset_name, transform, "train")
#         return train_loader, 10


def separate_bn_paras(modules):
    decay_param = []
    no_decay_param = []

    for _name, _layer in modules.named_modules():
        if hasattr(_layer, "weight") and _layer.weight is not None:
            if isinstance(_layer, Conv2d) or isinstance(_layer, Linear):
                decay_param.append(_layer.weight)
            else:
                no_decay_param.append(_layer.weight)
        if hasattr(_layer, "bias") and _layer.bias is not None:
            no_decay_param.append(_layer.bias)

    return decay_param, no_decay_param


def save_model(model, model_path):
    state = {
        "backbone": model.module.state_dict()
        if hasattr(model, "module")
        else model.state_dict()
    }
    torch.save(state, model_path)


def load_resnext_model(backbone, head, pretrained_model_path):
    print("load model from path: {}".format(pretrained_model_path))
    checkpoint = torch.load(pretrained_model_path)
    backbone.load_state_dict(checkpoint["backbone"])
    head.load_state_dict(checkpoint["head"])
    print("model loaded successfully")


def load_model(model, pretrained_model_path):
    print("load model from path: {}".format(pretrained_model_path))
    checkpoint = torch.load(pretrained_model_path)
    model.load_state_dict(checkpoint["backbone"])
    print("model loaded successfully")


def train_lenet_model(model, data_loader, exp_name, batch_size=128, train_config=None):
    config_model_path = "/data/mc_dropout_exps/{}.pth".format(exp_name)
    if os.path.exists(config_model_path):
        load_model(model, config_model_path)
        model.train()
        return

    model = model.cuda()
    model = torch.nn.DataParallel(model)
    optimizer = optim.Adadelta(model.parameters(), lr=train_config["base_lr"])
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config["gamma"])

    model.train()
    total_loss = 0.0
    for epoch_ind in range(train_config["max_epochs"]):
        for step_ind, (imgs, labels) in enumerate(data_loader):
            imgs, labels = imgs.cuda(), labels.cuda()
            optimizer.zero_grad()
            logits = model(imgs)

            loss = F.nll_loss(logits, labels)
            predicted = logits.argmax(1)
            total = labels.size(0)
            correct = (predicted.long() == labels.long()).sum().item()
            acc = correct / total

            total_loss += loss * (imgs.size(0) / batch_size)

            loss.backward()
            optimizer.step()
            if step_ind % 100 == 0:
                print(
                    "epoch:{}, step: {}, loss={}, acc={}, avg_loss={}".format(
                        epoch_ind,
                        step_ind,
                        "{:.4f}".format(loss),
                        "{:.4f}".format(acc),
                        "{:.4f}".format(total_loss / imgs.size(0)),
                    )
                )
        scheduler.step()

    save_model(model, config_model_path)
    print("model saved to {}".format(config_model_path))


def train_dropout_model(
    model, data_loader, exp_name, batch_size=128, train_config=None
):

    assert train_config is not None, "training config should be defined first."
    if exp_name.startswith("mnist") or exp_name.startswith("fmnist"):
        train_lenet_model(
            model,
            data_loader,
            exp_name,
            batch_size=train_config.get("batch_size", 128),
            train_config=train_config,
        )
        return

    config_model_path = "/data/mc_dropout_exps/{}.pth".format(exp_name)
    if os.path.exists(config_model_path):
        load_model(model, config_model_path)
        return

    base_lr = 1e-1
    base_momentum = 0.9
    max_epochs = 350
    lr_adjust_epochs = [150, 250]
    warmup_epochs = 0

    # decay_param, no_decay_param = separate_bn_paras(model)
    #
    # optimizer = optim.SGD(
    #     [
    #         {"params": decay_param, "weight_decay": 5e-4, "lr": base_lr},
    #         {"params": no_decay_param, "lr": base_lr},
    #     ],
    #     momentum=base_momentum,
    # )

    optimizer = optim.SGD(
        model.parameters(), lr=base_lr, momentum=base_momentum, weight_decay=5e-4
    )

    model = model.cuda()
    model = torch.nn.DataParallel(model)

    ce_loss = CrossEntropyLoss()

    print("model loaded")

    def adjust_lr(global_epoch, cnt_lr):
        """linear warmup + linear decay"""
        if warmup_epochs > 0 and global_epoch <= warmup_epochs:
            rate = global_epoch / warmup_epochs
            return base_lr * rate
        else:
            if global_epoch in lr_adjust_epochs:
                cnt_lr = cnt_lr / 10
                return cnt_lr
            else:
                return cnt_lr

    def pytorch_run_minibatch(imgs, labels):
        imgs = imgs.float()
        optimizer.zero_grad()
        logits = model(imgs)
        predicted = logits.argmax(1)
        total = labels.size(0)
        correct = (predicted.long() == labels.long()).sum().item()
        acc = correct / total
        loss = ce_loss(logits, labels)  # cross entropy
        loss.backward()
        running_loss = loss.item()
        optimizer.step()

        return [running_loss, acc]

    total_loss = 0.0
    batch_num = 0
    lr = base_lr

    for epoch_ind in range(max_epochs):
        nxt_lr = adjust_lr(epoch_ind, lr)
        if nxt_lr != lr:
            for param_group in optimizer.param_groups:
                param_group["lr"] = nxt_lr
            lr = nxt_lr
        for step_ind, (imgs, labels) in enumerate(data_loader):
            imgs = imgs.cuda()
            labels = labels.cuda()
            [tloss, acc] = pytorch_run_minibatch(imgs, labels)
            total_loss += tloss * (imgs.size(0) / batch_size)
            batch_num += 1
            if step_ind % 100 == 0:
                print(
                    "epoch:{}, step: {}, loss={}, acc={}, avg_loss={}, lr={}".format(
                        epoch_ind,
                        step_ind,
                        "{:.4f}".format(tloss),
                        "{:.4f}".format(acc),
                        "{:.4f}".format(total_loss / batch_num),
                        "{:.4f}".format(lr),
                    )
                )

    save_model(model, config_model_path)
    print("model saved to {}".format(config_model_path))


def dropout_inference(model, imgs, labels=None, predictions_per_example=32):
    avg_probs_fn = lambda x: scipy.special.softmax(x, axis=-1).mean(-3)
    imgs = imgs.type(torch.FloatTensor).cuda()
    imgs.requires_grad = True
    imgs.grad = None

    feats = []
    for _ in range(predictions_per_example):
        if type(model) == dict:
            backbone_feats = model["backbone"](imgs.cuda())
            logits = model["head"](backbone_feats, istraining=False)
        else:
            logits = model(imgs.cuda())
        feats.append(logits.detach().cpu().data.numpy())
    feats = np.array(feats)
    mean = avg_probs_fn(feats)
    # calculate max p(y|x)
    mean = np.max(mean, axis=1)
    var = np.var(feats, axis=0)
    mean_logits = np.mean(feats, axis=0)
    predicted = np.argmax(mean_logits, axis=1)

    # mean = mean[np.arange(mean.shape[0]), predicted]
    var = var[np.arange(var.shape[0]), predicted]
    if labels is not None:
        try:
            labels = labels.numpy()
            total = labels.shape[0]
            correct = np.sum(predicted == labels)
            acc = correct / total
        except:
            acc = None
    else:
        acc = None
    return mean, var, acc



def get_model(dataset_name, dropout_rate):
    return
# def get_model(dataset_name, dropout_rate):

#     # model preparation
#     if dataset_name in {"ms1m", "celeba"}:
#         num_classes = {"ms1m": 64736, "celeba": 10122}
#         model_dict = dict()
#         from lib.inference.MCDropout.facerec_models import dropout_resnet as dresnet
#         from lib.inference.MCDropout.facerec_models import head as dhead

#         dresnet.act = dresnet.make_act("relu")
#         model_dict["backbone"] = dresnet.resnext50_32x4d(dropout_rate=dropout_rate)
#         model_dict["head"] = dhead.CosFaceHead(num_classes=num_classes[dataset_name])

#         return model_dict

#     elif dataset_name == "cifar10":
#         class_num = 10
#         model = ResNet34(
#             num_c=class_num,
#             method="dropout",
#             dropout_rate=dropout_rate,
#             target_dataset="cifar10",
#         )
#     elif dataset_name == "imagenet":
#         raise NotImplementedError

#     elif dataset_name == "mnist":
#         class_num = 10
#         model = LeNet(num_c=class_num, method="dropout", dropout_rate=dropout_rate)
#     elif dataset_name == "fmnist":
#         class_num = 10
#         model = LeNet(num_c=class_num, method="dropout", dropout_rate=dropout_rate)
#     elif dataset_name == "dogs100":
#         # from lib.model.dropout_resnet import ResNet34

#         model = ResNet34(
#             100, target_dataset="imagenet", dropout_rate=dropout_rate, method="dropout"
#         )
#     elif dataset_name == "dogs50A":
#         # from lib.model.dropout_resnet import ResNet34

#         model = ResNet34(
#             50, target_dataset="imagenet", dropout_rate=dropout_rate, method="dropout"
#         )
#     else:
#         raise NotImplementedError
#     return model


def search_mc_dropout_hyperparams(dataset_name,):
    assert dataset_name in ["ms1m", "cifar10", "imagenet", "mnist", "fmnist"]
    raise NotImplementedError
    # dropout_rate_list = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    # for dropout_rate in dropout_rate_list:
    #     data_loader, class_num = construct_dataloader(
    #         dataset_name,
    #         input_size=INPUT_SIZE.get(dataset_name),
    #         batch_size=DEFAULT_TRAIN_CONFIG[dataset_name].get("batch_size", 128),
    #     )
    #
    #     # model preparation
    #     if dataset_name == "ms1m":
    #         model = ResNet34(
    #             num_c=class_num,
    #             method="dropout",
    #             dropout_rate=dropout_rate,
    #             target_dataset="ms1m",
    #         )
    #     elif dataset_name == "cifar10":
    #         model = ResNet34(
    #             num_c=class_num,
    #             method="dropout",
    #             dropout_rate=dropout_rate,
    #             target_dataset="cifar10",
    #         )
    #     elif dataset_name == "imagenet":
    #         model = ResNet34(
    #             num_c=class_num,
    #             method="dropout",
    #             dropout_rate=dropout_rate,
    #             target_dataset="imagenet",
    #         )
    #     elif dataset_name == "mnist":
    #         model = LeNet(num_c=class_num, method="dropout", dropout_rate=dropout_rate)
    #     elif dataset_name == "fmnist":
    #         model = LeNet(num_c=class_num, method="dropout", dropout_rate=dropout_rate)
    #     else:
    #         raise NotImplementedError
    #
    #     # train model
    #     train_dropout_model(
    #         model,
    #         data_loader,
    #         exp_name="{}_{}".format(dataset_name, dropout_rate),
    #         train_config=DEFAULT_TRAIN_CONFIG[dataset_name],
    #     )


def get_mc_dropout_score(model, dataloader):
    from tqdm import tqdm

    if type(model) == dict:
        model["backbone"] = model["backbone"].cuda()
        model["head"] = model["head"].cuda()
        model["backbone"].train()
        model["head"].train()
    else:
        model = model.cuda()

    mean_scores = []
    var_scores = []
    mean_acc = []
    for data in tqdm(dataloader, desc="get_MC_dropout_score"):
        if type(data) in [tuple, list] and len(data) == 2:
            imgs, labels = data
        elif isinstance(data, torch.Tensor):
            imgs, labels = data, None
        else:
            print(type(data))
            raise NotImplementedError

        with torch.no_grad():
            mean, var, acc = dropout_inference(model, imgs, labels=labels)
            if mean is not None:
                mean_scores.append(mean)
            if var is not None:
                var_scores.append(var)
            if acc is not None:
                mean_acc.append(np.mean(acc))
    mean_scores = np.concatenate(mean_scores)
    var_scores = np.concatenate(var_scores)
    print("mean acc = {:.6f}".format(np.mean(mean_acc)))

    return mean_scores, var_scores, np.mean(mean_acc)
