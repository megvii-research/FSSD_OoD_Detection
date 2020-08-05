import os
import sys
import pickle

from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from lib.metric import get_metrics, train_lr

def sample_estimator(model, num_classes, feature_dim_list, train_loader):
    """
    compute sample mean and precision (inverse of covariance)
    return: sample_class_mean: list of class mean
             precision: list of precisions
    """
    import sklearn.covariance
    model.eval()
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    num_output = len(feature_dim_list)
    num_sample_per_class = np.empty(num_classes)
    num_sample_per_class.fill(0)
    list_features = []
    for i in range(num_output):
        temp_list = []
        for j in range(num_classes):
            temp_list.append(0)
        list_features.append(temp_list)

    correct, total = 0, 0
    for data, target in tqdm(train_loader, desc="sample_estimator"):
        # if total > 100:
        #     break
        total += data.size(0)
        data = data.cuda()
        with torch.no_grad():
            try:
                output, out_features = model.nonflat_feature_list(data)
            except:
                output, out_features = model.module.nonflat_feature_list(data)
            # get hidden features
            for i in range(num_output):
                out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
                out_features[i] = torch.mean(out_features[i].data, 2)
            
        # compute the accuracy
        pred = output.data.max(1)[1]
        equal_flag = pred.eq(target.cuda()).cpu()
        correct += equal_flag.sum()
        
        # construct the sample matrix
        for i in range(data.size(0)):
            label = target[i]
            if num_sample_per_class[label] == 0:
                out_count = 0
                for out in out_features:
                    # list_features[out_count][label] = out[i].view(1, -1)
                    list_features[out_count][label] = out[i].view(1, -1).cpu()
                    out_count += 1
            else:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] = torch.cat((list_features[out_count][label], out[i].view(1, -1).cpu()), 0)
                    out_count += 1                
            num_sample_per_class[label] += 1
            
    sample_class_mean = []
    out_count = 0
    for num_feature in tqdm(feature_dim_list, desc="feature_dim_list"):
        # temp_list = torch.Tensor(num_classes, int(num_feature)).cuda()
        temp_list = torch.Tensor(num_classes, int(num_feature))
        for j in range(num_classes):
            try:
                temp_list[j] = torch.mean(list_features[out_count][j], 0)
            except Exception:
                from IPython import embed
                embed()
        sample_class_mean.append(temp_list)
        out_count += 1
        
    precision = []
    for k in tqdm(range(num_output), desc="range(num_output)"):
        # X = 0 
        # for i in tqdm(range(num_classes), desc="range(num_classes)"):
        #     if i == 0:
        #         X = list_features[k][i] - sample_class_mean[k][i]
        #     else:
        #         X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)
        X = []
        for i in tqdm(range(num_classes), desc="range(num_classes)"):
            X.append(list_features[k][i] - sample_class_mean[k][i])
        X = torch.cat(X, 0)

        # find inverse            
        group_lasso.fit(X.numpy())
        temp_precision = group_lasso.precision_
        temp_precision = torch.from_numpy(temp_precision).float().cuda()
        precision.append(temp_precision)

    sample_class_mean = [t.cuda() for t in sample_class_mean]
    print('\n Training Accuracy:({:.2f}%)\n'.format(100. * correct / total))
    return sample_class_mean, precision


def get_Mahalanobis_score(model, dataloader, num_classes, sample_mean, precision, layer_index, magnitude, std):
    model.eval()
    model = model.cuda()

    scores = []
    for data in tqdm(dataloader, desc=f"get_Mahalanobis_score for layer {layer_index}"):
        if type(data) in [tuple, list] and len(data) == 2:
            imgs, _ = data
        elif isinstance(data, torch.Tensor):
            imgs = data
        else:
            print(type(data))
            raise NotImplementedError

        imgs = imgs.type(torch.FloatTensor).cuda()
        imgs.requires_grad = True
        imgs.grad = None
        model.zero_grad()

        try:
            feat = model.intermediate_forward(imgs, layer_index)
        except:
            feat = model.module.intermediate_forward(imgs, layer_index)
        n,c = feat.shape[:2]
        feat = feat.view(n,c,-1)
        feat = torch.mean(feat, 2)

        # compute Mahalanobis score
        gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = feat.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                gaussian_score = term_gau.view(-1,1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1)

        # Input_processing
        sample_pred = gaussian_score.max(1)[1]
        batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
        zero_f = feat - batch_sample_mean
        pure_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
        loss = torch.mean(-pure_gau)
        loss.backward()

        gradient =  torch.ge(imgs.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        if len(std) ==3:
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / std[0])
            gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / std[1])
            gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / std[2])
        elif len(std) == 1:
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / std[0])


        tempInputs = torch.add(imgs.data, -magnitude, gradient)
        with torch.no_grad():
            try:
                noise_out_features = model.intermediate_forward(tempInputs, layer_index)
            except:
                noise_out_features = model.module.intermediate_forward(tempInputs, layer_index)
            noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
            noise_out_features = torch.mean(noise_out_features, 2)
        noise_gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = noise_out_features.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                noise_gaussian_score = term_gau.view(-1,1)
            else:
                noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1,1)), 1)      

        noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
        scores.extend(noise_gaussian_score.cpu().numpy())
    return scores

def get_Mahalanobis_score_ensemble(model, dataloader, layer_indexs, num_classes, sample_mean, precision, magnitude, std=[255,255,255]):
    scores_list = []
    for layer_id in layer_indexs:
        scores = get_Mahalanobis_score(model, dataloader, num_classes, sample_mean, precision, layer_id, magnitude, std)
        scores = np.array(scores).reshape(-1,1)
        scores_list.append(scores)
    return np.concatenate(scores_list, axis=1)


def search_Mahalanobis_hyperparams(model, 
                            sample_mean, 
                            precision,
                            layer_indexs,
                            num_classes,
                            ind_dataloader_val_for_train, 
                            ood_dataloader_val_for_train, 
                            ind_dataloader_val_for_test, 
                            ood_dataloader_val_for_test, 
                            std):
    # magnitude_list = [0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005]
    magnitude_list = [0.02, 0.015, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.0014, 0.001, 0.0005]
    best_magnitude = None
    best_tnr = 0
    for m in tqdm(magnitude_list, desc="magnitude"):
        ind_features_val_for_train = get_Mahalanobis_score_ensemble(model, ind_dataloader_val_for_train, layer_indexs, num_classes, sample_mean, precision, m, std)
        ood_features_val_for_train = get_Mahalanobis_score_ensemble(model, ood_dataloader_val_for_train, layer_indexs, num_classes, sample_mean, precision, m, std)

        ind_features_val_for_test = get_Mahalanobis_score_ensemble(model, ind_dataloader_val_for_test, layer_indexs, num_classes, sample_mean, precision, m, std)
        ood_features_val_for_test = get_Mahalanobis_score_ensemble(model, ood_dataloader_val_for_test, layer_indexs, num_classes, sample_mean, precision, m, std)
        
        lr = train_lr(ind_features_val_for_train, ood_features_val_for_train)
        
        metrics = get_metrics(lr, ind_features_val_for_test, ood_features_val_for_test, acc_type="best")
        print("m:{}, metrics:{}".format(m, metrics))
        if metrics['TNR@tpr=0.95'] > best_tnr:
            best_tnr = metrics['TNR@tpr=0.95']
            best_magnitude = m
    return best_magnitude
