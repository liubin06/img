import argparse
import os
import pandas
import sys
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import roc_auc_score
import random

import utils
from model import Model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Home device: {}'.format(device))

def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask



def criterion(out_1, out_2, batch_size, temperature, target):
    # neg score

    out = torch.cat([out_1, out_2], dim=0)
    target = torch.cat([target, target], dim=0)
    scores = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)


    posmask = torch.logical_xor(target.unsqueeze(-1) == target, torch.eye(2 * batch_size).bool().to(device)) # [2bs, 2bs] 与锚点标签相同，包含增强，不包含自身，定义为正例
    negmask = target.unsqueeze(-1) != target             #[2bs, 2bs] 与锚点标签不同，定义为负例

    neg = (scores * negmask).sum(dim=-1,keepdim=True) #[2bs, 1]  #1 分母只包含不同类别
    # neg = scores.sum(dim=-1, keepdim=True)  # [2bs, 1] #2 分母本包含所有样本
    loss = - torch.log(scores / (scores + neg)) * posmask #[2bs, 2bs]
    loss = loss.sum(dim=-1)/posmask.sum(dim=-1) #[2bs]
    return loss.sum()


def train(net, data_loader, train_optimizer, temperature):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for pos_1, pos_2, target in train_bar:
        pos_1, pos_2, target = pos_1.to(device, non_blocking=True), pos_2.to(device, non_blocking=True),target.to(device, non_blocking=True)
        feature_1, out_1 = net(pos_1)
        feature_2, out_2 = net(pos_2)

        loss = criterion(out_1, out_2,  batch_size, temperature,target)

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size

        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num



# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader,subclasses):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    c,k = len(subclasses),2
    with torch.no_grad():
        # generate feature bank
        for data, target in memory_data_loader:
            feature, out = net(data.cuda(non_blocking=True))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature, out = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1).long(), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :2] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@2:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100

def save_result(epoch,acc1,acc2):
    if not os.path.exists('../results'):
        os.makedirs('../results')
    acc = []
    acc.append([acc1,acc2])
    if epoch == epochs:
        np.savetxt('../results/{}acc.csv'.format(dataset_name), np.array(acc), delimiter=',', fmt='%.2f')

    if epoch % 1 == 0:
        torch.save(model.state_dict(),
                       '../results/{}/{}_{}_model_{}.pth'.format(dataset_name, temperature, batch_size, epoch))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--root', type=str, default='../data', help='Path to data directory')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--batch_size', default=128, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=400, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--dataset_name', default='self', type=str, help='Choose dataset')
    parser.add_argument('--classes', default=(0,1,2,3,4,5,6,7,8,9), type=tuple, help='Choose subset')


    # args parse
    args = parser.parse_args()
    feature_dim, dataset_name, epochs = args.feature_dim, args.dataset_name, args.epochs
    temperature, batch_size= args.temperature, args.batch_size



    # data prepare
    train_data, memory_data, test_data = utils.get_dataset(dataset_name,args.classes, root=args.root)
    print(args.classes)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,drop_last=True)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # model setup and optimizer config
    model = Model(feature_dim).to(device)
    model = nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

    # training loop

    acc = []
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, temperature)
        acc1, acc2 = test(model, memory_loader, test_loader,args.classes)
        save_result(epoch, acc1, acc2)



