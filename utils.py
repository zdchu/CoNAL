import numpy as np
from torch.utils import data
import torch
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split
import IPython
import os
from torchvision.datasets.utils import download_url, check_integrity
import torchvision.transforms as transforms
import sys
import pandas as pd
from PIL import Image
import pickle
import torchvision.models as models
from sklearn.preprocessing import normalize


def map_data(data):
    uniq = list(set(data))

    id_dict = {old: new for new, old in enumerate(sorted(uniq))}
    data = np.array(list(map(lambda x: id_dict[x], data)))
    n = len(uniq)

    return data, id_dict, n


def one_hot(target, n_classes):
    targets = np.array([target]).reshape(-1)
    one_hot_targets = np.eye(n_classes)[targets]
    return one_hot_targets

def transform_onehot(answers, N_ANNOT, N_CLASSES, empty=-1):
    answers_bin_missings = []
    for i in range(len(answers)):
        row = []
        for r in range(N_ANNOT):
            if answers[i, r] == -1:
                row.append(empty * np.ones(N_CLASSES))
            else:
                row.append(one_hot(answers[i, r], N_CLASSES)[0, :])
        answers_bin_missings.append(row)
    answers_bin_missings = np.array(answers_bin_missings).swapaxes(1, 2)
    return answers_bin_missings


class Dataset(data.Dataset):
    def __init__(self, mode='train', k = 0, dataset='labelme', sparsity=0):
        if mode[:5] == 'train':
            self.mode = mode[:5]
        else:
            self.mode = mode
        if dataset == 'labelme':
            data_path = './data/labelme/'
            X = np.load(data_path + self.mode + '/data_%s_vgg16.npy' % self.mode)
            y = np.load(data_path + self.mode + '/labels_%s.npy' % self.mode)
            X = X.reshape(X.shape[0], -1)
            if mode == 'train':
                answers = np.load(data_path + self.mode + '/answers.npy')
                self.answers = answers
                self.num_users = answers.shape[1]
                classes = np.unique(answers)
                if -1 in classes:
                    self.num_classes = len(classes) - 1
                else:
                    self.num_classes = len(classes)
                self.input_dims = X.shape[1]
                self.answers_onehot = transform_onehot(answers, answers.shape[1], 8)

                y = np.load(data_path + self.mode + '/labels_%s.npy' % self.mode)
            elif mode == 'train_dmi':
                answers = np.load(data_path + self.mode + '/answers.npy')
                self.answers = transform_onehot(answers, answers.shape[1], 8)
                self.num_users = answers.shape[1]
                classes = np.unique(answers)
                if -1 in classes:
                    self.num_classes = len(classes) - 1
                else:
                    self.num_classes = len(classes)
                self.input_dims = X.shape[1]
        if dataset == 'music':
            data_path = './data/music/'
            X = np.load(data_path + 'data_%s.npy' % self.mode)
            y = np.load(data_path + 'labels_%s.npy' % self.mode).astype(np.int)
            if mode == 'train':
                answers = np.load(data_path + '/answers.npy')
                self.answers = answers
                self.num_users = answers.shape[1]
                classes = np.unique(answers)
                if -1 in classes:
                    self.num_classes = len(classes) - 1
                else:
                    self.num_classes = len(classes)
                self.input_dims = X.shape[1]
                self.answers_onehot = transform_onehot(answers, answers.shape[1], self.num_classes)
        elif dataset == 'synthetic':
            data_path = './data/synthetic/'
            X = np.load(data_path + 'item_features_%s.npy' % self.mode).astype(np.float)
            y = np.load(data_path + 'labels_%s.npy' % self.mode)
            if mode == 'train':
                answers = np.load(data_path + 'answers.npy')
                self.num_users = answers.shape[1]
                num_users = self.num_users
                if -1 in np.unique(answers):
                    self.num_classes = len(np.unique(answers)) - 1
                else:
                    self.num_classes = len(np.unique(answers))
                self.input_dims = X.shape[1]
                for v_idx, v_res in enumerate(answers):
                    np.random.seed(v_idx)
                    test_idx = np.random.choice(num_users, int(sparsity * num_users), replace=False)
                    answers[v_idx, test_idx] = -1
                self.answers = answers.astype(np.int)
                self.answers_onehot = transform_onehot(self.answers, answers.shape[1], self.num_classes)

        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y)


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.mode == 'train':
            return idx, self.X[idx], self.answers[idx], self.answers_onehot[idx], self.y[idx]
        else:
            return idx, self.X[idx], self.y[idx]


def simple_majority_voting(response, empty=-1):
    mv = []
    for row in response:
        bincount = np.bincount(row[row != empty])
        mv.append(np.argmax(bincount))
    return np.array(mv)
