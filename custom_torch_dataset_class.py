# 2022 03 25 Custom PyTorch Dataset for sklearn dataset

from sklearn.datasets import make_circles
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons

import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np


class MakeCircles(Dataset):
    def __init__(self, n_samples=300, noise=0.02):
        X, y = make_circles(n_samples=n_samples, random_state=0, noise=noise)
        self.X = torch.from_numpy(np.concatenate((X, X), axis=1))  # For 2 features
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]


class MakeBlobs(Dataset):
    def __init__(self, n_samples=300, n_features=2, centers=4):
        X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers)
        self.X = torch.from_numpy(np.concatenate((X, X), axis=1))  # For 2 features
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]


if __name__ == "__main__":
    dataset = MakeCircles()
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    for data_batch, labels in data_loader:
        print("Batch of images has shape: ", data_batch.shape)
        print("Batch of labels has shape: ", labels.shape)

    dataset = MakeBlobs()
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    for data_batch, labels in data_loader:
        print("Batch of images has shape: ", data_batch.shape)
        print("Batch of labels has shape: ", labels.shape)
        print(data_batch)
        print(labels)
