# 2022 03 25 Custom PyTorch Dataset for sklearn dataset

from sklearn.datasets import (
    load_iris,
    make_blobs,
    make_circles,
    make_moons,
    load_digits,
)

import torch
from torch.utils.data import Dataset, DataLoader


class MakeCircles(Dataset):
    def __init__(self, n_samples=300, noise=0.02):
        X, y = make_circles(n_samples=n_samples, random_state=0, noise=noise)
        # self.X = torch.from_numpy(np.concatenate((X, X), axis=1))  # For 2 features
        self.X = torch.from_numpy(X)  # For 2 features
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]


class MakeBlobs(Dataset):
    def __init__(self, n_samples=300, n_features=2, centers=4):
        X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers)
        # self.X = torch.from_numpy(np.concatenate((X, X), axis=1))  # For 2 features
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]


class Iris(Dataset):
    def __init__(self, id_list):
        data = load_iris()
        self.X = torch.from_numpy(data.data)
        max, _ = torch.max(self.X, dim=0)
        min, _ = torch.min(self.X, dim=0)
        # mean = self.X.mean(dim=0)
        # std = self.X.std(dim=0)
        self.X = (self.X - min) / (max - min)
        self.y = torch.from_numpy(data.target)
        self.id_list = id_list

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):

        return self.X[self.id_list[idx]], self.y[self.id_list[idx]]


class Digits(Dataset):
    def __init__(self, id_list, n_class=4):
        data = load_digits(n_class=n_class)
        self.X = torch.from_numpy(data.data)
        max, _ = torch.max(self.X, dim=0)
        min, _ = torch.min(self.X, dim=0)
        # self.X = (self.X - min) / (max - min)
        self.y = torch.from_numpy(data.target)
        self.id_list = id_list

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):

        return self.X[self.id_list[idx]], self.y[self.id_list[idx]]


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
