import os
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import TensorDataset
from torchvision import transforms
from torchvision.datasets import FashionMNIST

INPUT_SIGNAL_TYPES = ["body_acc_x", "body_acc_y", "body_acc_z", "total_acc_x",
                      "total_acc_y", "total_acc_z", "body_gyro_x", "body_gyro_y", "body_gyro_z"]


def load_X(X_signals_paths):
    X_signals = []
    for signal_type_path in X_signals_paths:
        # Read dataset from disk, dealing with text files' syntax
        with open(signal_type_path, 'r') as file:
            X_signals.append([np.array(serie, dtype=np.float32) for serie in [
                             row.replace('  ', ' ').strip().split(' ') for row in file]])
    return np.transpose(np.array(X_signals), (1, 2, 0))


def load_y(y_path):
    # Read dataset from disk, dealing with text file's syntax
    with open(y_path, 'r') as file:
        y_ = np.array([elem for elem in [row.replace(
            '  ', ' ').strip().split(' ') for row in file]], dtype=np.int32)
    # Substract 1 to each output class for friendly 0-based indexing
    return (y_ - 1).T[0]


class HARDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data/",
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def prepare_data(self):
        if os.path.exists(self.data_dir + "processed/train.npz") and \
                os.path.exists(self.data_dir + "processed/test.npz"):
            return

        os.mkdir(self.data_dir + "processed/")
        X_train_paths = [self.data_dir + "raw/train/Inertial Signals/" +
                         signal + "_train.txt" for signal in INPUT_SIGNAL_TYPES]
        X_test_paths = [self.data_dir + "raw/test/Inertial Signals/" +
                        signal + "_test.txt" for signal in INPUT_SIGNAL_TYPES]
        X_train, X_test = load_X(X_train_paths), load_X(X_test_paths)
        y_train_path = self.data_dir + "raw/train/y_train.txt"
        y_test_path = self.data_dir + "raw/test/y_test.txt"
        y_train, y_test = load_y(y_train_path), load_y(y_test_path)

        with open(self.data_dir + "processed/train.npz", "wb") as f:
            np.savez(f, X=X_train, y=y_train)
        with open(self.data_dir + "processed/test.npz", "wb") as f:
            np.savez(f, X=X_test, y=y_test)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            with open(self.data_dir + "processed/train.npz", "rb") as f:
                npzfile = np.load(f)
                X_full = torch.from_numpy(npzfile["X"]).float()
                y_full = torch.from_numpy(npzfile["y"]).long()
            full = TensorDataset(X_full, y_full)
            self.train, self.val = random_split(full, [7000, 352])
            self.dims = tuple(self.train[0][0].shape)
        if stage == "test" or stage is None:
            with open(self.data_dir + "processed/test.npz", "rb") as f:
                npzfile = np.load(f)
                X_test = torch.from_numpy(npzfile["X"]).float()
                y_test = torch.from_numpy(npzfile["y"]).int()
            self.test = TensorDataset(X_test, y_test)
            self.dims = tuple(self.test[0][0].shape)

    def train_dataloader(self):
        return DataLoader(
            self.train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
