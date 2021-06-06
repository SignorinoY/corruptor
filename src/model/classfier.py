from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn


class HARBaseClassfier(pl.LightningModule):

    classes = (
        "Walking",
        "Upstairs",
        "Downstairs",
        "Sitting",
        "Standing",
        "Laying"
    )

    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.zeros(1, 128, 9)
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.train_acc(y_hat, y)
        self.log_dict({"train_loss": loss, "train_acc": self.train_acc})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.val_acc(y_hat, y)
        self.log_dict({"val_loss": loss, "val_acc": self.val_acc})

    def test_step(self, batch, batch_idx):
        x, _ = batch
        y_hat = self(x)
        return y_hat

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", type=float, default=0.0001)
        return parser


class HARLSTMClassfier(HARBaseClassfier):

    def __init__(self, learning_rate):
        super().__init__(learning_rate=learning_rate)

        self.lstm = nn.LSTM(9, 32, 2)
        self.fc = nn.Linear(32, 6)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        x = F.softmax(x, dim=1)
        return x


class HARBiLSTMClassfier(HARBaseClassfier):

    def __init__(self, learning_rate):
        super().__init__(learning_rate=learning_rate)

        self.lstm = nn.LSTM(9, 16, 2, bidirectional=True)
        self.fc = nn.Linear(32, 6)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        x = F.softmax(x, dim=1)
        return x