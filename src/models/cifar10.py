from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import torch
from torch import nn as nn, Tensor
from torch.nn import functional as F

from utilities.utils import to_numpy


class TaskBasedNets(ABC, nn.Module):
    @abstractmethod
    def forward(self, features: Tensor, task_id: str):
        pass

    def block_forward(self, features: Tensor, task_id: str) -> Dict[str, torch.Tensor]:
        blocks_recorded = dict()
        for key, operations in self.blocks.items():
            features = operations(features)
            blocks_recorded[key] = torch.flatten(features, 1).detach()
        return blocks_recorded

    def predict(self, features: Tensor, task_id: str) -> np.ndarray:
        outputs = self.forward(features=features, task_id=task_id)
        _, predictions = torch.max(outputs, 1)
        return to_numpy(predictions)


class TinyNet(TaskBasedNets):
    def __init__(self):
        super().__init__()
        self.fc_task1 = nn.Linear(84, 5)
        self.fc_task2 = nn.Linear(84, 5)
        self.blocks = nn.ModuleDict(
            {
                "block0": nn.Sequential(nn.Conv2d(3, 6, 5), nn.ReLU(), nn.MaxPool2d(2, 2)),  # out: 6*14*14
                "block1": nn.Sequential(nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2, 2)),  # out: 16*5*5
                "block2": nn.Sequential(nn.Flatten(), nn.Linear(16 * 5 * 5, 120), nn.ReLU()),
                "block3": nn.Sequential(nn.Linear(120, 84), nn.ReLU()),
            }
        )
        self.block_output_size = {"block0": 6 * 14 * 14, "block1": 16 * 5 * 5, "block2": 120, "block3": 84}

    def forward(self, features: Tensor, task_id: str):
        for operations in self.blocks.values():
            features = operations(features)
        # This is the prediction layer
        if task_id == "1":
            features = self.fc_task1(features)
        else:
            features = self.fc_task2(features)
        return features


class VGG(TaskBasedNets):
    def __init__(self):
        super().__init__()
        self.fc_task1 = nn.Linear(1024, 5)
        self.fc_task2 = nn.Linear(1024, 5)
        self.fc1 = nn.Linear(128, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.blocks = nn.ModuleDict(
            {
                "block0": nn.Sequential(
                    nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                ),  # out: 16*32*32
                "block1": nn.Sequential(
                    nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                ),  # out: 16*16*16
                "block2": nn.Sequential(
                    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                ),  # out: 32*16*16
                "block3": nn.Sequential(
                    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                ),  # out: 32*8*8
                "block4": nn.Sequential(
                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                ),  # out: 64*8*8
                "block5": nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                ),  # out: 64*4*4
                "block6": nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                ),  # out: 128*2*2
                "block7": nn.Sequential(
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                ),  # out: 128*1*1
            }
        )
        self.block_output_size = {
            "block0": 16 * 32 * 32,
            "block1": 16 * 16 * 16,
            "block2": 32 * 16 * 16,
            "block3": 32 * 8 * 8,
            "block4": 64 * 8 * 8,
            "block5": 64 * 4 * 4,
            "block6": 128 * 2 * 2,
            "block7": 128 * 1 * 1,
        }

    def forward(self, features: Tensor, task_id: str):
        for operations in self.blocks.values():
            features = operations(features)
        # This is the prediction layer
        features = torch.flatten(features, 1)
        features = nn.ReLU()(self.fc1(features))
        features = nn.ReLU()(self.fc2(features))
        if task_id == "1":
            features = self.fc_task1(features)
        else:
            features = self.fc_task2(features)
        return features


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )  # downsample with first conv

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module(
                "conv",
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),  # downsample
            )
            self.shortcut.add_module("bn", nn.BatchNorm2d(out_channels))  # BN

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        y += self.shortcut(x)
        y = F.relu(y, inplace=True)  # apply ReLU after addition
        return y


class ResNet(TaskBasedNets):
    def __init__(self):
        super().__init__()

        initial_channels = 32
        n_stages = 4
        n_blocks_per_stage = 2
        n_channels = [initial_channels * 2 ** i for i in range(n_stages)]

        self.stage0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=initial_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=initial_channels),
            nn.ReLU(inplace=True),
        )  # out: 32*32*32
        self.stage1 = self._make_stage(n_channels[0], n_channels[0], n_blocks_per_stage, ResidualBlock, stride=1)
        self.stage2 = self._make_stage(n_channels[0], n_channels[1], n_blocks_per_stage, ResidualBlock, stride=2)
        self.stage3 = self._make_stage(n_channels[1], n_channels[2], n_blocks_per_stage, ResidualBlock, stride=2)

        self.blocks = nn.ModuleDict({"block0": self.stage0})

        self.blocks.update(
            nn.ModuleDict(
                {
                    f"block{stage_id * 2 + residual_block_id + 1}": residual_block
                    for stage_id, stage in enumerate([self.stage1, self.stage2, self.stage3])
                    for residual_block_id, residual_block in enumerate([stage.block1, stage.block2])
                }
            )
        )

        self.block_output_size = {
            "block0": 32 * 32 * 32,
            "block1": 32 * 32 * 32,
            "block2": 32 * 32 * 32,
            "block3": 64 * 16 * 16,
            "block4": 64 * 16 * 16,
            "block5": 128 * 8 * 8,
            "block6": 128 * 8 * 8,
        }

        # compute conv feature size
        with torch.no_grad():
            dummy_data = torch.zeros(
                (1, 3, 32, 32),
                dtype=torch.float32,
            )
            self.feature_size = self._forward_conv(dummy_data).view(-1).shape[0]

        self.fc_task1 = nn.Linear(self.feature_size, 5)
        self.fc_task2 = nn.Linear(self.feature_size, 5)

    def _make_stage(self, in_channels, out_channels, n_blocks, block, stride):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_name = f"block{index + 1}"
            if index == 0:
                stage.add_module(block_name, block(in_channels, out_channels, stride=stride))
            else:
                stage.add_module(block_name, block(out_channels, out_channels, stride=1))
        return stage

    def _forward_conv(self, features: Tensor):
        for operations in self.blocks.values():
            features = operations(features)
        features = F.adaptive_avg_pool2d(features, output_size=1)
        return features

    def forward(self, features: Tensor, task_id: str):
        features = self._forward_conv(features)
        features = torch.flatten(features, 1)
        if task_id == "1":
            features = self.fc_task1(features)
        else:
            features = self.fc_task2(features)
        return features
