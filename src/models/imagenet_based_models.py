from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import torch
from torch import Tensor
from torch.nn import ModuleDict, Linear, Module
from torchvision.models import vgg16

from models.cifar10 import ResidualBlock
from utilities.utils import xavier_uniform_initialize, safely_load_state_dict
from utilities.utils import to_numpy
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class PredictionLayerConfig:
    task_id: str
    nb_classes: int


class VGG16(Module):
    def __init__(self, prediction_layers: List[PredictionLayerConfig], pretrained_backbone: bool = True):
        super().__init__()
        self.prediction_layers = prediction_layers
        self.pretrained_backbone = pretrained_backbone
        self.backbone = vgg16(pretrained=self.pretrained_backbone, progress=False)
        self.task_prediction = ModuleDict(
            {conf.task_id: Linear(in_features=4096, out_features=conf.nb_classes) for conf in self.prediction_layers}
        )
        self.block_output_size = {"block0": 4096, "block1": 4096}

        for layer_w in self.task_prediction.values():
            xavier_uniform_initialize(layer_w)

    def _forward2representation(self, features: Tensor) -> Tensor:
        features = self.backbone.features(features)
        features = self.backbone.avgpool(features)
        features = torch.flatten(features, 1)
        features = self.backbone.classifier[:6](features)  # removing the final classifier
        return features

    def block_forward(self, features: Tensor, task_id, numpy_return: bool = True) -> Dict[str, np.ndarray]:
        # TODO: task_id is legacy parameter, need to refactor the code and remove it.
        features = self.backbone.features(features)
        features = self.backbone.avgpool(features)
        features = torch.flatten(features, 1)
        if numpy_return:
            blocks_recorded = {
                "block0": to_numpy(self.backbone.classifier[0](features)),
                "block1": to_numpy(self.backbone.classifier[:4](features)),
            }
        else:
            blocks_recorded = {
                "block0": self.backbone.classifier[0](features),
                "block1": self.backbone.classifier[:4](features),
            }
        return blocks_recorded

    def forward(self, features: Tensor, task_id: str):
        if task_id == "imagenet":
            return self.backbone(features)
        else:
            features = self._forward2representation(features=features)
            # pass to the right task head
            return self.task_prediction[task_id](features)


class MiniResNetPretrained(Module):
    def __init__(self):
        super().__init__()

        initial_channels = 64
        n_stages = 4
        n_blocks_per_stage = 2
        n_channels = [initial_channels * 2 ** i for i in range(n_stages)]

        self.stage0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=initial_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(num_features=initial_channels),
            nn.ReLU(inplace=True),
        )
        # out: 64*32*32
        self.stage1 = self._make_stage(n_channels[0], n_channels[0], n_blocks_per_stage, ResidualBlock, stride=1)
        # out: 64*32*32
        self.stage2 = self._make_stage(n_channels[0], n_channels[1], n_blocks_per_stage, ResidualBlock, stride=2)
        # out: 128*16*16
        self.stage3 = self._make_stage(n_channels[1], n_channels[2], n_blocks_per_stage, ResidualBlock, stride=2)
        # out: 256*8*8
        self.stage4 = self._make_stage(n_channels[2], n_channels[3], n_blocks_per_stage, ResidualBlock, stride=2)
        # out: 512*4*4

        with torch.no_grad():
            dummy_data = torch.zeros(
                (1, 3, 64, 64),
                dtype=torch.float32,
            )
            self.feature_size = self.forward_conv(dummy_data).view(-1).shape[0]

        self.fc_imagenet = nn.Linear(self.feature_size, 1000)

    def _make_stage(self, in_channels, out_channels, n_blocks, block, stride):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_name = f"block{index + 1}"
            if index == 0:
                stage.add_module(block_name, block(in_channels, out_channels, stride=stride))
            else:
                stage.add_module(block_name, block(out_channels, out_channels, stride=1))
        return stage

    def forward_conv(self, features: Tensor):
        features = self.stage0(features)
        features = self.stage1(features)
        features = self.stage2(features)
        features = self.stage3(features)
        features = self.stage4(features)
        features = F.adaptive_avg_pool2d(features, output_size=1)
        features = torch.flatten(features, 1)
        return features

    def forward(self, features: Tensor):
        features = self.forward_conv(features)
        features = self.fc_imagenet(features)
        return features


class MiniResNet(Module):
    def __init__(self, back_bone_path: Optional[str], prediction_layers: List[PredictionLayerConfig]):
        super().__init__()

        self.prediction_layers = prediction_layers
        self.backbone = MiniResNetPretrained()

        if back_bone_path is not None:
            self.backbone.load_state_dict(safely_load_state_dict(back_bone_path))
        else:
            xavier_uniform_initialize(self.backbone)

        self.task_prediction = ModuleDict(
            {conf.task_id: Linear(in_features=512, out_features=conf.nb_classes) for conf in self.prediction_layers}
        )
        self.block_output_size = {"block0": self.backbone.feature_size}

        for layer_w in self.task_prediction.values():
            xavier_uniform_initialize(layer_w)

    def block_forward(self, features: Tensor, task_id, numpy_return: bool = True) -> Dict[str, np.ndarray]:
        # TODO: task_id is legacy parameter, need to refactor the code and remove it.
        features = self.backbone.forward_conv(features)
        if numpy_return:
            blocks_recorded = {"block0": to_numpy(features)}
        else:
            blocks_recorded = {"block0": features}
        return blocks_recorded

    def forward(self, features: Tensor, task_id: str):
        if task_id == "imagenet":
            return self.backbone(features)
        else:
            features = self.backbone.forward_conv(features=features)
            # pass to the right task head
            return self.task_prediction[task_id](features)


class MiniResNetSupConPretrained(Module):
    def __init__(self):
        super().__init__()

        initial_channels = 64
        n_stages = 4
        n_blocks_per_stage = 2
        n_channels = [initial_channels * 2 ** i for i in range(n_stages)]

        self.stage0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=initial_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(num_features=initial_channels),
            nn.ReLU(inplace=True),
        )
        # out: 64*32*32
        self.stage1 = self._make_stage(n_channels[0], n_channels[0], n_blocks_per_stage, ResidualBlock, stride=1)
        # out: 64*32*32
        self.stage2 = self._make_stage(n_channels[0], n_channels[1], n_blocks_per_stage, ResidualBlock, stride=2)
        # out: 128*16*16
        self.stage3 = self._make_stage(n_channels[1], n_channels[2], n_blocks_per_stage, ResidualBlock, stride=2)
        # out: 256*8*8
        self.stage4 = self._make_stage(n_channels[2], n_channels[3], n_blocks_per_stage, ResidualBlock, stride=2)
        # out: 512*4*4

        with torch.no_grad():
            dummy_data = torch.zeros(
                (1, 3, 64, 64),
                dtype=torch.float32,
            )
            self.feature_size = self.forward_conv(dummy_data).view(-1).shape[0]

        self.sup_con_projection = nn.Linear(self.feature_size, self.feature_size)
        self.fc_imagenet = nn.Linear(self.feature_size, 1000)

    def _make_stage(self, in_channels, out_channels, n_blocks, block, stride):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_name = f"block{index + 1}"
            if index == 0:
                stage.add_module(block_name, block(in_channels, out_channels, stride=stride))
            else:
                stage.add_module(block_name, block(out_channels, out_channels, stride=1))
        return stage

    def forward_conv(self, features: Tensor):
        features = self.stage0(features)
        features = self.stage1(features)
        features = self.stage2(features)
        features = self.stage3(features)
        features = self.stage4(features)
        features = F.adaptive_avg_pool2d(features, output_size=1)
        features = torch.flatten(features, 1)
        return features

    def forward_supcon(self, features: Tensor):
        features = self.forward_conv(features)
        features = self.sup_con_projection(features)
        return features

    def forward(self, features: Tensor, encoder_only: bool = False):
        if encoder_only:
            return self.forward_supcon(features=features)
        with torch.inference_mode():
            features = self.forward_conv(features)
        features = self.fc_imagenet(features.clone())
        return features


class MiniResNetSupCon(Module):
    def __init__(self, back_bone_path: Optional[str], prediction_layers: List[PredictionLayerConfig]):
        super().__init__()

        self.prediction_layers = prediction_layers
        self.backbone = MiniResNetSupConPretrained()

        if back_bone_path is not None:
            self.backbone.load_state_dict(safely_load_state_dict(back_bone_path))
        else:
            xavier_uniform_initialize(self.backbone)

        self.task_prediction = ModuleDict(
            {conf.task_id: Linear(in_features=512, out_features=conf.nb_classes) for conf in self.prediction_layers}
        )
        self.block_output_size = {"block0": self.backbone.feature_size}

        for layer_w in self.task_prediction.values():
            xavier_uniform_initialize(layer_w)

    def block_forward(self, features: Tensor, task_id, numpy_return: bool = True) -> Dict[str, np.ndarray]:
        # TODO: task_id is legacy parameter, need to refactor the code and remove it.
        features = self.backbone.forward_conv(features)
        if numpy_return:
            blocks_recorded = {"block0": to_numpy(features)}
        else:
            blocks_recorded = {"block0": features}
        return blocks_recorded

    def forward(self, features: Tensor, task_id: str, encoder_only: bool = False):
        if encoder_only:
            return self.backbone.forward_supcon(features=features)
        if task_id == "imagenet":
            return self.backbone(features)
        else:
            with torch.inference_mode():
                features = self.backbone.forward_conv(features=features)
            # pass to the right task head
            return self.task_prediction[task_id](features.clone())
