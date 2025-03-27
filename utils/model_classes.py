import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    maskrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

"""
Functions in this File:
- get_model: returns a specified model
    - instantiates the pretrained maskrcnn model, adjusts it to my dataset, and adds a preprocessing module if specified
- calc_final_dim: calculates the final dimension after input goes through multiple convolution blocks and flattens

Classes in this File:
- AdaptiveContrastPreprocessing: determines dynamically "clamp" and "shift" values for brightening and increasing image contrast
- CompositeModel: Combines a preprocessing model and a base model
"""

# Preprocessing hyperparameters
HEAVY_MODEL_PARAMS = {
    "hidden_units": 128,
    "kernel_size": 1,
    "padding": 3,
    "num_interm_blocks": 2,
    "max_pool": True,
    "mp_size": 3,
}


# build preprocessing model
def build_preproc_model(**kwargs):
    hidden_units = kwargs.get("hidden_units")
    kernel_size = kwargs.get("kernel_size")
    padding = kwargs.get("padding")
    num_interm_blocks = kwargs.get("num_interm_blocks")
    max_pool = kwargs.get("max_pool")
    mp_size = kwargs.get("mp_size")
    dtype = kwargs.get("dtype")
    device = kwargs.get("device")
    return AdaptiveContrastPreprocessing(
        hidden_units=hidden_units,
        kernel_size=kernel_size,
        padding=padding,
        num_interm_blocks=num_interm_blocks,
        max_pool=max_pool,
        mp_size=mp_size,
    ).to(device, dtype=dtype)


# load model
def get_model(
    class_names,
    device,
    base_model_path=None,
    freeze_params=False,
    dtype=torch.float16,
):
    # Initialize pretrained model weights
    base_model = maskrcnn_resnet50_fpn_v2(
        weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    )

    # Replace box predictor/classifier
    in_features_box = base_model.roi_heads.box_predictor.cls_score.in_features
    base_model.roi_heads.box_predictor = FastRCNNPredictor(
        in_channels=in_features_box, num_classes=len(class_names)
    )

    # Replace mask predictor
    in_features_mask = base_model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    base_model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_channels=in_features_mask,
        dim_reduced=hidden_layer,
        num_classes=len(class_names),
    )

    if base_model_path is not None:
        base_model.load_state_dict(
            torch.load(
                base_model_path,
                weights_only=True,
                map_location=torch.device(device),
            )
        )
    base_model.to(device=device, dtype=dtype)
    if freeze_params:
        for p in base_model.parameters():
            p.requires_grad = False

    kwargs = {**HEAVY_MODEL_PARAMS, "device": device, "dtype": dtype}
    return CompositeModel(
        base_model=base_model,
        preprocess_model=build_preproc_model(**kwargs),
    )


class AdaptiveContrastPreprocessing(nn.Module):
    def __init__(
        self,
        hidden_units=128,
        kernel_size=1,
        padding=0,
        num_interm_blocks=0,
        max_pool=False,
        mp_size=1,
    ):
        super().__init__()
        self.img_size = (1608, 1608)

        # Small CNN to predict threshold and shift dynamically
        self.in_block = nn.Sequential(
            nn.Conv2d(1, hidden_units, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
        )

        # Intermediate blocks
        intermed_layers = []
        for i in range(num_interm_blocks):
            intermed_layers += [
                nn.Conv2d(
                    hidden_units, hidden_units, kernel_size=kernel_size, padding=padding
                ),
                nn.ReLU(),
                nn.MaxPool2d(mp_size) if max_pool else nn.Identity(),
            ]
        self.intermedConvBlock = nn.Sequential(*intermed_layers)

        # calculate final_dim
        final_dim = self.calc_final_dim(
            self.img_size[0],
            padding,
            kernel_size,
            mp_size,
            num_interm_blocks,
            hidden_units,
        )
        self.final_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(final_dim, 1),  # Output two values: black_threshold & shift
        )

        self.param_net = nn.Sequential(
            self.in_block,
            self.intermedConvBlock,
            nn.Conv2d(hidden_units, 1, kernel_size=kernel_size, padding=padding),
            nn.Sigmoid(),  # Clamp values between 0 and 1
        )

    def forward(self, img):
        """
        img: Tensor of shape (B, C, H, W) with values in [0, 255]
        """
        # Convert RGB to grayscale
        gray_img = torch.mean(img, dim=1, keepdim=True)

        # pass through CNN
        processed_img = self.param_net(gray_img)  # Shape: (B, 1, new_H, new_W)

        # Convert back to 3-channel RGB format
        processed_img = gray_img.expand(-1, 3, -1, -1)  # (B, 3, new_H, new_W)

        return processed_img

    def calc_final_dim(self, d, p, k, m, n_layers, h):
        """
        Parameters -
        :d is starting dim
        :p is padding
        :k is kernel size
        :m is maxpool size
        :n_layers is number layers
        :h is hidden units
        """
        # first layer
        d += 2 * (p) - (k - 1)
        # intermediate layers
        for _ in range(n_layers):
            d += 2 * (p) - (k - 1)
            d = math.floor(d / m)
        # flatten
        d = h * d * d
        return d


# Add to nn block
class CompositeModel(nn.Module):
    def __init__(self, preprocess_model, base_model):
        super().__init__()
        self.preprocess = preprocess_model
        self.base_model = base_model

    def forward(self, input, targets=None):
        # preprocess portion
        input = self.preprocess(input)

        # train: forward pass mask r cnn
        if targets is not None:
            return self.base_model(input, targets)
        # test: forward pass mask r cnn
        else:
            return self.base_model(input)
