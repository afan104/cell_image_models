import os

import matplotlib.pyplot as plt
import torch
import torchvision
from tqdm import tqdm

from utils.dataset_manipulation import load_data
from utils.model_classes import get_model

# ------------------------------ #
# Config
# ------------------------------ #
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
NUM_WORKERS = 1
BASE_MODEL_PATH = "save_models/model_2025-03-13_06-24-45.pth"
PREPROCESS_TYPE = "manual"
INCLUDE_DATA_AUG_PATHS = False
KERNEL_SIZE = 1
FREEZE_MODEL_PARAMS = False
SAVE_DIR = "conv_vis"


# ------------------------------ #
# Utility Functions
# ------------------------------ #
def normalize_kernels(kernels):
    kernels = kernels - kernels.min()
    kernels = kernels / kernels.max()
    return kernels


def visualize_and_save_kernels(kernels, layer_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    kernel_number, kernel_depth, kernel_height, kernel_width = kernels.shape

    if kernel_depth > 3:
        kernels = kernels.mean(dim=1, keepdim=True)

    # kernel is size [kernel_number, kernel_depth, kernel_height, kernel_width]

    grid = torchvision.utils.make_grid(kernels, nrow=10, normalize=True, pad_value=1)
    grid = grid.permute(1, 2, 0).numpy().astype("float32")

    # Plot and save
    save_path = os.path.join(save_dir, f"{layer_name}.png")
    plt.figure(figsize=(10, 10))
    plt.imshow(grid)
    plt.axis("off")
    plt.ioff()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def extract_conv_weights(model):
    layers = []

    # Input conv layer
    in_conv = model.preprocess.in_block[0]
    if isinstance(in_conv, torch.nn.Conv2d):
        layers.append(("in_block_0", in_conv.weight.detach().cpu()))

    # Intermediate conv layers
    for i, layer in enumerate(model.preprocess.intermedConvBlock):
        if isinstance(layer, torch.nn.Conv2d):
            layers.append((f"intermedConvBlock_{i}", layer.weight.detach().cpu()))

    return layers


# ------------------------------ #
# Main Workflow
# ------------------------------ #
def main():
    # Step 1: Load data (we only need class_names here)
    _, _, _, class_names, _ = load_data(
        preprocess_type=PREPROCESS_TYPE, include_data_aug_paths=INCLUDE_DATA_AUG_PATHS
    )

    # Step 2: Load model
    model = get_model(
        class_names=class_names,
        device=DEVICE,
        base_model_path=BASE_MODEL_PATH,
        freeze_params=FREEZE_MODEL_PARAMS,
        kernel_size=KERNEL_SIZE,
    )

    # Step 3: Extract and visualize conv layers
    conv_layers = extract_conv_weights(model)

    for layer_name, weights in conv_layers:
        normalized_kernels = normalize_kernels(weights)
        visualize_and_save_kernels(normalized_kernels, layer_name, SAVE_DIR)

    print(f"✅ All visualizations saved in '{SAVE_DIR}'")


if __name__ == "__main__":
    main()