import os
import torch.nn.functional as F
from torchvision.utils import save_image
from pathlib import Path
import torch
from utils.model_classes import get_model
from utils.dataset_manipulation import (
    Kog1Classifier,
    get_dataloader_params,
    final_tfms,
    load_data,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------- #
# Config
# ---------------------------- #
HEAVY_MODEL_PARAMS = {
    "hidden_units": 128,
    "padding": 3,
    "num_interm_blocks": 2,
    "max_pool": True,
    "mp_size": 3,
}
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_MODEL_PATH = "save_models/model_2025-03-13_06-24-45.pth"

# ---------------------------- #
# Utility
# ---------------------------- #

def normalize_kernels(kernels):
    kernels = kernels - kernels.min()
    kernels = kernels / kernels.max()
    return kernels


def extract_conv_weights(model):
    layers = []

    # Input conv layer
    in_conv = model.preprocess.in_block[0]
    if isinstance(in_conv, torch.nn.Conv2d):
        weight = in_conv.weight.detach().cpu()
        bias = in_conv.bias.detach().cpu() if in_conv.bias is not None else None
        layers.append(("in_block_0", weight, bias))

    # Intermediate conv layers
    for i, layer in enumerate(model.preprocess.intermedConvBlock):
        if isinstance(layer, torch.nn.Conv2d):
            weight = layer.weight.detach().cpu()
            bias = layer.bias.detach().cpu() if layer.bias is not None else None
            layers.append((f"intermedConvBlock_{i}", weight, bias))

    return layers

def create_full_dataloader(
    img_dict,
    annotation_df,
    class_names,
    device,
    bs,
    num_workers,
):
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    # get test files from test directory
    test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "./Data/test"))
    files_test = [
        f.split(".")[0]
        for f in os.listdir(test_dir)
        if f.startswith("fz") and f.endswith(".json")
    ]

    # get val files from test directory
    val_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "./Data/val"))
    files_val = [
        f.split(".")[0]
        for f in os.listdir(val_dir)
        if f.startswith("fz") and f.endswith(".json")
    ]

    # get train files from dataset directory
    dataset_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "./Data/train")
    )
    files_train = [
        f.split(".")[0]
        for f in os.listdir(dataset_dir)
        if f.startswith("fz") and f.endswith(".json")
    ]

    # transforms in dataset
    full_dataset = Kog1Classifier(
        img_keys=files_train + files_val + files_test,
        annotation_df=annotation_df,
        img_dict=img_dict,
        class_to_idx=class_to_idx,
        transforms=final_tfms,
    )

    dataloader_params = get_dataloader_params(
        device=device, bs=bs, num_workers=num_workers
    )
    full_dataloader = DataLoader(full_dataset, **dataloader_params, shuffle=False)
    return full_dataloader


if __name__ == "__main__":
    # get data
    img_dict, annotation_df, shapes_df, class_names, int_colors = load_data(
        preprocess_type=None, include_data_aug_paths=False
    )
    full_dataloader = create_full_dataloader(
        img_dict=img_dict,
        annotation_df=annotation_df,
        class_names=class_names,
        device=DEVICE,
        bs=1,
        num_workers=NUM_WORKERS,
    )
    
    # load model
    model_path = "save_models/freeze_False_optimizer_v1_contraster_type_None_data_aug_False_kernel_size_1_final.pth"
    loaded_model = get_model(
            class_names=class_names,
            device=DEVICE,
            base_model_path=BASE_MODEL_PATH,
            freeze_params=False,
            kernel_size=1,
        )
    loaded_model.load_state_dict(torch.load(
        model_path,
        weights_only=True,
        map_location=torch.device(DEVICE),
    ))
    loaded_model.to(DEVICE, torch.float16)
    for p in loaded_model.parameters():
        assert p.device.type == 'cuda'
    
    # extract and visualize conv layers
    conv_layers = extract_conv_weights(loaded_model)

    for batch, (x,_) in enumerate(full_dataloader):
        x = torch.stack(x).to(DEVICE, torch.float16)
        x = torch.mean(x, dim=1, keepdim=True)
        # save input image
        output_dir = f"conv_vis/{batch}/"
        os.makedirs(output_dir, exist_ok=True)
        save_image(x[0], f"{output_dir}/input.png")
        current = x.clone()
        # show intermediate images
        for layer_name, weights, bias in conv_layers:
            # implement layer
            weights = weights.to(DEVICE, torch.float16)
            bias = bias.to(DEVICE, torch.float16) if bias is not None else None
            conv_layer = F.conv2d(current, weights, bias=bias, stride=1, padding=3)
            conv_layer = F.relu(conv_layer)

            # save after image
            mean_image = conv_layer[0].mean(dim=0, keepdim=True)
            save_image(mean_image, f"{output_dir}/{layer_name}.png")
            current = conv_layer
