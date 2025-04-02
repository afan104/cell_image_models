import argparse
import os
import random
from pathlib import Path

import torch
from tqdm import tqdm

from utils.dataset_manipulation import create_dataloaders, load_data, train_tfms2
from utils.model_classes import get_model
from utils.train_test_utils import (
    OPS_HPS,
    get_optim,
    save_model,
    test_one_epoch,
    train_one_epoch,
)
from utils.visualization_utils import visualize_losses, visualize_maps, visualize_multi

PREPROCESS_FOLDER_NAMES = {
    "manual": "manual_autocontrast",
    "torch": "torch_autocontrast",
    "pil": "pil_autocontrast",
}

# Create an argument parser
parser = argparse.ArgumentParser()


# Define the hyperparameters that will be passed
def parse_args():
    """Parse command-line arguments for hyperparameter tuning."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate the model with various hyperparameters."
    )

    # Hyperparameters
    parser.add_argument(
        "--freeze",
        type=bool,
        required=False,
        default=False,
        help="Whether to freeze the model. Skip the flag to not freeze.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--contraster_type",
        type=str,
        required=False,
        default=None,
        help="""Type of contrast adjustment to apply to images. 
        Can be manual, pil, torch, None""",
    )

    return parser.parse_args()


# fixed hyperparameters
bs = 2
num_workers = 1
dataset_path = Path(f"{Path(os.getcwd())}/Data")
epochs = 30
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16

if __name__ == "__main__":
    # hyperparameters/adjustables
    args = parse_args()
    optimizer_info = OPS_HPS[args.optimizer]

    # set seed
    seed = 1234
    random.seed(seed)

    # Get Data
    img_dict, annotation_df, shapes_df, class_names, int_colors = load_data(
        dataset_path=dataset_path
    )

    # Use the contrast adjustment type if specified
    if args.contraster_type:
        script_dir = os.path.dirname(__file__)
        preprocess_img_dir = os.path.join(
            script_dir, f"Data/{PREPROCESS_FOLDER_NAMES[args.contraster_type]}"
        )
        preprocessed_img_dict = {
            k: Path(f"{preprocess_img_dir}/{k}.png") for k in img_dict.keys()
        }
        img_dict = preprocessed_img_dict

    # get dataloaders
    train_dataloader, valid_dataloader = create_dataloaders(
        img_dict=img_dict,
        annotation_df=annotation_df,
        class_names=class_names,
        device=device,
        dtype=dtype,
        bs=bs,
        num_workers=num_workers,
        train_tfms=train_tfms2,
    )

    # load model
    base_model_path = "save_models/model_2025-03-13_06-24-45.pth"
    model = get_model(
        class_names=class_names,
        device=device,
        base_model_path=base_model_path,
        freeze_params=args.freeze,
        dtype=dtype,
    )
    # train/test loop
    params_to_optimize = [
        p for p in model.parameters() if p.requires_grad
    ]  # skip frozen params otherwise the optimizer will throw an error
    optimizer = get_optim(optimizer_info, params_to_optimize)
    loss_per_epoch = []  # = List[List[tensor]]
    maps_logger = {  # Dict[str: List[List[tensor]]]
        "segm_map_50": [],
        "bbox_map_50": [],
        "kog1_segm_map": [],
        "normal_segm_map": [],
        "kog1_bbox_map": [],
        "normal_bbox_map": [],
    }
    for epoch in tqdm(range(epochs), desc="Epochs"):
        loss_per_epoch.append(
            train_one_epoch(
                model=model,
                dataloader=train_dataloader,
                device=device,
                optimizer=optimizer,
            )
        )
        maps_logger = test_one_epoch(
            model=model,
            dataloader=valid_dataloader,
            device=device,
            maps_logger=maps_logger,
        )

    # save model
    # save_model(model=model)

    # check all values
    print(f"all losses: {loss_per_epoch}")
    print(f"maps: {maps_logger}")

    # visualization
    args = {k: v for k, v in args._get_kwargs()}
    version = "_".join([f"{k}_{v}" for k, v in args.items()])

    # make directory if not exists
    print(f"Saving version: {version}")
    if not os.path.exists(f"output/{version}"):
        print(f"Creating directory: output/{version}")
        os.makedirs(f"output/{version}")

    # losses
    visualize_losses(loss_per_epoch, f"output/{version}/losses.png")

    # maps
    visualize_maps(maps_logger, f"output/{version}/maps.png")

    # visualize
    visualize_multi(
        model,
        annotation_df,
        img_dict,
        int_colors,
        class_names,
        device,
        n=3,
        folder=f"output/{version}",
    )

    print("Done")
