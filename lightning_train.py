import argparse
import os
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping

from utils.dataset_manipulation import create_dataloaders, load_data, train_tfms2
from utils.lightning_engine import LightningMaskRCNNModel, LossAndMapLogger
from utils.model_classes import get_model
from utils.train_test_utils import OPS_HPS, get_optim, save_model
from utils.visualization_utils import visualize_losses, visualize_maps, visualize_multi

PREPROCESS_FOLDER_NAMES = {
    "manual": "manual_autocontrast",
    "torch": "torch_autocontrast",
    "pil": "pil_autocontrast",
}

BATCH_SIZE = 2
NUM_WORKERS = 1
DATA_PATH = Path(f"{Path(os.getcwd())}/Data")
EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_MODEL_PATH = "save_models/model_2025-03-13_06-24-45.pth"

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
    parser.add_argument(
        "--data_aug",
        type=bool,
        required=False,
        default=False,
        help="""Whether to include the additional 160 augmented training images (rotate90 and mirror) during training.
          Skip the flag to not exclude them.""",
    )
    parser.add_argument(
        "--kernel_size",
        type=int,
        required=True,
        help="""Kernel size for conv2d layers in the AdaptivePreprocessing module.""",
    )

    return parser.parse_args()


def get_data(dataset_path, args):

    # Get Data
    img_dict, annotation_df, shapes_df, class_names, int_colors = load_data(
        dataset_path=dataset_path, include_data_aug_paths=args.data_aug
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

    return img_dict, annotation_df, shapes_df, class_names, int_colors


def find_last_ckpt(version):
    base_dir = f"output/{version}/lightning_logs"
    if os.path.exists(base_dir):
        # Get latest version
        version_before_current = sorted(os.listdir(base_dir))[-1]
        ckpt_dir = os.path.join(base_dir, version_before_current, "checkpoints")

        # Get latest checkpoint
        all_files = os.listdir(ckpt_dir)
        latest_ckpt = sorted(
            [f for f in all_files if os.path.isfile(os.path.join(ckpt_dir, f))]
        )[-1]
        latest_ckpt = os.path.join(ckpt_dir, latest_ckpt)
        print(f"Loading checkpoint: {latest_ckpt}")
        return latest_ckpt
    return None


def setup_version(args, seed: int = 1234):
    """Seed, set precision, and create directories."""
    L.seed_everything(seed)
    torch.set_float32_matmul_precision("high")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "caching_allocator"

    # Versioning
    args_dict = {k: v for k, v in args._get_kwargs()}
    version = "_".join([f"{k}_{v}" for k, v in args_dict.items()])

    # make directory if not exists
    print(f"Saving version: {version}")
    if not os.path.exists(f"output/{version}"):
        print(f"Creating directory: output/{version}")
        os.makedirs(f"output/{version}")

    return version


def get_callbacks(train_dataloader, valid_dataloader, version):
    loss_and_maps_logger_callback = LossAndMapLogger(
        epochs=EPOCHS,
        train_dataloader_size=len(train_dataloader),
        val_dataloader_size=len(valid_dataloader),
        version=version,
    )
    early_stop_callback = EarlyStopping(
        monitor="bbox_map_50", min_delta=0.00, patience=5, verbose=False, mode="max"
    )
    return [loss_and_maps_logger_callback, early_stop_callback]


def build_model(class_names, args, optimizer_info):
    lit_model = None
    with trainer.init_module():
        # get model
        model = get_model(
            class_names=class_names,
            device=DEVICE,
            base_model_path=BASE_MODEL_PATH,
            freeze_params=args.freeze,
            kernel_size=args.kernel_size,
        )

        params_to_optimize = [p for p in model.parameters() if p.requires_grad]
        optimizer = get_optim(optimizer_info, params_to_optimize)

        lit_model = LightningMaskRCNNModel(model, optimizer)
    return lit_model


def visualize(
    loss_and_maps_logger_callback,
    version,
    img_dict,
    annotation_df,
    int_colors,
    class_names,
    multi_example_count=3,
):
    # losses
    visualize_losses(
        data=loss_and_maps_logger_callback.loss_logger,
        file_name=f"output/{version}/losses.png",
    )

    # maps
    visualize_maps(
        maps=loss_and_maps_logger_callback.maps_logger_train,
        file_name=f"output/{version}/maps_train.png",
    )
    visualize_maps(
        maps=loss_and_maps_logger_callback.maps_logger_val,
        file_name=f"output/{version}/maps_val.png",
    )

    # visualize
    visualize_multi(
        lit_model.model,
        annotation_df,
        img_dict,
        int_colors,
        class_names,
        DEVICE,
        n=multi_example_count,
        folder=f"output/{version}",
    )


if __name__ == "__main__":
    # hyperparameters/adjustables
    args = parse_args()
    optimizer_info = OPS_HPS[args.optimizer]
    version = setup_version(args=args)

    # Get data
    img_dict, annotation_df, shapes_df, class_names, int_colors = get_data(
        dataset_path=DATA_PATH, args=args
    )

    # Get dataloaders
    train_dataloader, valid_dataloader, test_dataloader = create_dataloaders(
        img_dict=img_dict,
        annotation_df=annotation_df,
        class_names=class_names,
        device=DEVICE,
        bs=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        train_tfms=train_tfms2,
    )

    # Get Callbacks
    callbacks = get_callbacks(
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        version=version,
    )

    # Lightning Trainer
    trainer = L.Trainer(
        max_epochs=EPOCHS,
        logger=True,
        callbacks=callbacks,
        accelerator=DEVICE,
        log_every_n_steps=5,
        num_sanity_val_steps=0,
        default_root_dir=f"output/{version}",
    )

    lit_model = build_model(
        class_names=class_names, args=args, optimizer_info=optimizer_info
    )
    latest_ckpt = find_last_ckpt(version)

    trainer.fit(
        model=lit_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
        ckpt_path=latest_ckpt,
    )

    save_model(model=lit_model.model, save_path=f"save_models/{version}_final.pth")

    # visualize
    visualize(
        loss_and_maps_logger_callback=callbacks[0],
        version=version,
        img_dict=img_dict,
        annotation_df=annotation_df,
        int_colors=int_colors,
        class_names=class_names,
        multi_example_count=3,
    )

    print("Done")
