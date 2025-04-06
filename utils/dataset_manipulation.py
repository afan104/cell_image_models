import json
import os
import re
import shutil
from pathlib import Path

import distinctipy
import pandas as pd
import torch
import torchvision
import torchvision.transforms.v2 as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.tv_tensors import BoundingBoxes, Mask

from utils.base_utils import create_polygon_mask

"""
Constants in this file (transformations):
- data_aug_tfms
- final_tfms
- train_tfms

Functions in this file:
- collate_fn: zips two lists into a list of tuples
- get_dataloader_params: returns the dataloader parameters as a dictionary
- load_data: loads data from directory into variables
- create_dataloaders: creates a custom dataset with transforms, then converts it into a dataloader
- move_file: moves a file from one directory to another
- move_test_data: moves test images/jsons into a test folder

Classes in this file:
- Kog1Classifier: Dataset class that transforms my image+mask data into a custom dataset
"""


# Transforms for data aug
data_aug_tfms2 = transforms.Compose(
    transforms=[
        transforms.GaussianBlur(3, 0.5),
        transforms.RandomEqualize(),
        transforms.RandomPosterize(bits=3, p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
    ],
)
# Compose transforms to sanitize bounding boxes and normalize input data
final_tfms = transforms.Compose(
    transforms=[
        transforms.ToImage(),
        transforms.ToDtype(torch.float16, scale=True),
        transforms.SanitizeBoundingBoxes(),
    ]
)
# Define the transformations for training and validation datasets
train_tfms2 = transforms.Compose(
    transforms=[
        data_aug_tfms2,
        final_tfms,
        # transforms.GaussianNoise(0, 0.01, True),
    ]  # valid_tfms is same as final_tfms
)


# Define parameters for DataLoader
def collate_fn(batch):
    return tuple(zip(*batch))


def get_dataloader_params(device, bs, num_workers):
    return {
        "batch_size": bs,
        "num_workers": num_workers,
        "persistent_workers": True,
        "pin_memory": "cuda" in device,
        "pin_memory_device": device,
        "collate_fn": collate_fn,
        "persistent_workers": True,
    }


# Load dataset images and annotations
def load_data(dataset_path, include_data_aug_paths=False):
    # Define path to test directory
    test_dir = Path(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../Data/test"))
    )

    # Collect all image file paths from main dataset and test directory
    img_file_paths = list(dataset_path.glob("*.png")) + list(test_dir.glob("*.png"))

    # Optionally include data augmentation images
    if include_data_aug_paths:
        data_aug_dirs = [
            Path(
                os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "../Data/Data_aug/mirror")
                )
            ),
            Path(
                os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "../Data/Data_aug/rotate90")
                )
            ),
        ]

        for aug_dir in data_aug_dirs:
            aug_images = list(aug_dir.glob("*.png"))
            img_file_paths += aug_images

        # Validate that each augmentation folder has exactly 80 images
        assert (
            len(list(data_aug_dirs[0].glob("*.png"))) == 80
        ), f"Expected 80 images in {data_aug_dirs[0]}"
        assert (
            len(list(data_aug_dirs[1].glob("*.png"))) == 80
        ), f"Expected 80 images in {data_aug_dirs[1]}"

    # Map image filenames (without extensions) to their full file paths
    img_dict = {file.stem: file for file in img_file_paths}

    # Collect all annotation file paths (JSON) from dataset and test directory
    annotation_file_paths = list(dataset_path.glob("fz*.json")) + list(
        test_dir.glob("fz*.json")
    )

    # Optionally include annotation files from data augmentation directories
    if include_data_aug_paths:
        for aug_dir in data_aug_dirs:
            annotation_file_paths += list(aug_dir.glob("*.json"))

    # Load and combine annotation JSON files into a single dataframe
    mask_dataframes = (
        pd.read_json(f, orient="index").transpose() for f in annotation_file_paths
    )
    annotation_df = pd.concat(mask_dataframes, ignore_index=False)

    # Extract index from imageName for proper alignment with img_dict
    annotation_df["index"] = annotation_df["imageName"].apply(
        lambda name: re.split(r"[./]", name)[-2]
    )
    annotation_df = annotation_df.set_index("index")
    annotation_df = annotation_df.loc[list(img_dict.keys())]

    # Convert 'shapes' column into a separate dataframe
    shapes_df = annotation_df["shapes"].explode().to_frame()
    shapes_df = shapes_df["shapes"].apply(pd.Series)

    # Manually define class names to ensure consistent ordering
    class_names = ["background", "kog1", "normal"]

    # Generate a colormap for the classes
    colors = distinctipy.get_colors(len(class_names))
    int_colors = [tuple(int(c * 255) for c in color) for color in colors]

    return img_dict, annotation_df, shapes_df, class_names, int_colors


# create dataloaders
def create_dataloaders(
    img_dict,
    annotation_df,
    class_names,
    device,
    bs,
    num_workers,
    train_tfms=None,
):
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    # get test files from test directory
    test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../Data/test"))
    files_test = [f.split(".")[0] for f in os.listdir(test_dir) if f.endswith(".png")]

    # get val files from test directory
    val_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../Data/val"))
    files_val = [f.split(".")[0] for f in os.listdir(val_dir) if f.endswith(".png")]

    # get train files from dataset directory
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../Data"))
    files_train = [
        f.split(".")[0] for f in os.listdir(dataset_dir) if f.endswith(".png")
    ]

    # transforms in dataset
    train_dataset = Kog1Classifier(
        img_keys=files_train,
        annotation_df=annotation_df,
        img_dict=img_dict,
        class_to_idx=class_to_idx,
        transforms=train_tfms if train_tfms is not None else final_tfms,
    )
    valid_dataset = Kog1Classifier(
        img_keys=files_val,
        annotation_df=annotation_df,
        img_dict=img_dict,
        class_to_idx=class_to_idx,
        transforms=final_tfms,
    )
    test_dataset = Kog1Classifier(
        img_keys=files_test,
        annotation_df=annotation_df,
        img_dict=img_dict,
        class_to_idx=class_to_idx,
        transforms=final_tfms,
    )

    dataloader_params = get_dataloader_params(
        device=device, bs=bs, num_workers=num_workers
    )
    train_dataloader = DataLoader(train_dataset, **dataloader_params, shuffle=True)
    valid_dataloader = DataLoader(
        valid_dataset,
        **dataloader_params,
        shuffle=False,
    )
    test_dataloader = DataLoader(
        test_dataset,
        **dataloader_params,
        shuffle=False,
    )

    return train_dataloader, valid_dataloader, test_dataloader


# Function to move a file
def move_file(src, dest):
    if os.path.exists(src):
        shutil.move(src, dest)
        print(f"Moved: {src} to {dest}")
    else:
        print(f"File not found: {src}")


# Function that moves test data into test folder
def move_test_data():
    # Define paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../Data"))
    json_path = os.path.join(base_dir, "annotation_tracker.json")
    test_dir = os.path.join(base_dir, "test")

    # Create test directory if it doesn't exist
    os.makedirs(test_dir, exist_ok=True)

    # Load JSON data
    print(f"Loading JSON data from {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)

    # Get list of test filenames
    test_names = data.get("test", [])

    # Move test images and json files
    for test_name in test_names:
        # Construct full source and destination paths for both .png and .json
        src_png_path = os.path.join(base_dir, f"{test_name}.png")
        dest_png_path = os.path.join(test_dir, f"{test_name}.png")
        src_json_path = os.path.join(base_dir, f"{test_name}.json")
        dest_json_path = os.path.join(test_dir, f"{test_name}.json")

        # Move .png and .json files
        move_file(src_png_path, dest_png_path)
        move_file(src_json_path, dest_json_path)


class Kog1Classifier(Dataset):
    def __init__(
        self,
        img_keys,
        annotation_df,
        img_dict,
        class_to_idx,
        transforms=None,
    ):
        """
        Parameters:
            img_keys (list): List of unique identifiers for images.
            annotation_df (DataFrame): DataFrame containing the image annotations.
            img_dict (dict): Dictionary mapping image identifiers to image file paths.
            class_to_idx (dict): Dictionary mapping class labels to indices.
            transforms (callable, optional): Optional transform to be applied on a sample.
        """
        super().__init__()

        self.img_keys = img_keys
        self.annotation_df = annotation_df
        self.img_dict = img_dict
        self.class_to_idx = class_to_idx
        self.transforms = transforms

    def __len__(self):
        """
        Returns length of dataset.
        """
        return len(self.img_keys)

    def __getitem__(self, index):
        """
        Returns a tuple containing image and target annotations.
        """
        img_key = self.img_keys[index]
        annotation = self.annotation_df.loc[img_key]
        image, target = self.load_image_and_target(annotation)
        image, target = self.transforms(image, target)

        return image, target

    def load_image_and_target(self, annotation):
        """
        Load image and its target from giving the annotation

        Parameters:
        annotation (pd.Series): image annotation

        Returns:
        Tuple containing image and dictionary with 'boxes' and 'labels'
        """
        # get image
        filepath = self.img_dict[annotation.name]
        image = Image.open(filepath).convert("RGB")

        # convert labels to indices
        try:
            labels = torch.Tensor(
                [self.class_to_idx[shape["label"]] for shape in annotation["shapes"]]
            ).to(dtype=torch.int64)
        except KeyError:
            # show which file_id it is
            print(f"KeyError: {annotation.name}")
            raise KeyError(
                f"KeyError: {annotation.name} - Check if the class names are consistent."
            )

        # get masks
        shape_points = [shape["points"] for shape in annotation["shapes"]]
        all_shapes = [[tuple(p) for p in points] for points in shape_points]
        mask_imgs = [create_polygon_mask(image.size, shape) for shape in all_shapes]
        masks = Mask(
            torch.concat(
                [
                    Mask(transforms.PILToTensor()(mask_img), dtype=torch.bool)
                    for mask_img in mask_imgs
                ]
            )
        )

        # get bounding boxes
        bboxes = BoundingBoxes(
            data=torchvision.ops.masks_to_boxes(masks),
            format="xyxy",
            canvas_size=image.size[::-1],
            dtype=torch.float16,
        )

        return image, {"masks": masks, "boxes": bboxes, "labels": labels}
