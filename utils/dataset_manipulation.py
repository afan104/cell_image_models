import re

import distinctipy
import pandas as pd
import sklearn.model_selection as skm
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

Classes in this file:
- KoggClassifier: Dataset class that transforms my image+mask data into a custom dataset
"""

# Transforms for data aug
data_aug_tfms1 = transforms.Compose(
    transforms=[
        # transforms.RandomIoUCrop(
        #     min_scale=0.3,
        #     max_scale=1.0,
        #     min_aspect_ratio=0.5,
        #     max_aspect_ratio=2.0,
        #     sampler_options=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        #     trials=400,
        # ),
        transforms.ColorJitter(
            brightness=(0.875, 1.125),
            contrast=(0.5, 1.5),
            saturation=(0.5, 1.5),
            hue=(-0.05, 0.05),
        ),
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
train_tfms1 = transforms.Compose(
    transforms=[data_aug_tfms1, final_tfms]  # valid_tfms is same as final_tfms
)
# Transforms for data aug
data_aug_tfms2 = transforms.Compose(
    transforms=[
        # transforms.RandomIoUCrop(
        #     min_scale=0.3,
        #     max_scale=1.0,
        #     min_aspect_ratio=0.5,
        #     max_aspect_ratio=2.0,
        #     sampler_options=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        #     trials=400,
        # ),
        transforms.GaussianBlur(3, 0.5),
        transforms.RandomEqualize(),
        transforms.RandomPosterize(bits=3, p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
    ],
)

# Define the transformations for training and validation datasets
train_tfms2 = transforms.Compose(
    transforms=[
        data_aug_tfms2,
        final_tfms,
        # transforms.GaussianNoise(0, 0.01, True),
    ]  # valid_tfms is same as final_tfms
)

DATA_AUG_HPS = {"v0": train_tfms1, "v1": train_tfms2}


# Define parameters for DataLoader
def collate_fn(batch):
    return tuple(zip(*batch))


def get_dataloader_params(device, bs, num_workers):
    return {
        "batch_size": bs,
        "num_workers": num_workers,
        "persistent_workers": True,
        "pin_memory": "cuda" in device,
        "pin_memory_device": (device if "cuda" in device else ""),
        "collate_fn": collate_fn,
    }


# load data
def load_data(dataset_path):
    # Get a list of image files in the dataset
    img_file_paths = list(dataset_path.glob("*.png"))
    img_dict = {file.stem: file for file in img_file_paths}

    # create generator of json dataframes
    annotation_file_paths = list(dataset_path.glob("fz*.json"))
    mask_dataframes = (
        pd.read_json(f, orient="index").transpose() for f in annotation_file_paths
    )

    # concatenate into single dataframe with filename indices
    annotation_df = pd.concat(mask_dataframes, ignore_index=False)
    annotation_df["index"] = annotation_df.apply(
        lambda row: re.split(r"[./]", row["imageName"])[-2], axis=1
    )  # axis 1 applies on rows, re.split splits on both . and / characters
    annotation_df = annotation_df.set_index("index")
    annotation_df = annotation_df.loc[list(img_dict.keys())]

    # apply to shapes column to convert to series (separate column)
    shapes_df = annotation_df["shapes"].explode().to_frame()
    shapes_df = shapes_df["shapes"].apply(pd.Series)

    # hard-code class_names to be in the right order since my pretrained model requires a consistent order
    class_names = ["background", "kogg", "normal"]

    # Generate colormap
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
    dtype=torch.float16,
):
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    file_IDs = list(img_dict.keys())
    files_train, files_test = skm.train_test_split(
        file_IDs, train_size=0.8, test_size=0.2
    )

    # transforms in dataset
    train_dataset = KoggClassifier(
        img_keys=files_train,
        annotation_df=annotation_df,
        img_dict=img_dict,
        class_to_idx=class_to_idx,
        transforms=train_tfms if train_tfms is not None else final_tfms,
        dtype=dtype,
    )
    valid_dataset = KoggClassifier(
        img_keys=files_test,
        annotation_df=annotation_df,
        img_dict=img_dict,
        class_to_idx=class_to_idx,
        transforms=final_tfms,
        dtype=dtype,
    )

    dataloader_params = get_dataloader_params(
        device=device, bs=bs, num_workers=num_workers
    )
    train_dataloader = DataLoader(
        train_dataset,
        **dataloader_params,
        shuffle=True,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        **dataloader_params,
        shuffle=False,
    )

    return train_dataloader, valid_dataloader


class KoggClassifier(Dataset):
    def __init__(
        self,
        img_keys,
        annotation_df,
        img_dict,
        class_to_idx,
        transforms=None,
        dtype=torch.float16,
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
        self.dtype = dtype

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

        return image.to(self.dtype), target

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
        labels = torch.Tensor(
            [self.class_to_idx[shape["label"]] for shape in annotation["shapes"]]
        ).to(dtype=torch.int64)

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
