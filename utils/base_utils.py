from typing import List

import torch
import torchvision.transforms.v2 as transforms
from PIL import Image, ImageDraw

"""
Functions in this File:
- tensor_to_pil: converts a tensor output into a pil image
- stack_imgs: stacks multiple images into one image
- append_json_filenames: TODO: need to fix and apply
- create_polygon_mask: creates a grayscale image with a white polygonal area on a black background
"""


OPS_HPS = {
    "v0": {"optimizer": torch.optim.SGD, "lr": 0.2},
    "v1": {
        "optimizer": torch.optim.SGD,
        "lr": 0.01,
        "momentum": 0.9,
        "weight_decay": 3e-4,
    },
    "v2": {
        "optimizer": torch.optim.Adam,
        "lr": 0.001,
        "betas": (0.9, 0.999),
        "weight_decay": 1e-3,
    },
}


def tensor_to_pil(tensor: torch.Tensor):
    """
    convert a PyTorch tensor to a PIL image
    :param tensor: PyTorch tensor of shape (C, H, W) or (B, C, H, W)
    :return: PIL image
    """
    # Remove batch dim, if present
    if len(tensor.shape) == 4:
        tensor.squeeze(0)

    # torchvision's built-in ToPILImage() transform
    return transforms.ToPILImage()(tensor)


def stack_imgs(imgs: List[Image.Image], text: List[str] = None, h_text=10):
    """
    Stack a list of PIL images vertically with text in between
    :param imgs: List of PIL images of identical dimensions
    :param text: List of text to display in between images
    :param h_text: height of text
    :return stack: PIL image
    """
    w, h = imgs[0].size
    if text == None:
        text = [f"IMAGE {str(i)}" for i in range(len(imgs))]

    # initialize blank image
    h_stack = (h + h_text) * len(imgs)
    stacked_img = Image.new("RGB", size=(w, h_stack))

    # paste in each image with text above
    for i, img in enumerate(imgs):
        ImageDraw.Draw(stacked_img).text(  # Image
            (0, i * (h + h_text)), text[i], (0, 0, 0)  # Coordinates  # Text  # Color
        )
        stacked_img.paste(img, (0, i * (h + h_text) + h_text))
    return stacked_img


# def append_json_filenames():
#     # Define the directory containing JSON files and the history file
#     json_directory = "./Data"
#     history_filepath = "./Data/annotation_tracker.json"
#     # Get list of all JSON files in the directory
#     json_files = [f for f in os.listdir(json_directory) if f.endswith(".json")]

#     # Append filenames to the history file
#     with open(history_filepath, "a") as file:
#         for json_file in json_files:
#             file.write(json_file[:-5] + "\n")


# convert sgementation polygons to images
def create_polygon_mask(image_size, vertices):
    """
    Create a grayscale image with a white polygonal area on a black background.

    Parameters:
    - image_size (tuple): A tuple representing the dimensions (width, height) of the image.
    - vertices (list): A list of tuples, each containing the x, y coordinates of a vertex
                        of the polygon. Vertices should be in clockwise or counter-clockwise order.

    Returns:
    - PIL.Image.Image: A PIL Image object containing the polygonal mask.
    """

    # Create a new black image with the given dimensions
    mask_img = Image.new("L", image_size, 0)

    # Draw the polygon on the image. The area inside the polygon will be white (255).
    ImageDraw.Draw(mask_img, "L").polygon(vertices, fill=(255))

    # Return the image with the drawn polygon
    return mask_img
