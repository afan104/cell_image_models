import torch
from torchvision import transforms
from typing import List
from PIL import Image, ImageDraw


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
