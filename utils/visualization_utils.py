import os
import random
from functools import partial

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms.v2 as transforms
from PIL import Image
from torchvision.tv_tensors import BoundingBoxes, Mask
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

from utils.base_utils import create_polygon_mask, stack_imgs, tensor_to_pil

"""
Functions in this File
- visualize_pred: Performs forward pass on one image, saves result as an annotated png file
- visualize_multi: Performs visualize_pred on multiple images (randomly from test or list of input files)
- visualize_loss: Visualizes training losses as a png file
- visualize_map: Visualizes test accuracies as a png file
"""


def visualize_pred(
    model,
    annotation_df,
    img_dict,
    int_colors,
    class_names,
    device,
    file_id,
    test_img="",
    folder="output",
):
    """
    Parameters -
    :model - pytorch model
    :img_dict - dict of name, path pairs
    :file_id - string
    :test_img - optional image object, default is the original file_id image
    """
    # grab image
    path = img_dict[file_id]
    test_img = test_img if test_img != "" else Image.open(path).convert("RGB")

    # grab target annotations
    target_shape_points = [
        shape["points"] for shape in annotation_df.loc[file_id]["shapes"]
    ]
    target_all_shapes = [[tuple(p) for p in points] for points in target_shape_points]
    target_mask_imgs = [
        create_polygon_mask(test_img.size, shape) for shape in target_all_shapes
    ]
    target_masks = Mask(
        torch.concat(
            [
                Mask(transforms.PILToTensor()(mask), dtype=torch.bool)
                for mask in target_mask_imgs
            ]
        )
    )

    target_labels = [shape["label"] for shape in annotation_df.loc[file_id]["shapes"]]
    target_bboxes = BoundingBoxes(
        data=torchvision.ops.masks_to_boxes(target_masks),
        format="xyxy",
        canvas_size=test_img.size[::-1],
    )

    # model forward pass
    model.eval()
    model.to(device)
    input_tensor = transforms.Compose(
        [transforms.ToImage(), transforms.ToDtype(torch.float16, scale=True)]
    )(test_img)[None].to(device)
    with torch.inference_mode():
        model_output = model(input_tensor)

    # filter output
    threshold = 0.05
    scores_mask = model_output[0]["scores"] > threshold

    # get output bboxes and labels
    pred_bboxes = BoundingBoxes(
        model_output[0]["boxes"][scores_mask],
        format="xyxy",
        canvas_size=test_img.size[::-1],
    )
    pred_labels = [
        class_names[int(label)] for label in model_output[0]["labels"][scores_mask]
    ]

    # get output masks
    pred_scores = model_output[0]["scores"]
    pred_masks_filtered = model_output[0]["masks"][scores_mask]
    pred_masks = [
        Mask(torch.where(mask >= threshold, 1, 0), dtype=torch.bool)
        for mask in pred_masks_filtered
    ]

    # check if masks are empty
    if len(pred_masks) != 0:
        pred_masks = torch.concat(pred_masks)
    else:
        pred_masks = None

    # colors
    target_colors = [
        int_colors[i] for i in [class_names.index(label) for label in target_labels]
    ]
    pred_colors = [
        int_colors[i] for i in [class_names.index(label) for label in pred_labels]
    ]

    # convert image to tensor
    img_tensor = transforms.PILToTensor()(test_img)

    # annotate image with GT masks and bounding boxes
    draw_bboxes = partial(
        draw_bounding_boxes,
        fill=False,
        width=2,
        font="KFOlCnqEu92Fr1MmEU9vAw.ttf",
        font_size=12,
    )
    annotated_tensor_truemask = draw_segmentation_masks(
        image=img_tensor, masks=target_masks, alpha=0.3, colors=target_colors
    )
    annotated_tensor_truemaskbox = draw_bboxes(
        image=annotated_tensor_truemask,
        boxes=target_bboxes,
        labels=target_labels,
        colors=target_colors,
    )
    gt_annotated_test_img = tensor_to_pil(annotated_tensor_truemaskbox)

    # annotate image with prediction masks and bounding boxes
    if pred_masks != None:
        annotated_tensor_predmask = draw_segmentation_masks(
            image=img_tensor, masks=pred_masks, alpha=0.3, colors=pred_colors
        )
        annotated_tensor_predmaskbox = draw_bboxes(
            image=annotated_tensor_predmask,
            boxes=pred_bboxes,
            labels=[
                f"{label}\n{prob*100:.2f}%"
                for label, prob in zip(pred_labels, pred_scores)
            ],
            colors=pred_colors,
        )
        pred_annotated_test_img = tensor_to_pil(annotated_tensor_predmaskbox)
    else:
        to_pil = transforms.ToPILImage()
        pred_annotated_test_img = to_pil(img_tensor)

    # visualize
    stacked_img = stack_imgs([gt_annotated_test_img, pred_annotated_test_img])
    plt.figure(figsize=(10, 5))
    plt.imshow(stacked_img)
    plt.axis("off")  # Hide axes
    plt.tight_layout()

    # summary print statements
    target_bboxes_text = ", ".join(
        [f"{label}: {bbox}" for label, bbox in zip(target_labels, target_bboxes)]
    )
    pred_bboxes_text = ", ".join(
        [f"{label}: {bbox}" for label, bbox in zip(pred_labels, pred_bboxes)]
    )
    confidence_scores = ", ".join(
        [f"{label}:{score*100:.2f}%" for label, score in zip(pred_labels, pred_scores)]
    )
    # print statements
    print(f"Target BBoxes: {target_bboxes_text}")
    print(f"Predicted BBoxes: {pred_bboxes_text}")
    print(f"Confidence Scores: {confidence_scores}")
    plt.savefig(f"{folder}/{file_id}_annotated.png", dpi=300)


def visualize_multi(
    model,
    annotation_df,
    img_dict,
    int_colors,
    class_names,
    device,
    file_ids=[],
    n=None,
    folder="output",
):
    """
    Visualizes the prediction for n images or a list of files
    Always includes file id: fz_173_t0_Nstarve_1

    If n is not none, will visualize n - 1 additional predictions
    """
    visualize_pred(
        model=model,
        annotation_df=annotation_df,
        img_dict=img_dict,
        int_colors=int_colors,
        class_names=class_names,
        device=device,
        file_id="fz_173_t0_Nstarve_1",
        folder=folder,
    )
    if len(file_ids) != 0:
        for file_id in file_ids:
            visualize_pred(
                model=model,
                annotation_df=annotation_df,
                img_dict=img_dict,
                int_colors=int_colors,
                class_names=class_names,
                device=device,
                file_id=file_id,
                folder=folder,
            )
    elif n is not None:
        # randomly select n - 1 files from the test set
        current_dir = os.path.dirname(os.path.abspath(__file__))
        test_dir = os.path.join(current_dir, "..", "Data", "test")
        all_test_files = os.listdir(test_dir)
        all_test_ids = [file.split(".")[0] for file in all_test_files]
        random_test_ids = random.sample(all_test_ids, n)

        for file_id in random_test_ids:
            visualize_pred(
                model=model,
                annotation_df=annotation_df,
                img_dict=img_dict,
                int_colors=int_colors,
                class_names=class_names,
                device=device,
                file_id=file_id,
                folder=folder,
            )
    else:
        print("please input value for either 'file_ids' or 'n'")


def visualize_losses(data, file_name):
    time_steps = np.arange(data.shape[1])  # Time steps (x-axis)

    # Compute averages for each time step
    averages = np.mean(data, axis=1)

    # Create figure with two subplots
    _, axes = plt.subplots(2, 1, figsize=(10, 8))

    # ---- First Plot: Individual Lists Over Time ----
    num_series = len(data)
    colors = [
        cm.viridis(i / num_series) for i in range(num_series)
    ]  # Use 'viridis' colormap
    for i, series in enumerate(data):
        axes[0].plot(
            time_steps,
            series,
            marker="o",
            linestyle="-",
            color=colors[i],
            label=f"Epoch {i}",
        )

    axes[0].set_title("Losses At Each Step (Epochs separate)")
    axes[0].set_xlabel("Steps")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid()

    # ---- Second Plot: Average Over Time ----
    axes[1].plot(
        np.arange(data.shape[0]),
        averages,
        marker="s",
        color="red",
        linestyle="-",
        linewidth=2,
        label="Average",
    )

    axes[1].set_title("Average Loss Over Epochs")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Average Loss")
    axes[1].legend()
    axes[1].grid()

    # Adjust layout and save as image
    plt.tight_layout()
    plt.savefig(file_name, dpi=300)  # High-resolution image


def visualize_maps(maps, file_name):

    epochs = maps["segm_map_50"].shape[0]  # Number of epochs
    steps_per_epoch = maps["segm_map_50"].shape[1]  # Number of steps per epoch

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ---- Plot 1: Class-wide MAP 50s ----
    x_epochs = np.arange(epochs)
    axes[0].plot(
        x_epochs,
        np.mean(maps["segm_map_50"], axis=1),
        marker="o",
        linestyle="-",
        label="Segmentation MAP 50",
    )
    axes[0].plot(
        x_epochs,
        np.mean(maps["bbox_map_50"], axis=1),
        marker="s",
        linestyle="--",
        label="BBox MAP 50",
    )
    axes[0].set_title("Class-wide MAP over Epochs")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("MAP 50")
    axes[0].legend()

    # ---- Plot 2: Kog1 v normal MAP Values ----
    colors = [cm.viridis(i / (2)) for i in range(2)]
    # kog1 mean
    a = np.mean(maps["kog1_segm_map"], 1)
    b = np.mean(maps["kog1_bbox_map"], 1)
    axes[1].plot(
        x_epochs,
        np.mean([a, b], axis=0),
        marker="o",
        linestyle="-",
        color=colors[0],
        label="kog1 MAP",
    )
    # bbox mean
    c = np.mean(maps["normal_segm_map"], 1)
    d = np.mean(maps["normal_bbox_map"], 1)
    axes[1].plot(
        x_epochs,
        np.mean([c, d], axis=0),
        marker="s",
        linestyle="-",
        color=colors[1],
        label=f"normal MAP",
    )

    axes[1].set_title("Kog1 vs Normal MAP")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("MAP")
    axes[1].legend()

    # ---- Plot 3: Batch MAP per Class ----
    colors2 = [cm.viridis(i / (epochs)) for i in range(epochs)]

    for i in range(epochs):
        # Plot class 0 bbox MAP per batch
        a = maps["kog1_bbox_map"][i]
        b = maps["normal_bbox_map"][i]
        axes[2].plot(
            np.arange(steps_per_epoch),
            np.mean([a, b], axis=0),
            marker="s",
            linestyle="--",
            color=colors2[i],
            label=f"Epoch {i}",
        )
    axes[2].set_title("BBOX MAP per Batch")
    axes[2].set_xlabel("Steps")
    axes[2].set_ylabel("MAP")
    axes[2].legend()

    plt.tight_layout
    plt.savefig(file_name, dpi=300)  # High-resolution image()
