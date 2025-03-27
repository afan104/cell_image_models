import gc
from pathlib import Path

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.tv_tensors import BoundingBoxes, Mask
from tqdm import tqdm

"""
Functions in this File:
- get_map: returns multiple bbox and segm mean average precision values as a list
- train_one_epoch: trains for one epoch and adds the loss to the loss_logger before returning it
- test_one_epoch: tests for one epoch and adds the map value to the map_logger before returning it
- save_model: saves a model state dict to a given path
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


def get_optim(optim_info, model_params):
    optim_params = {k: v for k, v in optim_info.items() if k != "optimizer"}
    return optim_info["optimizer"](model_params, **optim_params)


def get_map(maskrcnn_map, model_output, target_dict):
    """
    Parameters -
    :maskrcnn_map ( MeanAveragePrecision instance )
    :model_output ( List[Dict] with 1 item in List )
    :target_dict  ( List[Dict] with 1 item in List)

    Returns -
    List[segm mean avg precision iou 0.5, bbox mean avg precision iou 0.5, bbox_per_class, segm_per_class]
    """
    # reshape predictions for map calculation
    img_size = (1608, 1608)
    # check shape of all items:
    for i in range(len(model_output)):
        model_output[i]["masks"] = Mask(
            model_output[i]["masks"].type(torch.bool).squeeze(dim=1)
        )

        model_output[i]["boxes"] = BoundingBoxes(
            data=model_output[i]["boxes"], format="xyxy", canvas_size=img_size
        )

    maskrcnn_map.update(model_output, target_dict)
    print(
        f"bbox map50: {maskrcnn_map.compute()['bbox_map_50']}",
        f"segm map50: {maskrcnn_map.compute()['segm_map_50']}",
    )
    return [
        maskrcnn_map.compute()["bbox_map_50"],
        maskrcnn_map.compute()["segm_map_50"],
        maskrcnn_map.compute()["bbox_map_per_class"],
        maskrcnn_map.compute()["segm_map_per_class"],
    ]


def train_one_epoch(model, dataloader, device, optimizer, scaler=None):
    """ """
    # train
    loss_logger = []
    model.train()

    # loop over data
    for _, (inputs, targets) in tqdm(
        enumerate(dataloader), total=len(dataloader), desc="Training"
    ):
        inputs = torch.stack(inputs).to(device)
        targets = [
            {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in t.items()
            }
            for t in targets
        ]

        # forward pass & loss
        if scaler is not None:
            with torch.amp.autocast():
                loss_dict = model(inputs, targets)  # train loss format
                losses = sum(loss for loss in loss_dict.values())
        else:
            loss_dict = model(inputs, targets)  # train loss format
            losses = sum(loss for loss in loss_dict.values())

        # optimizer 0 grad
        optimizer.zero_grad()

        # backprop and optimizer step
        if scaler is not None:
            pass
        else:
            losses.backward()
            optimizer.step()

        loss_logger.append(losses)
        tqdm.write(f"training losses: {losses}")
        del inputs, targets, loss_dict
        torch.cuda.empty_cache()
        gc.collect()
    return loss_logger


def test_one_epoch(model, dataloader, device, maps_logger):
    """
    Parameters -
    :model (pyTorch model)
    :dataloader (batched data)
    :device (GPU or CPU)

    Returns -
    loss
    accuracy
    """
    # eval
    maps_logger_per_epoch = {
        "segm_map_50": [],
        "bbox_map_50": [],
        "kog1_segm_map": [],
        "normal_segm_map": [],
        "kog1_bbox_map": [],
        "normal_bbox_map": [],
    }
    model.eval()

    # loop over data
    with torch.inference_mode():
        for _, (inputs_raw, targets) in tqdm(
            enumerate(dataloader), total=len(dataloader), desc="Testing"
        ):
            inputs = torch.stack(inputs_raw).to(device)
            targets = [
                {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()
                }
                for t in targets
            ]

            # forward pass
            pred = model(inputs)

            # accuracy
            maskrcnn_map = MeanAveragePrecision(
                box_format="xyxy", iou_type=tuple(["bbox", "segm"]), class_metrics=True
            )

            bbox_map_50, segm_map_50, bbox_map_per_class, segm_map_per_class = get_map(
                maskrcnn_map=maskrcnn_map, model_output=pred, target_dict=targets
            )

            maps_logger_per_epoch["bbox_map_50"].append(bbox_map_50)
            maps_logger_per_epoch["segm_map_50"].append(segm_map_50)

            # map per class: when there are no kogg class values, impute the previous mAP for continuity
            if segm_map_per_class.ndim != 0:
                maps_logger_per_epoch["normal_segm_map"].append(segm_map_per_class[1])
                maps_logger_per_epoch["normal_bbox_map"].append(bbox_map_per_class[1])
                maps_logger_per_epoch["kog1_segm_map"].append(segm_map_per_class[0])
                maps_logger_per_epoch["kog1_bbox_map"].append(bbox_map_per_class[0])
            else:
                maps_logger_per_epoch["normal_segm_map"].append(segm_map_per_class)
                maps_logger_per_epoch["normal_bbox_map"].append(bbox_map_per_class)
                maps_logger_per_epoch["kog1_segm_map"].append(
                    maps_logger_per_epoch["kog1_segm_map"][-1]
                    if len(maps_logger_per_epoch["kog1_segm_map"]) > 0
                    else 0
                )
                maps_logger_per_epoch["kog1_bbox_map"].append(
                    maps_logger_per_epoch["kog1_bbox_map"][-1]
                    if len(maps_logger_per_epoch["kog1_bbox_map"]) > 0
                    else 0
                )

        tqdm.write(f"test accuracy: {maps_logger_per_epoch.items()}")
        # Append per-epoch logs to the main logger
        for key in maps_logger.keys():
            maps_logger[key].append(maps_logger_per_epoch[key])
        return maps_logger


def save_model(model):
    # 1. Create models directory
    MODEL_PATH = Path("save_models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    # 2. Create model save path
    MODEL_NAME = "mp.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    # 3. Save model state dict
    print(f"Saving model to: {MODEL_SAVE_PATH}")
    torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)
