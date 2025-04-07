import gc
import json
import os

import lightning as L
import numpy as np
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.tv_tensors import BoundingBoxes, Mask

TRAIN_LOSS = "train_loss"
SEGM_MAP_50 = "segm_map_50"
BBOX_MAP_50 = "bbox_map_50"
KOG1_SEGM_MAP = "kog1_segm_map"
KOG1_BBOX_MAP = "kog1_bbox_map"
NORMAL_SEGM_MAP = "normal_segm_map"
NORMAL_BBOX_MAP = "normal_bbox_map"
IMG_SIZE = (1608, 1608)


# Callback to store losses and maps
class LossAndMapLogger(L.Callback):
    def __init__(self, epochs, train_dataloader_size, val_dataloader_size, version):
        super().__init__()
        self.current_train_idx = 0
        self.current_val_idx = 0
        self.cur_train_epoch = 0
        self.cur_val_epoch = 0
        self.train_dataloader_size = train_dataloader_size
        self.val_dataloader_size = val_dataloader_size
        self.loss_logger = np.zeros((epochs, train_dataloader_size))
        self.maps_logger_train = {
            SEGM_MAP_50: np.zeros((epochs, train_dataloader_size)),
            BBOX_MAP_50: np.zeros((epochs, train_dataloader_size)),
            KOG1_SEGM_MAP: np.zeros((epochs, train_dataloader_size)),
            NORMAL_SEGM_MAP: np.zeros((epochs, train_dataloader_size)),
            KOG1_BBOX_MAP: np.zeros((epochs, train_dataloader_size)),
            NORMAL_BBOX_MAP: np.zeros((epochs, train_dataloader_size)),
        }
        self.maps_logger_val = {
            SEGM_MAP_50: np.zeros((epochs, val_dataloader_size)),
            BBOX_MAP_50: np.zeros((epochs, val_dataloader_size)),
            KOG1_SEGM_MAP: np.zeros((epochs, val_dataloader_size)),
            NORMAL_SEGM_MAP: np.zeros((epochs, val_dataloader_size)),
            KOG1_BBOX_MAP: np.zeros((epochs, val_dataloader_size)),
            NORMAL_BBOX_MAP: np.zeros((epochs, val_dataloader_size)),
        }
        self.version = version

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (
            self.current_train_idx > 0
            and self.current_train_idx % self.train_dataloader_size == 0
        ):
            self.cur_train_epoch += 1
            self.current_train_idx = 0

        # Store losses
        loss = pl_module.loss
        self.loss_logger[self.cur_train_epoch][self.current_train_idx] = loss

        # Store maps
        self.update_maps(
            pl_module.maps_train,
            self.maps_logger_train,
            self.cur_train_epoch,
            self.current_train_idx,
        )
        self.current_train_idx += 1

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if (
            self.current_val_idx > 0
            and self.current_val_idx % self.val_dataloader_size == 0
        ):
            self.cur_val_epoch += 1
            self.current_val_idx = 0

        # Store maps
        self.update_maps(
            pl_module.maps_val,
            self.maps_logger_val,
            self.cur_val_epoch,
            self.current_val_idx,
        )

        self.current_val_idx += 1

    def update_maps(self, source_map, target_map, cur_epoch, cur_idx):
        for key in source_map.keys():
            item = source_map[key]
            if item is not None:
                # print
                target_map[key][cur_epoch][cur_idx] = item.item()
            else:
                prev_val = target_map[key][cur_epoch][cur_idx - 1] if cur_idx > 0 else 0
                target_map[key][cur_epoch][cur_idx] = prev_val
        return target_map

    def on_train_epoch_end(self, trainer, pl_module):
        if not self.cur_train_epoch % 10 == 0:
            return

        # Save losses as npy file
        np.save(f"output/{self.version}/losses.npy", self.loss_logger)

        train_map_copy, val_map_copy = {}, {}
        for keya, keyb in zip(
            self.maps_logger_train.keys(), self.maps_logger_val.keys()
        ):
            train_map_copy[keya] = self.maps_logger_train[keya].copy().tolist()
            val_map_copy[keyb] = self.maps_logger_val[keyb].copy().tolist()

        # save dicts as json file
        with open(f"output/{self.version}/maps_train.json", "w") as f:
            json.dump(train_map_copy, f)

        with open(f"output/{self.version}/maps_val.json", "w") as f:
            json.dump(val_map_copy, f)


class LightningMaskRCNNModel(L.LightningModule):
    def __init__(self, model, optimizer):
        super().__init__()
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss = None
        self.maps_train = {
            SEGM_MAP_50: None,
            BBOX_MAP_50: None,
            KOG1_SEGM_MAP: None,
            KOG1_BBOX_MAP: None,
            NORMAL_BBOX_MAP: None,
        }
        self.maps_val = {
            SEGM_MAP_50: None,
            BBOX_MAP_50: None,
            KOG1_SEGM_MAP: None,
            KOG1_BBOX_MAP: None,
            NORMAL_BBOX_MAP: None,
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        # send to device
        x = torch.stack(x).to(self.device)
        y = [
            {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in t.items()
            }
            for t in y
        ]

        # get pred
        self.model.eval()
        with torch.no_grad():
            pred = self.model(x)
            # get maps
            maskrcnn_map = MeanAveragePrecision(
                box_format="xyxy", iou_type=tuple(["bbox", "segm"]), class_metrics=True
            )
            self.get_map(
                maskrcnn_map=maskrcnn_map, model_output=pred, target_dict=y, train=True
            )

        # Get loss
        self.model.train()
        train_loss_dict = self.model(x, y)
        train_loss = sum(loss for loss in train_loss_dict.values())

        # log loss
        self.log(TRAIN_LOSS, train_loss, prog_bar=True, batch_size=len(batch))
        self.loss = train_loss.item()

        del x, y, train_loss_dict
        gc.collect()
        torch.cuda.empty_cache()

        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = torch.stack(x).to(self.device)
        y = [
            {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in t.items()
            }
            for t in y
        ]
        pred = self.model(x)

        maskrcnn_map = MeanAveragePrecision(
            box_format="xyxy", iou_type=tuple(["bbox", "segm"]), class_metrics=True
        )
        self.get_map(
            maskrcnn_map=maskrcnn_map, model_output=pred, target_dict=y, train=False
        )

        # log maps
        self.log_dict(
            {
                SEGM_MAP_50: self.maps_val[SEGM_MAP_50],
                BBOX_MAP_50: self.maps_val[BBOX_MAP_50],
            },
            prog_bar=True,
            batch_size=len(batch),
        )

        del x, y, pred
        gc.collect()
        torch.cuda.empty_cache()

        return self.maps_val[BBOX_MAP_50]

    def get_map(self, maskrcnn_map, model_output, target_dict, train=False):
        for i in range(len(model_output)):
            model_output[i]["masks"] = Mask(
                model_output[i]["masks"].type(torch.bool).squeeze(dim=1)
            )

            model_output[i]["boxes"] = BoundingBoxes(
                data=model_output[i]["boxes"], format="xyxy", canvas_size=IMG_SIZE
            )

        # update MAP
        maskrcnn_map.update(model_output, target_dict)

        # grab values
        bbox_map_per_class_val = (
            maskrcnn_map.compute()["bbox_map_per_class"].detach().cpu()
        )
        segm_map_per_class_val = (
            maskrcnn_map.compute()["segm_map_per_class"].detach().cpu()
        )
        bbox_map_50_val = maskrcnn_map.compute()[BBOX_MAP_50].detach().cpu()
        segm_map_50_val = maskrcnn_map.compute()[SEGM_MAP_50].detach().cpu()
        bbox_map_kog1_val, bbox_map_normal_val = (
            bbox_map_per_class_val
            if bbox_map_per_class_val.ndim != 0
            else (None, bbox_map_per_class_val)
        )
        segm_map_kog1_val, segm_map_normal_val = (
            segm_map_per_class_val
            if segm_map_per_class_val.ndim != 0
            else (None, segm_map_per_class_val)
        )

        # sanitize values
        bbox_map_kog1_val = bbox_map_kog1_val if bbox_map_kog1_val != -1 else None
        segm_map_kog1_val = segm_map_kog1_val if segm_map_kog1_val != -1 else None

        # save
        if train:
            self.maps_train[BBOX_MAP_50] = bbox_map_50_val
            self.maps_train[SEGM_MAP_50] = segm_map_50_val
            self.maps_train[KOG1_BBOX_MAP] = bbox_map_kog1_val
            self.maps_train[KOG1_SEGM_MAP] = segm_map_kog1_val
            self.maps_train[NORMAL_BBOX_MAP] = bbox_map_normal_val
            self.maps_train[NORMAL_SEGM_MAP] = segm_map_normal_val
        else:
            self.maps_val[BBOX_MAP_50] = bbox_map_50_val
            self.maps_val[SEGM_MAP_50] = segm_map_50_val
            self.maps_val[KOG1_BBOX_MAP] = bbox_map_kog1_val
            self.maps_val[KOG1_SEGM_MAP] = segm_map_kog1_val
            self.maps_val[NORMAL_BBOX_MAP] = bbox_map_normal_val
            self.maps_val[NORMAL_SEGM_MAP] = segm_map_normal_val

    def configure_optimizers(self):
        return self.optimizer
