# CellSeg-MaskRCNN

A Mask R-CNN model for cell instance segmentation, built with PyTorch, torchvision, and PyTorch Lightning.

## What it does

Given microscopy images of cells, the model predicts a bounding box, class, and pixel mask for each individual cell instance.

## Structure

- `train_model.py`, `parallel_train.py`, `lightning_train.py` — training entry points (single-GPU, multi-GPU, and Lightning-managed runs)
- `utils/` — dataset loading and augmentation (`dataset_manipulation.py`), model definition (`model_classes.py`), training/eval loops (`train_test_utils.py`, `lightning_engine.py`), and visualization helpers
- `PS_Scripts/` — Photoshop scripts (`drawMasksToPS.js`, `psMasksToJSON.js`) used to hand-annotate cell masks and convert them into JSON training labels
- `MaskRCNNCells.ipynb` / `MaskRCNNCellsBrief.ipynb` — Colab-runnable notebooks covering data loading, training, and evaluation end to end

## Stack

PyTorch, torchvision, PyTorch Lightning, pycocotools, scikit-learn

## Data

Cell images are annotated by hand in Photoshop; the `PS_Scripts` convert the layer-based masks into per-instance JSON annotations (`Data/annotation_tracker.json`, `Data/format.json`) consumed by the training pipeline.
