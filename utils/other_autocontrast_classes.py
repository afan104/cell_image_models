from multiprocessing import Pool, cpu_count
from pathlib import Path

import torch
import torchvision.transforms.v2 as transforms
from PIL import Image
from PIL.ImageOps import autocontrast
from torchvision.transforms.functional import adjust_contrast, to_pil_image


class AutoContraster:
    def apply_pillow_autocontrast(self, img):
        return autocontrast(img, cutoff=1, preserve_tone=False)

    def apply_torch_adjustcontrast(self, img):
        return adjust_contrast(img, 3)  # 3 is the contrast factor


class ManualContraster:
    def apply_autocontrast(self, img):
        hist = img.histogram()
        cdf = [sum(hist[: i + 1]) for i in range(256)]
        clamp_percent = 0.55 * cdf[-1]
        cutoff = next((i for i in range(256) if cdf[i] >= clamp_percent), 0)

        img = img.point(lambda p: p if p > cutoff else 0)

        transform = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float16, scale=False),
            ]
        )
        img_tensor = transform(img)

        min_val, max_val = torch.min(img_tensor), torch.max(img_tensor)
        scale = 255.0 / (max_val - min_val)
        img_tensor = torch.clamp((img_tensor - min_val) * scale, 0, 255)

        return to_pil_image(img_tensor)


def process_image(img_path):
    img = Image.open(img_path).convert("L")
    img_name = img_path.name

    auto_contraster = AutoContraster()
    manual_contraster = ManualContraster()

    save_dirs = {
        "pil": img_path.parent / "pil_autocontrast",
        "torch": img_path.parent / "torch_autocontrast",
        "manual": img_path.parent / "manual_autocontrast",
    }
    for save_dir in save_dirs.values():
        save_dir.mkdir(parents=True, exist_ok=True)

    auto_contraster.apply_pillow_autocontrast(img).save(save_dirs["pil"] / img_name)
    auto_contraster.apply_torch_adjustcontrast(img).save(save_dirs["torch"] / img_name)
    manual_contraster.apply_autocontrast(img).save(save_dirs["manual"] / img_name)


if __name__ == "__main__":
    repo_dir = Path(__file__).resolve().parent.parent
    img_dir = repo_dir / "Data"
    images = [img_path for img_path in img_dir.glob("fz*.png")]

    num_workers = min(cpu_count(), 4)
    with Pool(num_workers) as pool:
        pool.map(process_image, images)
