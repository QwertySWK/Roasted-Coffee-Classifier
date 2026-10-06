"""Create reproducible multi-bean roast images from licensed bean crops."""

from __future__ import annotations

import argparse
import random
import shutil
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from scipy import ndimage


CLASSES = ("Light", "Medium", "Dark")
IMAGE_SIZE = 224


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("datasets"))
    parser.add_argument("--output", type=Path, default=Path("datasets/grouped"))
    parser.add_argument("--train-per-class", type=int, default=600)
    parser.add_argument("--val-per-class", type=int, default=150)
    parser.add_argument("--test-per-class", type=int, default=150)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--skip-groups", action="store_true", help="Only add held-out source images to each split"
    )
    return parser.parse_args()


def source_splits(source: Path, class_name: str, seed: int) -> dict[str, list[Path]]:
    train_files = sorted((source / "train" / class_name).glob("*.png"))
    test_files = sorted((source / "test" / class_name).glob("*.png"))
    if len(train_files) < 10 or len(test_files) < 10:
        raise FileNotFoundError(f"Not enough source images for {class_name}")

    rng = random.Random(f"{seed}-{class_name}")
    rng.shuffle(train_files)
    val_count = max(1, round(len(train_files) * 0.2))
    return {
        "train": train_files[val_count:],
        "val": train_files[:val_count],
        "test": test_files,
    }


@lru_cache(maxsize=1024)
def extract_bean(path: Path) -> Image.Image:
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    image_float = image.astype(np.float32) / 255.0
    value = image_float.max(axis=2)
    chroma = image_float.max(axis=2) - image_float.min(axis=2)

    # The source photographs have neutral paper backgrounds. Brown beans have
    # appreciably more chroma, while their darkest creases remain below the
    # value threshold. Keeping only the largest component removes paper noise.
    candidate = ((chroma > 0.10) & (value < 0.92)) | (value < 0.48)
    labels, component_count = ndimage.label(candidate)
    if component_count == 0:
        raise ValueError(f"Could not isolate bean in {path}")
    sizes = ndimage.sum(candidate, labels, range(1, component_count + 1))
    bean_component = labels == (int(np.argmax(sizes)) + 1)
    bean_component = ndimage.binary_fill_holes(bean_component)
    bean_component = ndimage.binary_dilation(bean_component, iterations=2)

    mask = Image.fromarray((bean_component * 255).astype(np.uint8))
    mask = mask.filter(ImageFilter.GaussianBlur(1.1))
    bbox = mask.getbbox()
    if bbox is None:
        raise ValueError(f"Could not isolate bean in {path}")

    rgba = Image.fromarray(image).convert("RGBA")
    rgba.putalpha(mask)
    return rgba.crop(bbox)


def make_background(rng: random.Random) -> Image.Image:
    base_colors = ((246, 246, 244), (232, 228, 219), (218, 211, 198), (205, 207, 203))
    base = np.array(rng.choice(base_colors), dtype=np.float32)
    yy, xx = np.mgrid[:IMAGE_SIZE, :IMAGE_SIZE]
    direction = rng.uniform(-8, 8) * (xx / IMAGE_SIZE) + rng.uniform(-8, 8) * (yy / IMAGE_SIZE)
    noise_rng = np.random.default_rng(rng.randrange(2**32))
    noise = noise_rng.normal(0, rng.uniform(1.0, 3.5), (IMAGE_SIZE, IMAGE_SIZE, 1))
    canvas = np.clip(base + direction[..., None] + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(canvas, "RGB").convert("RGBA")


def compose_group(bean_paths: list[Path], rng: random.Random) -> Image.Image:
    canvas = make_background(rng)
    bean_count = rng.randint(12, 28)

    for _ in range(bean_count):
        bean = extract_bean(rng.choice(bean_paths))
        bean = ImageEnhance.Brightness(bean).enhance(rng.uniform(0.82, 1.18))
        bean = ImageEnhance.Contrast(bean).enhance(rng.uniform(0.9, 1.12))

        target_width = rng.randint(42, 76)
        target_height = max(18, round(target_width * bean.height / bean.width))
        bean = bean.resize((target_width, target_height), Image.Resampling.LANCZOS)
        bean = bean.rotate(rng.uniform(0, 360), Image.Resampling.BICUBIC, expand=True)

        center_x = int(rng.gauss(IMAGE_SIZE / 2, IMAGE_SIZE * 0.22))
        center_y = int(rng.gauss(IMAGE_SIZE / 2, IMAGE_SIZE * 0.22))
        x = max(-bean.width // 3, min(IMAGE_SIZE - bean.width * 2 // 3, center_x - bean.width // 2))
        y = max(-bean.height // 3, min(IMAGE_SIZE - bean.height * 2 // 3, center_y - bean.height // 2))
        canvas.alpha_composite(bean, (x, y))

    image = canvas.convert("RGB")
    if rng.random() < 0.35:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.15, 0.55)))
    return image


def generate_split(paths: list[Path], destination: Path, count: int, seed: int) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    for index in range(count):
        output = destination / f"group_{index:04d}.jpg"
        compose_group(paths, rng).save(output, quality=92, subsampling=0)


def copy_source_images(paths: list[Path], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(paths):
        shutil.copy2(source, destination / f"single_{index:04d}{source.suffix.lower()}")


def main() -> None:
    args = parse_args()
    counts = {"train": args.train_per_class, "val": args.val_per_class, "test": args.test_per_class}

    for class_index, class_name in enumerate(CLASSES):
        splits = source_splits(args.source, class_name, args.seed)
        for split_index, split in enumerate(("train", "val", "test")):
            destination = args.output / split / class_name
            if not args.skip_groups:
                generate_split(
                    splits[split],
                    destination,
                    counts[split],
                    args.seed + class_index * 1000 + split_index * 100,
                )
            copy_source_images(splits[split], destination)
            print(
                f"{split}/{class_name}: {counts[split]} groups + "
                f"{len(splits[split])} source images"
            )


if __name__ == "__main__":
    main()
