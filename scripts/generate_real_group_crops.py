"""Generate augmented crops while preserving source-level train/val/test splits."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


CLASSES = ("Light", "Medium", "Dark")
IMAGE_SIZE = 224


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("datasets/real-groups"))
    parser.add_argument("--single-source", type=Path, default=Path("datasets"))
    parser.add_argument("--output", type=Path, default=Path("datasets/real-group-crops"))
    parser.add_argument("--train-per-class", type=int, default=720)
    parser.add_argument("--eval-per-class", type=int, default=240)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def single_source_splits(source: Path, class_name: str, seed: int) -> dict[str, list[Path]]:
    train_files = sorted((source / "train" / class_name).glob("*.png"))
    test_files = sorted((source / "test" / class_name).glob("*.png"))
    rng = random.Random(f"{seed}-{class_name}")
    rng.shuffle(train_files)
    val_count = max(1, round(len(train_files) * 0.2))
    return {
        "train": train_files[val_count:],
        "val": train_files[:val_count],
        "test": test_files,
    }


def random_crop(image: Image.Image, rng: random.Random, training: bool) -> Image.Image:
    width, height = image.size
    short_side = min(width, height)
    scale = rng.uniform(0.38, 0.82) if training else rng.uniform(0.48, 0.78)
    crop_size = max(IMAGE_SIZE, round(short_side * scale))
    left = rng.randint(0, max(0, width - crop_size))
    top = rng.randint(0, max(0, height - crop_size))
    crop = image.crop((left, top, left + crop_size, top + crop_size))
    crop = crop.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)

    if training:
        if rng.random() < 0.5:
            crop = ImageOps.mirror(crop)
        crop = crop.rotate(rng.uniform(-12, 12), Image.Resampling.BICUBIC)
        crop = ImageEnhance.Brightness(crop).enhance(rng.uniform(0.82, 1.18))
        crop = ImageEnhance.Contrast(crop).enhance(rng.uniform(0.88, 1.14))
        crop = ImageEnhance.Color(crop).enhance(rng.uniform(0.88, 1.12))
        if rng.random() < 0.18:
            crop = crop.filter(ImageFilter.GaussianBlur(rng.uniform(0.1, 0.55)))
    return crop


def image_files(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def main() -> None:
    args = parse_args()
    for class_index, class_name in enumerate(CLASSES):
        singles = single_source_splits(args.single_source, class_name, args.seed)
        for split_index, split in enumerate(("train", "val", "test")):
            sources = image_files(args.source / split / class_name)
            if not sources:
                raise FileNotFoundError(f"No source photos in {args.source / split / class_name}")
            count = args.train_per_class if split == "train" else args.eval_per_class
            destination = args.output / split / class_name
            destination.mkdir(parents=True, exist_ok=True)
            rng = random.Random(args.seed + class_index * 1000 + split_index * 100)
            images = []
            for source in sources:
                with Image.open(source) as opened:
                    images.append(ImageOps.exif_transpose(opened).convert("RGB"))
            for crop_index in range(count):
                source_index = crop_index % len(images)
                crop = random_crop(images[source_index], rng, training=split == "train")
                filename = f"source_{source_index:02d}_crop_{crop_index:04d}.jpg"
                crop.save(destination / filename, quality=92, subsampling=0)
            for single_index, source in enumerate(singles[split]):
                shutil.copy2(
                    source,
                    destination / f"single_{single_index:04d}{source.suffix.lower()}",
                )
            print(
                f"{split}/{class_name}: {len(sources)} group sources, {count} crops, "
                f"{len(singles[split])} single-bean images"
            )


if __name__ == "__main__":
    main()
