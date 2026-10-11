"""Copy Green and Light into Light without changing the original dataset."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

CLASS_SOURCES = {'Light': ('Green', 'Light'), 'Medium': ('Medium',), 'Dark': ('Dark',)}


def prepare(source: Path, output: Path) -> None:
    output_path = output.resolve()
    if output_path == source.resolve() or any(
        output_path == (source / split).resolve()
        or (source / split).resolve() in output_path.parents
        or output_path in (source / split).resolve().parents
        for split in ('train', 'test')
    ):
        raise ValueError('Output must be separate from the original train/test splits')
    planned = []
    for split in ('train', 'test'):
        for target, originals in CLASS_SOURCES.items():
            for original in originals:
                folder = source / split / original
                files = sorted(p for p in folder.glob('*')
                               if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg'})
                if len(files) < 10:
                    raise ValueError(f'Need at least 10 source images in {folder}; found {len(files)}')
                for path in files:
                    destination = output / split / target / f'{original.lower()}__{path.name}'
                    planned.append((path, destination))
    # Validate every input before writing any output.
    for path, destination in planned:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    print(f'Prepared {len(planned)} images: Light = Green + Light, Medium, Dark')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('datasets'))
    parser.add_argument('--output', type=Path, default=Path('datasets/merged'))
    args = parser.parse_args()
    prepare(args.source, args.output)


if __name__ == '__main__':
    main()
