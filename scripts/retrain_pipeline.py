"""Rebuild both deployed models and select ensemble weights on validation."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    required = [ROOT / 'datasets' / split / name
                for split in ('train', 'test') for name in ('Green', 'Light', 'Medium', 'Dark')]
    required += [ROOT / 'datasets/real-groups' / split / name
                 for split in ('train', 'val', 'test') for name in ('Light', 'Medium', 'Dark')]
    missing = [str(path.relative_to(ROOT)) for path in required
               if not path.is_dir() or not any(path.iterdir())]
    if missing:
        raise SystemExit('Missing source images:\n' + '\n'.join(missing))
    commands = [
        ['scripts/prepare_merged_dataset.py'],
        ['scripts/generate_group_dataset.py', '--source', 'datasets/merged'],
        ['scripts/generate_real_group_crops.py', '--single-source', 'datasets/merged'],
        ['scripts/train_group_model.py'],
        ['scripts/train_group_model.py', '--data', 'datasets/real-group-crops',
         '--initial-model', 'src/coffee_roast_group_model.keras',
         '--model', 'src/coffee_roast_real_group_model.keras',
         '--tflite', 'src/coffee_roast_real_group_model_quantized.tflite',
         '--report', 'reports/real_group_model_metrics.json'],
        ['scripts/evaluate_ensemble.py', 'src/coffee_roast_group_model.keras',
         'src/coffee_roast_real_group_model.keras', '--single-data', 'datasets/merged/test'],
    ]
    for command in commands:
        subprocess.run([sys.executable, *command], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
