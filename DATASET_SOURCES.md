# Dataset sources

## Coffee Bean Dataset Resized (224 x 224)

- Source: https://www.kaggle.com/datasets/gpiosenka/coffee-bean-dataset-resized-224-x-224
- Original research: https://arxiv.org/abs/2206.01841
- License: CC BY-SA 4.0
- Output classes: Light, Medium, Dark. For the requested combined task,
  Light includes both original Green (unroasted) and Light (light roast).
  Original source labels and folders remain unchanged.

Run `python scripts/prepare_merged_dataset.py` to prepare
`datasets/merged/train` and `datasets/merged/test` from the original four
classes. The notebook uses this merged dataset. Run
`python scripts/retrain_pipeline.py` to retrain both deployed models and
export their TensorFlow Lite files, then restart the app. Existing weights
do not implement the combined class until retraining finishes.

The downloaded source images are excluded from Git. Run
`scripts/generate_group_dataset.py` after placing the Kaggle dataset under
`datasets/train` and `datasets/test`. The generated multi-bean images retain
the same license and are also excluded from Git.

## Real multi-bean photographs

The source photographs are downloaded from Wikimedia Commons by
`scripts/download_real_group_images.py` and are excluded from Git. Crops are
kept in source-level train/validation/test splits so that no crop of the same
photograph appears in more than one split.

- Roast progression at 385, 400, 410, 425, 440, 450, 460, 470 and 480°F by
  Dan Bollinger: CC BY-SA 3.0. Source category:
  https://commons.wikimedia.org/wiki/Category:Coffee_roasting
- Medium roasted Arabica coffee beans by Sage Ross: CC BY-SA 3.0 / GFDL.
  https://commons.wikimedia.org/wiki/File:Medium_roasted_Arabica_coffee_beans.jpg
- Peaberry coffee beans, close up by Sage Ross: CC BY-SA 3.0 / GFDL.
  https://commons.wikimedia.org/wiki/File:Peaberry_coffee_beans,_close_up.jpg
- Maragogipe Coffee by Raminagrobis: CC BY-SA 4.0.
  https://commons.wikimedia.org/wiki/File:Maragogipe_Coffee.jpg
- Dark roasted espresso blend coffee beans 1 and 2 by Sage Ross:
  CC BY-SA 3.0 / GFDL.
  https://commons.wikimedia.org/wiki/File:Dark_roasted_espresso_blend_coffee_beans_1.jpg
  https://commons.wikimedia.org/wiki/File:Dark_roasted_espresso_blend_coffee_beans_2.jpg

For the three-class task, 385–410°F are mapped to Light, 425–440°F to Medium,
and 450–480°F to Dark. Derived crops and trained weights are distributed under
the compatible CC BY-SA 4.0 terms.
