# Data Notes

Use public data only.

Do not use:

- Private employer images.
- Receipt images.
- Customer data.
- Internal POS screens.
- Store inventory exports.
- Any confidential workplace files.

## Baseline Dataset Strategy

Start with a clean public produce image dataset to validate the pipeline. Treat this as a warm-up benchmark, not as proof of real checkout performance.

After the baseline works, add or evaluate against more realistic grocery-style images where possible:

- hands in frame
- bags or stickers
- shadows and poor lighting
- multiple items
- partial views
- produce varieties that look similar

## Expected Raw Layout

Place image folders under:

```text
data/raw/imagefolder/
  apple/
    image_001.jpg
  banana/
    image_001.jpg
```

Then create manifests with:

```bash
python3 scripts/make_dataset_manifest.py --input data/raw/imagefolder --output data/processed/manifest.csv
```

