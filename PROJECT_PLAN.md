# CashierVision Project Plan

## Current Status

Phase 1 is complete as a scaffold:

- The project has a clear cashier-assist framing.
- The MVP class list exists in `configs/classes_mvp.txt`.
- The example PLU mapping exists in `data/plu_mapping.csv`.
- The PLU validation script and unit tests pass.
- The Streamlit app can already demonstrate the PLU lookup layer.

The project is not yet a trained ML system. The next priority is to build a reproducible data and baseline-model pipeline.

## Positioning

CashierVision should be presented as:

> A top-k produce recognition and PLU suggestion system for grocery checkout, inspired by cashier experience.

Avoid claiming:

- Fully automated checkout.
- Perfect PLU identification.
- Real-store deployment readiness.
- Performance on private or employer data.

## Roadmap Overview

1. Foundation and framing.
2. Public data setup.
3. Baseline classifier.
4. Evaluation and error analysis.
5. PLU-assisted inference.
6. Streamlit portfolio demo.
7. Model card and GitHub polish.

Each phase should produce a concrete artifact before moving on.

## Phase 1: Foundation

- Create project structure.
- Define initial produce classes.
- Create a documented PLU assist table.
- Add validation checks for the PLU table.
- Write the README and limitations up front.

Deliverable: a clean GitHub-ready scaffold with a defensible project framing.

Exit criteria:

- `python3 scripts/validate_plu_mapping.py` passes.
- `python3 -m unittest discover -s tests` passes.
- README explains what the project is and what it is not.

## Phase 2: Data

- Start with a public clean dataset for baseline training.
- Prefer common produce classes that map clearly to images.
- Create a manifest with `filepath`, `label`, and `split`.
- Keep all private workplace images out of the project.

Deliverable: reproducible train/validation/test manifests.

Recommended first dataset:

- Use Fruits-360 as the clean baseline dataset.
- Treat it as a warm-up benchmark, not proof of checkout readiness.
- Map only the matching MVP classes into the project class names.

Steps:

1. Download the public dataset outside Git tracking.
2. Arrange selected images in ImageFolder layout:

```text
data/raw/imagefolder/
  apple/
  banana/
  lemon/
```

3. Generate the first manifest:

```bash
python3 scripts/make_dataset_manifest.py \
  --input data/raw/imagefolder \
  --output data/processed/manifest.csv
```

4. Add a short data audit notebook or script that reports:
   - number of classes
   - images per class
   - train/validation/test counts
   - missing or unexpected labels

Exit criteria:

- `data/processed/manifest.csv` exists.
- All labels in the manifest match `configs/classes_mvp.txt`.
- Class imbalance is documented.
- No private or employer data is used.

## Phase 3: Baseline Model

- Train a transfer-learning image classifier.
- Report top-1 and top-3 accuracy.
- Save confusion matrix and most confused classes.
- Identify classes where visual ambiguity is expected.

Deliverable: baseline model plus evaluation notebook.

Recommended model:

- Start with a pretrained `torchvision` model such as ResNet18 or EfficientNet-B0.
- Freeze most layers for the first run.
- Train only enough to prove the pipeline works.

Steps:

1. Create a reusable dataset loader from `data/processed/manifest.csv`.
2. Train a baseline model with image augmentations.
3. Save the model checkpoint to `models/`.
4. Save class-to-index metadata with the model.
5. Report top-1 and top-3 accuracy.

Exit criteria:

- A model checkpoint exists in `models/`.
- Training can be reproduced from a script or notebook.
- Metrics are saved in `reports/`.
- The baseline is described honestly as a clean-dataset result.

## Phase 4: Evaluation and Error Analysis

- Generate a confusion matrix.
- Identify the most confused produce classes.
- Compare expected ambiguity against model mistakes.
- Save example failure cases for the portfolio writeup.

Deliverable: evaluation notebook plus figures.

Steps:

1. Evaluate on the held-out test split.
2. Save `reports/figures/confusion_matrix.png`.
3. Save a table of the top confused class pairs.
4. Review whether mistakes are visually reasonable.

Exit criteria:

- Top-1 and top-3 metrics are reported.
- Confusion matrix exists.
- Limitations are tied to actual observed errors.

## Phase 5: PLU Assist Layer

- Convert model predictions into class suggestions.
- Attach possible PLU codes from a curated mapping table.
- Show ambiguity clearly when multiple PLUs may apply.
- Add confidence threshold behavior:
  - high confidence: show top suggestion first
  - medium confidence: show top 3
  - low confidence: ask cashier to verify manually

Deliverable: reusable inference helper and tested PLU lookup.

Steps:

1. Load the trained model and class metadata.
2. Return top-k produce predictions with confidence scores.
3. Attach PLU suggestions for each predicted class.
4. Show a fallback message when the class has no clean PLU mapping.
5. Add tests for confidence-message behavior and PLU formatting.

Exit criteria:

- Model predictions and PLU suggestions are combined in one reusable function.
- Low-confidence predictions clearly ask for manual verification.
- PLU output never claims guaranteed correctness.

## Phase 6: Demo

- Build a Streamlit app with image upload.
- Show top-k produce predictions.
- Show possible PLU codes with notes.
- Include visible uncertainty and limitations.

Deliverable: portfolio demo.

Steps:

1. Replace the placeholder upload message with real model inference.
2. Show uploaded image, top-3 predictions, confidence, and possible PLUs.
3. Use clear visual states for high, medium, and low confidence.
4. Include a small limitations note directly in the app.
5. Add demo screenshots to `reports/figures/`.

Exit criteria:

- Streamlit app runs locally.
- A user can upload an image and see model-backed suggestions.
- The demo feels assistive, not overconfident.

## Phase 7: Portfolio Polish

- Add screenshots.
- Add model card.
- Add limitations and future work.
- Explain why the project matters from cashier experience.

Deliverable: complete portfolio project page.

Steps:

1. Add `MODEL_CARD.md`.
2. Add a final results section to the README.
3. Include dataset description, metrics, confusion analysis, and limitations.
4. Add screenshots or GIFs of the Streamlit demo.
5. Make sure GitHub excludes raw data, trained weights if too large, caches, and private files.

Exit criteria:

- README tells the project story clearly.
- Model card explains intended use and non-use.
- Results are reproducible.
- The project is safe to show publicly.

## Immediate Next Steps

Start here:

1. Confirm the final MVP class list.
2. Download or prepare a public baseline dataset.
3. Put selected images into `data/raw/imagefolder/`.
4. Run `scripts/make_dataset_manifest.py`.
5. Add a data audit step before training.

Do not train the model until the manifest and class mapping are clean.
