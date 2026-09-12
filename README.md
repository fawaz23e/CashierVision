# CashierVision

CashierVision is a cashier-assist machine learning project for grocery checkout. The goal is to recognize produce from an image and return top-k produce suggestions, with possible PLU codes where the mapping is clean enough to document.

This is not a fully automated checkout system. The intended framing is practical assistance: help a cashier narrow down likely produce items, reduce lookup friction, and surface ambiguity when a PLU depends on variety, size, organic status, or retailer policy.

## Why This Project

The project is inspired by cashier work. Produce PLU lookup can be stressful because many items look similar and there are many codes to remember. CashierVision turns that real workflow pain point into a portfolio ML project with observed image labels, top-k predictions, and an explicit PLU-assist layer.

## MVP Scope

- 20 to 40 common produce classes.
- Baseline image classifier trained on public image data.
- Top-k predictions with confidence scores.
- Curated PLU suggestion table for common classes.
- Streamlit demo for image upload and cashier-style suggestions.
- Model card and limitations section before portfolio publishing.

## Honest Limitations

- PLU codes can vary by retailer, region, variety, item size, and organic status.
- A clean benchmark dataset is not proof of checkout performance.
- Real checkout images include hands, bags, blur, poor lighting, shadows, partial views, and multiple items.
- The first version should be evaluated as a cashier-assist prototype, not a production system.

## Project Structure

```text
cashiervision/
  app/                 Streamlit demo
  configs/             Class lists and project settings
  data/                Public data notes, raw data, manifests, PLU mapping
  notebooks/           Experiments and EDA
  scripts/             Utility scripts
  src/cashiervision/   Reusable project code
  tests/               Lightweight tests
```

## First Milestones

1. Finalize the MVP produce class list.
2. Download a public baseline dataset into `data/raw/`.
3. Generate train/validation/test manifests.
4. Train a baseline classifier.
5. Add top-k inference and confusion analysis.
6. Connect model outputs to PLU suggestions.
7. Build a Streamlit demo and write a model card.

## Setup

```bash
cd /Users/fawazelahi/Documents/ML/cashiervision
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The current scaffold can be validated without installing ML dependencies:

```bash
python3 scripts/validate_plu_mapping.py
python3 -m unittest discover -s tests
```

## Train the Baseline

Start with a short check using two training batches and two validation batches:

```bash
python3 scripts/train_model.py --smoke-test
```

Then train for five epochs (five passes through the training images):

```bash
python3 scripts/train_model.py --epochs 5
```

The script loads pretrained ResNet18 weights and trains only its new final
classification layer. It audits the manifest first, uses validation top-1 accuracy
to select the best epoch, and leaves the test images for a later evaluation.
Top-1 means the first guess was correct; top-3 means the correct class was among
the first three guesses. Saved accuracies are fractions between 0 and 1.

Each run saves `best_model.pt` (the best validation model), `last_checkpoint.pt`
(the latest completed epoch, optimizer, random states, and best model), and
`metrics.json` (training and validation results per epoch). Defaults are
`models/resnet18_baseline/` for training and `models/resnet18_smoke/` for a smoke
test. Smoke-test results are only a pipeline check, not model performance claims.
Choose a fresh `--output-dir` to repeat a run; existing runs are not overwritten.

To continue the same run for five more epochs:

```bash
python3 scripts/train_model.py --resume models/resnet18_baseline/last_checkpoint.pt --epochs 5
```

Resume restores the model, optimizer, epoch count, shuffle and augmentation random
states, metrics history, and the best result so far. It keeps the saved batch size,
learning rate, and seed. Use the same manifest; `--manifest` can specify its path.
Checkpoints are replaced atomically after each completed epoch. If interrupted
mid-epoch, that incomplete epoch is repeated on resume. A smoke-test checkpoint
continues only as a smoke test. Older `best_model.pt` files lack the optimizer
state needed to resume. Resuming does not download pretrained weights again.

Use `--device cpu` to force CPU execution or `--learning-rate 0.001` to set the
update size. The default device is CUDA, then Apple MPS, then CPU, when available.
For an offline check, combine `--smoke-test --no-pretrained`; this uses random
weights. The fixed seed and saved settings help reproduce runs, but results may
vary across hardware and PyTorch versions.
