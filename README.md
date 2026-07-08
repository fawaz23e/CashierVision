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

