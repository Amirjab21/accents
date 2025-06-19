# Accent Classification

This repository contains experiments for training accent classifiers using
[Whisper](https://github.com/openai/whisper) and Wav2Vec2 based models.
It also provides utilities for preparing data from the Common Voice corpus and
visualising the resulting embeddings.

## Repository layout

### Core modules

- **`model/`** – Whisper model implementation with additions for accent
  classification.  Key files include `load_model.py` for downloading and loading
  checkpoints and `accent_model.py` which wraps Whisper with a linear classifier.
- **`training/`** – Dataset and training helpers.
  - `accent_dataset.py` defines `ContrastiveDataset` which converts Common Voice
    entries into mel spectrograms and tokenised text.
  - `train_utils.py` provides training loops and evaluation utilities.
- **`finetune-wav2vec/`** – Scripts for fine‑tuning Wav2Vec2 models.

### Training and evaluation scripts

- **`train.py`** – Main training script for the accent classifier.
- **`test.py`** and **`test-53-accents.py`** – Evaluation routines for small and
  large numbers of accent classes.
- **`add_new_accents.py`** / **`add_new_accents2.py`** – Extend an existing model
  to recognise additional accents by updating the classification head.

### Data preparation

- **`extract_commonvoice.py`** – Convert and filter Common Voice audio into a
  Parquet data set.
- **`format_dataset.py`** – Map the diverse accent labels from Common Voice to a
  smaller controlled vocabulary and group related accents by region.
- **`upload.py`** – Helper script to push processed datasets to the Hugging Face
  Hub.

### Visualisation

- **`visualise_accents.py`**, **`visualise_accents_new.py`** and
  **`visualise_accents new.py`** – Generate PCA and t‑SNE plots for the learned
  accent embeddings.

### Other utilities

- **`letter_wav2vec.py`** and **`phoneme.py`** – Experiments with phoneme level
  processing and Wav2Vec2 models.
- **`requirements.txt`** – Python dependencies for the project.

## Running the project

Install the dependencies and ensure a GPU is available for best performance.

```bash
pip install -r requirements.txt
```

Training can then be started with:

```bash
python train.py
```

The scripts in `finetune-wav2vec/` follow a similar pattern for Wav2Vec2 based
experiments.

Processed dataframes can be pushed to the Hugging Face Hub using `upload.py` once
`HF_TOKEN` is configured in your environment.

Visualization scripts read trained checkpoints to produce `tsne` and PCA plots of
accent embeddings.

