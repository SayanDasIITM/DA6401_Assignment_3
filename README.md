**Seq2Seq Transliteration Project**

This repository implements a character‐level Seq2Seq transliteration system for English to Bengali, with two variants:

1. **`Translation_attention.py`**: Encoder–Decoder with additive attention, rich metrics, connectivity visualization, and advanced logging.
2. **`Translation_vanilla.py`**: Vanilla Seq2Seq (no attention) with scheduled sampling, data augmentation, and WandB sweep support.

---

## 📋 Table of Contents

* [Project Overview](#project-overview)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Data Preparation](#data-preparation)
* [Scripts Overview](#scripts-overview)

  * [`Translation_attention.py` (Attention)](#translation_attentionpy-attention)
  * [`Translation_vanilla.py` (Vanilla Seq2Seq)](#translation_vanillapy-vanilla-seq2seq)
* [Configuration & Hyperparameters](#configuration--hyperparameters)
* [Logging & Metrics](#logging--metrics)
* [Output & Predictions](#output--predictions)
* [Visualization](#visualization)
* [Contact & Contribution](#contact--contribution)
* [W\&B Report](#wandb-report)

---

## 🔍 Project Overview

This assignment builds a transliteration pipeline on the Dakshina dataset (English→Bengali). The goal is to compare two Seq2Seq variants:

* A full‐featured model with attention, AMP, connectivity analysis, and detailed WandB dashboards.
* A baseline model without attention, equipped with data augmentation and hyperparameter sweep.

Both scripts produce test‐set predictions, log training/validation metrics, and save results for downstream evaluation.

---

## 🛠️ Prerequisites

* Python 3.8+
* PyTorch 1.10+
* wandb (Weights & Biases)
* pandas, numpy, matplotlib, tqdm, editdistance
* Access to Bengali font (for attention heatmaps)

> **Note:** Scripts assume CUDA availability for GPU acceleration. Fallback to CPU is supported.

---

## ⚙️ Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/seq2seq-translit.git
   cd seq2seq-translit
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Download and install a Bengali font (e.g., `kalpurush.ttf`) and update the path in `Translation_attention.py`.

---

## 🗂️ Data Preparation

1. Download the Dakshina transliteration subset for Bengali:

   ```text
   dakshina_dataset_v1.0/bn/lexicons/bn.translit.sampled.{train,dev,test}.tsv
   ```
2. Place the `.tsv` files in a directory and update the `--train_path`, `--dev_path`, and `--test_path` arguments when running the scripts.

---

## 📝 Scripts Overview

### `Translation_attention.py` (Attention)

**Purpose:** Train a bidirectional GRU Seq2Seq with additive attention, mixed precision, connectivity heatmaps, and advanced logging.

**Key Features:**

* **Attention Mechanism**: Computes soft alignment between encoder outputs and decoder state.
* **Mixed Precision**: Uses `torch.cuda.amp` for faster training.
* **Connectivity Viz**: Backprop‐based strongest‐link analysis per output step.
* **Rich Metrics**: Character‐level accuracy, exact‐sequence accuracy, tokens per word, plus edit distance.
* **Visualization**: Attention heatmaps and prediction tables logged to WandB.
* **Test Predictions**: Saved as TSV for GitHub upload.

**Usage:**

```bash
python Translation_attention.py \
  --train_path path/to/bn.translit.sampled.train.tsv \
  --dev_path   path/to/bn.translit.sampled.dev.tsv \
  --test_path  path/to/bn.translit.sampled.test.tsv \
  --project    DA6401_Assignment3 \
  --epochs     10 \
  --batch_size 64 \
  --lr         1e-3 \
  --emb_dim    256 \
  --hid_dim    512 \
  --dropout    0.3
```

### `Translation_vanilla.py` (Vanilla Seq2Seq)

**Purpose:** Train a unidirectional RNN/GRU/LSTM Seq2Seq without attention, with scheduled sampling and W\&B hyperparameter sweep.

**Key Features:**

* **Scheduled Sampling**: Teacher forcing decay to mitigate exposure bias.
* **Data Augmentation**: Random character deletion/insertion on the input side.
* **Hyperparameter Sweep**: Configurable in‐code sweep for emb\_dim, hid\_dim, dropout, etc.
* **Metrics**: Character‐level and word‐level accuracy.
* **Logging**: Enhanced W\&B table of predictions per epoch.
* **Test Predictions**: Saved as TSV in `predictions_vanilla/`.

**Usage:**

```bash
python Translation_vanilla.py \
  --train_path      path/to/bn.translit.sampled.train.tsv \
  --dev_path        path/to/bn.translit.sampled.dev.tsv \
  --test_path       path/to/bn.translit.sampled.test.tsv \
  --project         DA6401_Assignment3 \
  --predictions_dir predictions_vanilla
```

To launch a sweep (with default settings in the script):

```bash
python Translation_vanilla.py  # triggers `run_sweep()` if called as main
```

---

## ⚙️ Configuration & Hyperparameters

Both scripts expose command‐line arguments for key settings:

| Argument       | Default                    | Description               |
| -------------- | -------------------------- | ------------------------- |
| `--epochs`     | `10` (attention)           |                           |
|                | `<sweep>` (vanilla)        | Number of training epochs |
| `--batch_size` | `64`                       | Batch size                |
| `--lr`         | `1e-3`                     | Learning rate             |
| `--emb_dim`    | `256`                      | Embedding dimension       |
| `--hid_dim`    | `512` (attention), `384`\* | Hidden dimension          |
| `--dropout`    | `0.3`                      | Dropout probability       |
| `--project`    | `DA6401_Assignment3`       | WandB project name        |

> \*Vanilla script sets default sweep values in code.

---

## 📊 Logging & Metrics

* **WandB Dashboard**: Tracks loss, char‐acc, seq‐acc/word‐acc, and custom tables.
* **Tables**: Prediction tables with actual vs. predicted transliteration.
* **Connectivity**: WandB Table of strongest‐link connectivity for attention model.

---

## 📂 Output & Predictions

* **Attention Model**: `predictions_attention/test_preds.tsv`
* **Vanilla Model**: `predictions_vanilla/test_predictions_<run_name>.tsv`

Each TSV contains columns:

```
English_input    Actual_Bengali    Predicted_Bengali
```

---

## 📈 Visualization

* **Attention Heatmaps**: Logged as images showing the alignment between source and target characters.
* **Connectivity Table**: Strongest influence per decoder step back to encoder tokens.

---

## ✍️ Contact & Contribution

Contributions and issues are welcome! Please open a GitHub Issue or Pull Request.

---

## 📎 W\&B Report

*Link to the full WandB report and dashboards:*

\[Insert W\&B Report Link Here]

---

*Happy Transliteration!*
