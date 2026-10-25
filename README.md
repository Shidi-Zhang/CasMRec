# CaSMRec

# Overview

The model combines longitudinal clinical records with molecular and substructure representations:

- **Clinical encoding:** diagnosis and procedure embeddings, causal-effect graph encoding, explicit graph-node alignment, and recurrent visit encoding.
- **History encoding:** previous medications are incorporated into the prediction query when `--history_query` is enabled.
- **Medication prediction:** molecular/substructure scores are added to a direct patient-to-medication prediction branch, followed by causal-effect review.
- **Training:** binary cross entropy and multilabel margin loss, with an adaptive DDI penalty.

The improved implementation also handles weak positive causal effects and empty effect graphs, and combines recurrent states with current visit representations. Baseline models and baseline selection code have been removed.

# Requirements

The following versions were inspected in the local DrugRec environment used for verification. They describe the tested environment, not every compatible combination.

| Package | Tested version |
|---------|----------------|
| Python | 3.8.19 |
| PyTorch | 2.4.1 (CPU environment) |
| NumPy | 1.24.4 |
| SciPy | 1.10.1 |
| scikit-learn | 1.3.2 |
| pandas | 1.5.3 |
| dill | 0.3.9 |
| PyTorch Geometric | 2.6.1 |
| torch-scatter | 2.1.2+pt24cpu |
| OGB | 1.3.6 |
| RDKit | 2024.3.5 |
| NetworkX | 2.8.8 |

If the existing DrugRec environment works, reinstalling dependencies is unnecessary. On this computer, its Python executable is:

```bash
python -m pip install -r requirements-fixed.txt
```

The dependency file does not install PyTorch or torch-scatter, because their binary builds depend on the platform and CUDA version. Cached causal files are supplied, so the runtime does not require cdt, dowhy, or statsmodels.

# Dataset

All eight required files are included under `data/mimic3/`:

| File | Description |
|------|-------------|
| `input/idx2drug.pkl` | Medication codes mapped to molecular SMILES |
| `output/records_final.pkl` | Patient visit sequences and medication targets |
| `output/voc_final.pkl` | Diagnosis, procedure, and medication vocabularies |
| `output/ddi_A_final.pkl` | Drug–drug interaction adjacency matrix |
| `output/ddi_mask_H.pkl` | Medication-to-substructure mapping |
| `graphs/causal_graph.pkl` | Cached diagnosis, procedure, and medication graphs per visit |
| `graphs/Diag_Med_causal_effect.pkl` | Diagnosis-to-medication causal-effect table |
| `graphs/Proc_Med_causal_effect.pkl` | Procedure-to-medication causal-effect table |

The dataset contains 6,350 patients and 15,032 visits. The original patient order is preserved: 4,233 patients for training, 1,059 for validation, and 1,058 for testing. The medication vocabulary contains 131 labels. This package supports the supplied MIMIC-III data.

The entry point uses learned substructure embeddings. Substructure SMILES, raw-data preprocessing scripts, duplicate data files, and the unused causal-design matrix are therefore not included. The runtime reads existing causal caches; it does not regenerate them.

# Documentation

```text
CaSMRec-Improved/
├── README.md
├── requirements-fixed.txt
├── src/
│   ├── main.py                      # Training and test entry point
│   ├── training.py                  # Training, validation, and test sampling
│   ├── util.py                      # Metrics and molecular projection
│   └── modules/
│       ├── HemoMole.py               # Improved recommendation model
│       ├── Causal_inference.py       # Graph encoding and node alignment
│       ├── Causal_discovery.py       # Read-only causal cache loader
│       ├── SetTransformer.py        # Substructure attention
│       └── gnn/
│           ├── __init__.py
│           ├── GNNs.py               # Molecular graph encoders
│           ├── GNNConv.py            # Graph convolution layers
│           └── utils.py              # SMILES-to-graph conversion
├── data/mimic3/                      # Required preprocessed data and caches
├── tests/test_model_checks.py        # Model and graph regression checks
└── runs/                            # Created automatically during training
```

# How to Run CaSMRec-Improved

## Prerequisites

1. Activate an environment containing the dependencies above.
2. Run the following commands from this folder's **project root**.
3. Use a distinct `--model_name` for each experiment. A nonempty output directory is rejected to avoid overwriting results.

## Train the Model

Train the improved model with medication history enabled:

```bash
python src/main.py --history_query --epochs 30 --model_name improved_full
```

CUDA device 0 is used when CUDA is available; otherwise, execution falls back to CPU. To select a device explicitly:

```bash

# GPU 0, if available
python src/main.py --history_query --device 0 --model_name improved_gpu
```

For PowerShell on this computer, the equivalent command using the verified environment is:

```powershell
& 'D:/python/Anaconda3/envs/DrugRec/python.exe' src/main.py --history_query --epochs 30 --model_name improved_full
```

Omitting `--history_query` disables the history contribution to the prediction query, matching the original improved entry point's default. Include the flag in both training and testing to reproduce the history-enabled configuration.

## Default Training Settings

| Setting | Default |
|---------|---------|
| Architecture | Improved CaSMRec only |
| Dataset | MIMIC-III |
| Seed | 1203 |
| Hidden dimension | 64 |
| Optimizer | Adam |
| Learning rate | 0.0005 |
| Dropout | 0.7 |
| Epochs | 30 |
| Molecular graph layers | 4 |
| Substructure representation | Learned embeddings |
| History query | Off unless `--history_query` is supplied |
| Prediction threshold | 0.5 |
| Target DDI | 0.06 |
| DDI coefficient | 2.5 |
| Gradient clipping norm | 5.0 |
| CPU threads | 1 |

When predicted DDI exceeds the target, the training objective mixes the classification loss with the DDI penalty. The classification component is `0.95 × BCE + 0.05 × multilabel margin loss`. The DDI target is a training control, not a guarantee on the final DDI rate. Training's DDI activation calculation uses a fixed 0.5 prediction threshold; `--threshold` controls validation and test binarization.


## Evaluate on the Test Set

Select the model and settings using validation results first, then evaluate the saved checkpoint:

```bash
python src/main.py --Test --history_query --resume_path runs/improved_full/best.model
```

`--resume_path` is required in test mode. Training does not automatically evaluate the test set. The checkpoint contains model weights only: **the saved `config.json` is not automatically restored**. Match the training configuration, especially `--history_query` and `--dim`, and use the chosen prediction threshold. A missing history flag may load successfully while selecting a different prediction branch. The test mode is not a training-resume feature.

Default data and output paths are resolved from the project location. Paths explicitly passed to `--resume_path` are relative to the current working directory unless absolute.

# Outputs and Evaluation

Training creates `runs/MODEL_NAME/`:

| File | Contents |
|------|----------|
| `config.json` | Command-line training configuration |
| `history.pkl` | Per-epoch validation metrics |
| `best.model` | Weights selected by maximum validation Jaccard |
| `Epoch_*.model` | Weights saved after each epoch; epoch indices start at zero |
| `ddi_losses.txt` | Mean DDI penalty and validation DDI per epoch |
| `best.txt` | Running best epoch/Jaccard plus the current epoch's DDI |

The final column of `best.txt` is not necessarily the best checkpoint's DDI. For metrics at the selected epoch, use the same epoch index in every field of `history.pkl`.

Jaccard, PRAUC, precision, recall, and F1 are averaged across visits within each patient and then across patients. PRAUC uses scikit-learn average precision. DDI is the global fraction of recommended medication pairs with a known interaction; average medication count is reported per visit.

Test evaluation runs ten samples with replacement, each sized at 80% of the test patient count, and prints mean and standard deviation. These standard deviations describe test resampling, not variation across independently trained seeds. Validation metrics are not independent test results.

# Tests

```bash
python -B -m unittest discover -s tests -v
python src/main.py --help
```

The four model tests cover supplied DDI matrices, active history/direct-head gradients, checkpoint round trips, graph node ordering, weak positive effects, and empty graphs. The current-target exclusion test holds the synthetic graph fixture fixed; it does not audit real cache generation.
