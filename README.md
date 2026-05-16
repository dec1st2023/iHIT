# iHIT / FloPNet Reproduction

This repository provides a script-oriented reproduction workflow for an
iHIT / FloPNet-style vestibular function experiment. It is written for users
who want to reproduce preprocessing, training, and test-set evaluation from a
single root entry point.

```powershell
python main.py
```

`main.py` loads the root `config.yaml`, adds `src/` to `sys.path`, and calls
`workflow.run(config)`. The project does not require `python -m ihit`,
`pip install -e .`, or an `ihit` package namespace.

---

## Repository Layout

```text
repo-root/
+- main.py                  # Root script entry point
+- config.yaml              # Default experiment configuration
+- requirements.txt         # Runtime dependencies
+- RAFT/                    # Vendored RAFT source and default checkpoint
+- data/                    # Local raw/generated data
+- checkpoint/              # Optional checkpoints for evaluate mode
+- output/                  # Per-run outputs: output/case_YYYYMMDD_HHMMSS
+- src/
   +- config.py             # YAML loading and path resolution
   +- workflow.py           # train / evaluate / all orchestration
   +- data_setup.py         # Data readiness checks and rebuild dispatch
   +- dataset/              # Splits, datasets, rebuild helpers
   +- flow/                 # RAFT optical-flow integration
   +- preprocess/           # ROI, landmarks, sequence helpers
   +- models/               # Stage1 R3D and Stage2 LSTM models
   +- training/             # Training loops, checkpoints, curves
   +- evaluation/           # Test-set metrics and confusion matrices
```

Internal imports are direct from `src/` modules:

```python
from dataset.splits import load_split
from training.runners import train
from evaluation.runners import evaluate
```

---

## Installation

Create and activate a Python environment, then install dependencies:

```powershell
conda create -n ihit python=3.10
conda activate ihit
pip install -r requirements.txt
```

If your CUDA or PyTorch setup requires a specific build, install the matching
PyTorch package first, then install the remaining requirements.

---

## Required Assets

The workflow uses these paths when the corresponding stage is selected:

```text
data/raw/                         # Raw sample frame folders
data/label/                       # train / val / test split arrays and labels
data/flow/                        # RAFT optical-flow features
data/position/                    # leye / reye / nose displacement sequences
RAFT/model/raft-things.pth        # Default RAFT checkpoint
checkpoint/stage1.pth             # Stage1 checkpoint for evaluate mode
checkpoint/stage2.pth             # Stage2 checkpoint for evaluate mode
```

If required data assets under `data/raw`, `data/label`, `data/flow`, or
`data/position` are missing, the workflow attempts a full dataset rebuild. If a
non-generatable dependency is missing, such as a required Python package or an
evaluation checkpoint, the workflow fails with a clear error.

---

## Configuration

The default configuration file is the root `config.yaml`.

```yaml
mode: all
stage: full
train_stage: full
output_dir: output
seed: 42
```

### `mode`

| Value | Behavior |
| --- | --- |
| `train` | Prepare training data, then train only. |
| `evaluate` | Prepare evaluation data, then evaluate configured checkpoints on the test set. |
| `all` | Prepare data, train, then evaluate the newly trained checkpoints on the test set. |

### `stage`

| Value | Scope |
| --- | --- |
| `stage1` | SCC / direction classification only. |
| `stage2` | Polarity / positive-negative classification only. |
| `full` | Stage1 plus Stage2. |

### `train_stage`

`train_stage` is used only when `mode` is `train` or `all`.

```yaml
train_stage: full   # stage1 | stage2 | full
```

If `train_stage: stage2` is selected and no usable Stage1 checkpoint exists,
the workflow trains Stage1 first because Stage2 evaluation needs
Stage1-predicted SCC masks.

### `training.use_best_val_checkpoint`

```yaml
training:
  use_best_val_checkpoint: true
```

| Value | Checkpoint behavior |
| --- | --- |
| `true` | Save the epoch with the best validation accuracy and use it for test-set evaluation. |
| `false` | Save the final-epoch checkpoint. |

In `mode: all`, evaluation uses the checkpoints produced in the current run.
The configured `checkpoints.stage1` and `checkpoints.stage2` paths are used only
by `mode: evaluate`.

---

## Model Workflow

### Stage1: SCC Direction Classification

Stage1 consumes left-eye and right-eye RAFT flow clips and predicts one of six
SCC directions:

```text
LA / LL / LP / RA / RL / RP
```

Training logs include both training and validation metrics:

```text
stage1 epoch 12/80 train_loss=... train_acc=... val_loss=... val_acc=...
```

### Stage2: Polarity Classification

Stage2 is saved as one checkpoint, `stage2.pth`, and the model contains three
independent LSTM branches:

```text
Stage2LSTM
+- left_lstm   + left_fc
+- right_lstm  + right_fc
+- double_lstm + double_fc
```

Training behavior:

- VOR data trains the `double_lstm` branch.
- LVL data trains the `left_lstm` and `right_lstm` branches.
- Stage2 training uses true sample labels (`y_dir`) to select the branch.

Evaluation behavior:

- Stage1 first predicts the SCC direction mask.
- Stage2 receives the Stage1-predicted direction mask plus displacement features.
- Metrics and confusion matrices are computed on the `test` split.

---

## Outputs

Every run writes to a timestamped case directory:

```text
output/case_YYYYMMDD_HHMMSS/
```

Typical contents:

```text
checkpoints/
+- stage1.pth
+- stage2.pth

metrics/
+- stage1_history.json
+- stage2_vor_history.json
+- stage2_lvl_history.json
+- full_metrics.json

visuals/
+- stage1_training_curve.png
+- stage2_vor_training_curve.png
+- stage2_lvl_training_curve.png
+- stage1_conf_mat.png
+- stage2_vor_conf_mat.png
+- stage2_lvl_conf_mat.png
+- stage2_conf_mat.png
+- evaluation_metrics.png
```

Visualization files:

- `*_training_curve.png`: training accuracy and validation accuracy curves.
- `stage1_conf_mat.png`: Stage1 six-class confusion matrix on the test set.
- `stage2_vor_conf_mat.png`: VOR positive/negative confusion matrix on the test set.
- `stage2_lvl_conf_mat.png`: LVL positive/negative confusion matrix on the test set.
- `stage2_conf_mat.png`: paper-style 12-class Stage2 confusion matrix.
- `evaluation_metrics.png`: test-set accuracy, precision, recall, and F1 summary.

All evaluation confusion matrices are generated from the test set.

---

## Data Rebuild Rules

At startup the workflow checks:

```text
data/raw
data/label
data/flow
data/position
```

If required files are missing, it rebuilds the active dataset:

1. Stage raw sample folders.
2. Generate train / validation / test splits.
3. Detect eye and nose landmarks.
4. Generate displacement sequences.
5. Run RAFT optical flow.
6. Write the active dataset under `data/`.

Fixed reproduction constants:

```text
window size: 20
flow frames: 15
image size: 224x224
RAFT backend: local RAFT
```

---

## RAFT Third-Party Notice

This repository vendors the RAFT source used for optical-flow extraction. RAFT
is developed by Zachary Teed and Jia Deng and is available at
[princeton-vl/RAFT](https://github.com/princeton-vl/RAFT). The upstream project
is distributed under the BSD 3-Clause license and contains the source code for
the paper *RAFT: Recurrent All-Pairs Field Transforms for Optical Flow*.

The local `RAFT/LICENSE` is BSD 3-Clause. That license permits redistribution
of the source code when the copyright notice, license conditions, and
disclaimer are retained, and when the original authors or contributors are not
used to imply endorsement without permission.

Tracked in Git:

```text
RAFT/LICENSE
RAFT/core/
RAFT/model/raft-things.pth
```

Other RAFT checkpoint binaries remain ignored by default:

```text
RAFT/model/*.pth
RAFT/*.pth
```

---

## Common Commands

### Train and Evaluate on the Test Set

```yaml
mode: all
stage: full
train_stage: full
```

```powershell
python main.py
```

### Train Only

```yaml
mode: train
train_stage: stage1   # stage1 | stage2 | full
```

```powershell
python main.py
```

### Evaluate Existing Checkpoints

```yaml
mode: evaluate
stage: full
checkpoints:
  stage1: checkpoint\stage1.pth
  stage2: checkpoint\stage2.pth
```

```powershell
python main.py
```

---

## Development Checks

```powershell
python -m unittest discover -s tests -v
python -m compileall src tests main.py
python -m unittest tests.test_models -v
```

---

## Summary

Edit `config.yaml`, run `python main.py`, and inspect
`output/case_YYYYMMDD_HHMMSS`. Checkpoints, metric JSON files, training curves,
and test-set confusion matrices are grouped inside that case directory.
