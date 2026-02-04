## Viyog

**Viyog** is a lightweight **post-hoc technique** that improves the **reliability of machine learning systems** by enhancing **adversarial robustness**, *without requiring adversarial training*.

### How Viyog Fits In

- **Does not perform OOD or adversarial detection on its own**
- Operates **after** samples have been flagged by a joint detector  
  (e.g., **Mahalanobis-based methods**)
- Helps **separate non–in-distribution (non-ID)** samples  
  such as **OOD** and **adversarial inputs**
- Designed to work seamlessly with **pre-trained models**

Once OOD and adversarial samples are identified by an external detector,  
**Viyog provides an additional signal that helps distinguish and separate them**,  
improving downstream decision-making and overall system robustness.


## Table of contents

- [Key ideas](#key-ideas)
- [Features](#features)
- [Repository structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Context manager usage](#context-manager-usage)
- [Scoring function](#scoring-function)
- [OOD metrics](#ood-metrics)
- [Design notes](#design-notes)
- [Common pitfalls](#common-pitfalls)
- [Testing suggestions](#testing-suggestions)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

---

## Key ideas

- Early convolutional layers encode low-level structure.
- ADV samples often produce low activation magnitudes.
- Measuring and centering per-sample activation norms provides a simple, model-agnostic ADV vs OOD signal.

---

## Features

- Automatically finds and hooks the **first convolutional layer**
  - Prefers `conv1` if present
  - Falls back to first `Conv1d / Conv2d / Conv3d`
- Captures activations via forward hooks (no gradients)
- Computes per-sample **infinity norm** of flattened activations of Training data
- Training-time baseline via `fit()`
- Scores batches or full DataLoaders
- Temperature-scaled, bounded scoring function
- Safe hook cleanup (manual or context manager)
- Standard OOD metrics:
  - AUROC
  - AUPR (IN / OUT)
  - FPR\@95% TPR
  - Detection Error
  - AUTC

---

## Repository structure

```

V2/viyog_repo/
├── README.md
└── src/
└── main.py

```

All implementation lives in `src/main.py`.

---

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- scikit-learn

---

## Installation

Install dependencies:

```bash
pip install torch numpy scikit-learn
```

Make sure the `src/` directory is importable:

```bash
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
```

(Alternatively, install the repo in editable mode if packaging files are added.)

---

## Quick start

### 1. Create the wrapper

```python
from main import Viyog

v = Viyog(model)
```

Optionally specify a device:

```python
v = Viyog(model, device="cuda:0")
```

---

### 2. Fit the training baseline

```python
v.fit(trainloader)
```

This computes and stores the **mean per-sample infinity norm** of the first-layer activations.

> `fit()` **must** be called before `score()`.

---

### 3. Score data

Score a batch:

```python
scores = v.score(batch)
```

Score a full DataLoader:

```python
scores = v.score(testloader)
```

Each call returns a **1D tensor of per-sample scores**.

Higher scores indicate stronger OOD likelihood.

---

### 4. Cleanup

```python
v.close()
```

---

## Context manager usage

The wrapper supports safe automatic cleanup:

```python
from main import Viyog, viyog_metrics

with Viyog(model) as v:
    v.fit(trainloader)

    id_scores = v.score(id_loader)
    ood_scores = v.score(ood_loader)

metrics = viyog_metrics(
    id_scores.cpu().numpy(),
    ood_scores.cpu().numpy()
)

print(metrics)
```

---

## Scoring function

The scoring pipeline is:

1. Capture first-layer activations
2. Flatten per sample
3. Compute infinity norm
4. Center by training mean
5. Apply temperature-scaled nonlinearity

```python
Viyog.Viyog_Score(centered_norms, Temperature=1000.0)
```

Scores are approximately bounded in `(-1, 1)`.

---

## OOD metrics

The helper function:

```python
from main import viyog_metrics
metrics = viyog_metrics(id_scores, ood_scores)
```

Returns:

- `AUROC`
- `AUPR_IN`
- `AUPR_OUT`
- `FPR95`
- `DetectionError`
- `AUTC`
- `AUTC_components`
  - `AUFPR`
  - `AUFNR`

**Labeling convention**

- In-distribution → label `0`
- Out-of-distribution → label `1`
- Higher score → more likely OOD

---

## Design notes

- Hook is attached during `Viyog.__init__`
- Uses `torch.no_grad()` and `model.eval()`
- Device inferred from model parameters unless specified
- Accumulation during `fit()` uses CPU float64 for stability
- Only first element of `(inputs, labels)` batches is used

---

## Common pitfalls

- **Calling **``** before **`` Raises a runtime error.

- **No convolutional layer found** The model must contain `Conv1d/2d/3d` or a `conv1` attribute.

- **Hook not capturing features** Ensure the model’s forward pass actually uses the hooked layer.

- **Device mismatch** Explicitly pass `device="cuda:0"` if needed.

---

## Testing suggestions

Add tests that verify:

- Hook attachment to `conv1`
- `fit()` returns a finite mean
- `score()` returns correct shapes
- `viyog_metrics()` produces numeric outputs

Example (pytest):

```python
def test_viyog_basic():
    import torch
    from main import Viyog
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3, padding=1),
        torch.nn.Flatten(),
        torch.nn.Linear(4 * 32 * 32, 10),
    )
```

---

## Contributing

Contributions are welcome, especially:

- Unit tests and CI
- Example scripts
- Documentation improvements
- Additional scoring variants

Please include a minimal reproducible example when reporting issues.

---

## License

No license is currently specified. Add a `LICENSE` file (e.g. MIT or Apache-2.0) to clarify usage and redistribution.

---

## Citation

If you use this code in academic work, please cite the associated paper or reference this repository in your methods section.

