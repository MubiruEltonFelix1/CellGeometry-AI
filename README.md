<div align="center">

<br/>

```
  ██████╗███████╗██╗     ██╗      ██████╗ ███████╗ ██████╗
 ██╔════╝██╔════╝██║     ██║     ██╔════╝ ██╔════╝██╔═══██╗
 ██║     █████╗  ██║     ██║     ██║  ███╗█████╗  ██║   ██║
 ██║     ██╔══╝  ██║     ██║     ██║   ██║██╔══╝  ██║   ██║
 ╚██████╗███████╗███████╗███████╗╚██████╔╝███████╗╚██████╔╝
  ╚═════╝╚══════╝╚══════╝╚══════╝ ╚═════╝ ╚══════╝ ╚═════╝
```

### **`CellGeometry-AI`**
*Teaching machines to read the shape of disease*

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![Stars](https://img.shields.io/github/stars/yourusername/CellGeometry-AI?style=flat-square&color=f59e0b)](https://github.com/yourusername/CellGeometry-AI/stargazers)
[![Last Commit](https://img.shields.io/github/last-commit/yourusername/CellGeometry-AI?style=flat-square&color=8b5cf6)](https://github.com/yourusername/CellGeometry-AI/commits)
[![Notebooks](https://img.shields.io/badge/Notebooks-Open%20in%20Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)](https://colab.research.google.com)
[![Status](https://img.shields.io/badge/Status-Active%20Research-ef4444?style=flat-square)]()

<br/>

> *A pathologist can look at a cell under a microscope and — just from its shape — tell you if someone has cancer.*
> *This repo is about teaching a neural network to do the same thing.*

<br/>

</div>

---

## The Problem, In Plain English

A doctor takes a biopsy. A lab tech puts it under a microscope. A pathologist stares at it for 20 minutes and says *"that doesn't look right."*

What they're actually doing — without knowing it — is **running a geometric anomaly detector on biological data**. Healthy cells are round, smooth, uniform. Cancer cells are jagged, enlarged, asymmetric, wrong. The shape tells the story.

This repository is about building the computational version of that skill.

We take raw measurements of cell nuclei — radius, texture, concavity, fractal dimension — and feed them into neural networks that learn to separate signal from noise, malignant from benign, shape from chaos.

**No prior biology required. If you can read a matrix, you can read a cell.**

---

## What Lives Here

```
CellGeometry-AI/
│
├── 01_breast_cancer_mlp/          ← Where it all starts
│   ├── notebook.ipynb             # Full walkthrough: EDA → training → clinical report
│   ├── model.py                   # CancerClassifier: 30→128→64→32→1 architecture
│   ├── pipeline.py                # Z-score normalization + Dataset/DataLoader setup
│   └── report.py                  # Diagnostic output formatter
│
├── 02_nucleus_geometry_explorer/  ← Coming soon
│   └── ...                        # Interactive feature space visualization
│
├── 03_cnn_pathology_slides/       ← Coming soon
│   └── ...                        # PatchCamelyon: from geometry features → raw pixels
│
├── 04_graph_drug_discovery/       ← Planned
│   └── ...                        # GNNs on molecular graphs
│
├── data/                          # Cached datasets (gitignored if large)
├── outputs/                       # Training artifacts, confusion matrices, curves
└── shared/                        # Reusable layers, normalization utils, plot helpers
```

---

## Experiment 01 — Breast Cancer Geometry Classifier

> *"The same way a building inspector can tell a structure is unsound just by looking at stress fractures — a trained network can read nuclear geometry and flag malignancy. Both are just pattern recognition on shapes under pressure."*

**The dataset:** 569 real cell samples from fine-needle aspiration biopsies. Each sample is a `struct` of 30 float-valued measurements: radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, fractal dimension — measured three ways (mean, standard error, worst case) across all nuclei in the sample.

**The task:** Binary classification — `P(malignant | cell_geometry) → [0, 1]`

**What you'll build and understand:**

| Concept | The Code | The Intuition |
|---|---|---|
| Z-score normalization | `(X - μ) / σ` | Putting area (µm²) and smoothness (ratio) on the same ruler |
| PyTorch Dataset contract | `__len__` + `__getitem__` | Your data as a queryable cursor, not a loaded array |
| Feedforward architecture | `30 → 128 → 64 → 32 → 1` | Progressively distilling geometry into a single risk score |
| Binary Cross-Entropy | `-[y·log(ŷ) + (1-y)·log(1-ŷ)]` | Penalizing confident wrong answers exponentially harder |
| Backpropagation | `loss.backward()` | The chain rule, automated, across a computation graph |
| Dropout regularization | `nn.Dropout(0.3)` | Forcing redundancy so no single neuron becomes a critical path |
| Clinical threshold tuning | `P > 0.3` vs `P > 0.7` | Because missing a cancer and causing a false alarm are *different* kinds of wrong |

**Results:**

```
Test Set Performance (114 samples, never seen during training)

              precision    recall    f1-score    support
    Benign       0.97       0.99       0.98          71
 Malignant       0.98       0.95       0.96          43

  accuracy                            0.97         114
```

The recall number for malignant is the one that matters clinically. Every percentage point there is a real patient who doesn't get missed.

---

## The Core Biological Insight

Cancer doesn't just change what a cell *does* — it changes what a cell *looks like*. The geometric signature of malignancy is measurable, reproducible, and learnable.

| Feature | Healthy | Cancerous | Why |
|---|---|---|---|
| **Radius** | Small, uniform | Large, variable | Uncontrolled replication stretches the nucleus |
| **Texture** | Smooth | Grainy | Chromatin reorganizes chaotically |
| **Concavity** | Near zero | High | Nuclear membrane buckles under internal pressure |
| **Symmetry** | High | Low | Asymmetric chromosome segregation breaks the shape |
| **Fractal Dimension** | Low | Higher | Jagged, complex boundary = higher fractal complexity |

This is why geometry *is* information. The pathologist isn't making art — they're reading compressed biology.

---

## Quickstart

```bash
# Clone and set up
git clone https://github.com/yourusername/CellGeometry-AI.git
cd CellGeometry-AI
pip install -r requirements.txt

# Run the first experiment
cd 01_breast_cancer_mlp
jupyter notebook notebook.ipynb
```

Or hit it directly in the cloud:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/CellGeometry-AI/blob/main/01_breast_cancer_mlp/notebook.ipynb)

---

## Planned Experiments

These are next on the roadmap, roughly in order of complexity:

**`02` — Interactive Feature Space Explorer**
3D PCA / UMAP visualization of the 30-dimensional feature space. Watch malignant and benign clusters separate in real time as you add features. The goal: see *why* the network finds it easy or hard to draw a decision boundary.

**`03` — CNN Pathology Slides (PatchCamelyon)**
The geometry features in experiment 01 were extracted *by computer vision* from microscope images. In this experiment, we go one level deeper — skip the feature engineering entirely and learn directly from raw histopathology image patches. Same biological problem, different representational level.

**`04` — Graph Neural Networks for Molecular Property Prediction**
Molecules are graphs. Atoms are nodes. Bonds are edges. GNNs operate on this structure natively — no flattening, no hand-crafted features. We'll predict drug solubility and toxicity directly from molecular graph topology. The "geometry is information" insight, extended from cells to chemistry.

**`05` — Transformer on DNA Sequences**
If a cell is a struct of floats, a gene is a string of characters: `ATCGGCTA...`. Transformers were built for sequences. We'll explore nucleotide-level attention — treating DNA bases like tokens and asking the model what subsequences it finds meaningful.

---

## Who This Is For

- **Software engineers** who are curious about ML but want to see real math, not toy examples
- **Data scientists** who know sklearn but want to understand what's happening at the PyTorch layer
- **Biology-adjacent people** who want to see their domain through a computational lens
- **Anyone** who has stared at a neural network diagram and thought *"but what is it actually doing"*

The notebooks are written to be **read like documentation**, not just run like scripts. Every line has a comment. Every mathematical operation has an analogy. Every architectural decision has a reason.

---

## Key Dependencies

```python
torch>=2.0.0          # The engine: tensors, autograd, nn.Module
numpy>=1.24           # Matrix math at the metal
pandas>=2.0           # Structured data that behaves like a typed table
scikit-learn>=1.3     # Dataset loading and train/test splitting only — no models
matplotlib>=3.7       # Loss curves, distribution plots
seaborn>=0.12         # Correlation heatmaps, statistical visualization
```

Full environment: `pip install -r requirements.txt`

---

## Design Philosophy

Three rules this codebase tries to follow:

**1. No black boxes.**
Every function that does something non-obvious has a comment explaining the *why*, not just the *what*. `optimizer.zero_grad()` has a comment. The `unsqueeze(1)` on the label tensor has a comment. The `+ 1e-8` in the normalization denominator has a comment.

**2. Math you can follow.**
Where there's an equation, it's also in the code. Where there's code, the corresponding equation is in the markdown. The notebook is a translation layer between the two.

**3. Deliberate failure modes.**
Several experiments include intentional breaks — train without normalization, train an overparameterized network, remove dropout — so you can *see* why the standard practices exist. You understand a constraint much better after you've violated it.

---

## Contributing

This is an open research notebook. If you:
- Run an experiment and get different results — open an issue with your setup
- Have a biological dataset that maps to a geometric learning problem — open a PR
- Find a comment that's wrong or an explanation that's confusing — open a PR

The goal is for this to be the kind of repository someone bookmarks when they're trying to actually understand bioinformatics + deep learning from first principles.

---

## References & Further Reading

- **UCI Breast Cancer Wisconsin Dataset** — [Wolberg et al., 1995](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- **Deep Learning** — Goodfellow, Bengio, Courville (the textbook; free online)
- **AlphaFold2** — Jumper et al., 2021 — the logical endpoint of "geometry is information" in structural biology
- **Enformer** — Avsec et al., 2021 — Transformers applied to DNA sequences for gene regulation prediction
- **PatchCamelyon** — Veeling et al., 2018 — the CNN histopathology benchmark used in experiment 03
- **Graph Neural Networks for Drug Discovery** — Gilmer et al., 2017 — the MPNN paper that defines the GNN-for-molecules paradigm

---

<div align="center">

<br/>

*Biology is a 3.8-billion-year-old codebase.*
*We're just trying to read the comments.*

<br/>

[![GitHub](https://img.shields.io/badge/MubiruEltonFelix1-CellGeometry--AI-181717?style=flat-square&logo=github)](https://github.com/MubiruEltonFelix1/CellGeometry-AI)

</div>
