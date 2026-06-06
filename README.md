# 🧫 CellGeometry-AI
### *A geometric deep learning pipeline for cellular morphology classification*

---

> A low-level, mathematically explicit neural pipeline for analyzing cellular nuclei geometry and classifying malignancy risk using structured biomedical features.

---

## ⚙️ Tech Stack

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![MIT License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

---

## 🧠 System Philosophy

CellGeometry-AI is designed around a single principle:

> **Expose every transformation from raw biological geometry to final prediction.**

No hidden pipelines. No black-box preprocessing. Every tensor movement is explicit, traceable, and mathematically grounded.

---

## 🧬 Problem Domain: Cellular Morphology

Under microscopic imaging, nuclei geometry encodes diagnostic signals:

- **Benign structures** → smooth, symmetric, low-variance geometry  
- **Malignant structures** → irregular boundaries, high concavity, spatial instability  

We model these differences using structured feature vectors derived from:

- Radius mean & standard error  
- Perimeter scaling  
- Area distortion ratios  
- Concavity & symmetry measures  

---


---

## 🔬 Mathematical Formulation

### 1. Affine Transformation

Each layer computes:

\[
Y = XW^T + b
\]

Where:
- \(X\) = input feature tensor  
- \(W\) = learnable weight matrix  
- \(b\) = bias vector  

---

### 2. Non-Linearity Injection

We apply ReLU activation to enforce sparsity:

\[
f(x) = \max(0, x)
\]

This introduces geometric segmentation in latent feature space.

---

### 3. Output Probability Mapping

Final layer uses sigmoid activation:

\[
\sigma(x) = \frac{1}{1 + e^{-x}}
\]

Interpreted as:

> Probability of malignancy ∈ [0, 1]

---

## 📊 Tensor Flow Snapshot

```
Input Tensor Shape        : [B, 30]
Hidden Layer 1            : [B, 16] → ReLU + Dropout
Hidden Layer 2            : [B, 8]  → ReLU
Output Layer              : [B, 1]  → Sigmoid Probability
```

---

## 🧱 Project Structure

```
cellgeometry-ai/
│
├── data/
│   └── sample_biopsy.csv
│
├── notebook/
│   └── cellular_diagnostic_starter.ipynb
│
├── src/
│   ├── model.py        # Neural network architecture (PyTorch)
│   ├── pipeline.py     # Feature scaling + preprocessing
│   ├── dataset.py      # Tensor dataset builder
│   └── predict.py      # CLI inference engine
│
├── README.md
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/cellgeometry-ai.git
cd cellgeometry-ai
```

### 2. Create Environment
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run Inference
```bash
python src/predict.py
```

---

## 🧾 Output Example

```
==================================================
 CELLULAR MORPHOLOGY DIAGNOSTIC REPORT
==================================================

Patient ID | Malignancy Risk | Classification
-----------|----------------|-----------------
#01        | 94.32%         | Malignant
#02        | 1.14%          | Benign
#03        | 89.65%         | Malignant
#04        | 0.04%          | Benign

==================================================
```

---

## 🧪 Engineering Stress Tests

### 1. Batch Scaling Stability
Increase batch size from `32 → 64` and observe:
- Shape invariance across all linear layers
- Stable gradient propagation under higher throughput

---

### 2. Normalization Ablation
Remove Z-score normalization:

**Expected behavior:**
- Gradient instability
- Slower convergence
- Feature scale dominance effects

---

### 3. Overparameterization Test
Increase hidden size to `1024` and disable dropout:

**Expected behavior:**
- Near-zero training loss
- Severe generalization degradation

---

## 🧠 Key Design Insights

- Explicit tensor tracking replaces “hidden preprocessing”
- Architecture prioritizes interpretability over abstraction
- Every transformation is reproducible from raw NumPy operations
- Designed for educational clarity + research prototyping

---

## 📌 Future Extensions

- Convolutional morphology encoder (cell image input)
- Attention-based diagnostic weighting
- Multi-class tumor grading system
- ONNX export for clinical deployment simulation

---

## 📄 License

MIT License — free to use, modify, and extend.

---
```

