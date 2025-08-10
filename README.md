# 📌 Multi-task model with attribute-specific heads for person re-identification

[![DOI](https://img.shields.io/badge/DOI-10.1007%2Fs10044--025--01421--0-blue)](https://doi.org/10.1007/s10044-025-01421-0)
[![Paper](https://img.shields.io/badge/View%20Paper-Springer-green)](https://link.springer.com/article/10.1007/s10044-025-01421-0)
[![Stars](https://img.shields.io/github/stars/TripleTheGreatDali/ReIDMTMASH?style=social)](https://github.com/TripleTheGreatDali/ReIDMTMASH/stargazers)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📰 Publication
**Published**: *Pattern Analysis and Applications*, Volume 28, Article 38 — 31 January 2025  
**DOI**: [10.1007/s10044-025-01421-0](https://doi.org/10.1007/s10044-025-01421-0)

---

## 📝 Overview
This **multi-task learning model** performs **simultaneous Person Re-Identification (ReID)** and **Pedestrian Attribute Recognition**.  
It enhances **digital surveillance**, **smart city systems**, and **security analytics** by combining robust identity classification with fine-grained attribute predictions, including:

- Gender
- Age group
- Clothing type & color
- Accessories (hat, backpack, bag, etc.)

By **sharing features across tasks** while using **attribute-specific heads**, ReIDMTMASH achieves **state-of-the-art** performance in real-world scenarios.

---

## 💡 Why It Matters
Most ReID systems focus only on identity matching, ignoring semantic attributes that provide **critical contextual cues**.  
ReIDMTMASH addresses this by:
- Combining **ReID** with **detailed attribute recognition**
- Improving **accuracy** and **reliability** in challenging environments
- Offering interdisciplinary value across **AI, computer vision, Digital surveillance, and public safety**

---

## 🚀 Key Features
- **Shared Backbone** — Choose **ResNet50** or **EfficientNet** for flexible performance/efficiency trade-offs.
- **Generalized Mean (GeM) Pooling** — Learns to prioritize salient visual cues adaptively.
- **Attribute-Specific Heads**:
  - **Binary Heads** — Gender, hat, backpack, etc.
  - **Multi-Class Heads** — Age group, clothing category, etc.
  - **Color Heads** — Detect upper/lower clothing colors.
- **Balanced Multi-Task Optimization** — Dynamic loss weighting prevents one task from dominating.
- **Validated on Benchmark Datasets** — **Market1501** & **DukeMTMC-reID**.

---

## 📊 Results at a Glance

| Dataset / Backbone        | ReID Acc | Overall Acc | Precision | Recall | F1-Score |
|---------------------------|----------|-------------|-----------|--------|----------|
| DukeMTMC-reID (ResNet50)  | **99.57%** | 98.03%    | 93.80%    | 94.75% | 94.21%   |
| DukeMTMC-reID (EffNet)    | 94.59%     | 96.90%     | 92.09%    | 92.97% | 92.43%   |
| Market1501 (ResNet50)     | **94.59%** | **100%**  | **100%**  | **100%** | **100%** |
| Market1501 (EffNet)       | 88.79%   | 99.99%     | 99.98%    | 99.98% | 99.98%   |

> **Finding:** GeM pooling **outperforms** both max- and average-pooling in all tested configurations.

---

## 🛠 Installation

### Prerequisites
- Python **3.8+**
- PyTorch **1.9+**
- NVIDIA GPU (e.g., RTX 3060)+ recommended
- Dependencies: `PyTorch`, `torchvision`, `numpy`, `scipy`, `scikit-learn`

## Citation

If this work has been helpful, please cite:

```bibtex
@article{ahmed2025multi,
  title={Multi-task model with attribute-specific heads for person re-identification},
  author={Ahmed, Md Foysal and Oyshee, Adiba An Nur},
  journal={Pattern Analysis and Applications},
  volume={28},
  number={1},
  pages={38},
  year={2025},
  publisher={Springer},
  doi={10.1007/s10044-025-01421-0}
}
