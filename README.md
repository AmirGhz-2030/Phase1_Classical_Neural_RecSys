# 🎬 Phase 1: Fundamentals of Classical & Neural Recommender Systems (Phase 1 RecSys)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![README: English](https://img.shields.io/badge/README-English-brightgreen.svg)](README.md)
[![README: Persian](https://img.shields.io/badge/README-Persian-orange.svg)](README.fa.md)

> 🌐 **Read this in other languages:** **English** | [فارسی (Persian)](README.fa.md)

## 📌 Project Overview
* **Research Topic:** Next-Generation Agentic Recommender Systems
* **Current Phase:** **Phase 1 — Fundamentals & Benchmarks of Classical and Neural Recommender Systems**
* **Developer:** [AmirGhz-2030](https://github.com/AmirGhz-2030)
* **Reference Dataset:** [MovieLens 20M Dataset (GroupLens)](https://grouplens.org/datasets/movielens/20m/) / [Kaggle Source](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset)
* **License:** [MIT License](LICENSE)

---

## 🎯 Phase 1 Objectives
This project is implemented directly in alignment with **Phase 1: Fundamentals of Recommender Systems** and accomplishes the following objectives:
1. End-to-end **Data Preprocessing Pipeline** featuring $20$-Core filtering and **Temporal Leave-One-Last** splitting.
2. Implementation of classical baseline recommender models:
   - **Popularity Baseline:** Global frequency- and popularity-based recommender.
   - **Item-KNN (CF):** Item-based Collaborative Filtering using Cosine Similarity.
   - **Matrix Factorization (SVD):** Matrix factorization in a 64-dimensional latent factor space.
3. Deep learning implementation of **Neural Collaborative Filtering (NCF / NeuMF)** in PyTorch, combining GMF and MLP branches.
4. Academic-standard Top-K ranking evaluation engine with 5 metrics: $HR@K$, $NDCG@K$, $MRR@K$, $Precision@K$, and $Recall@K$.
5. Comparative evaluation under two established scientific protocols: **All-Item Ranking (13,130 items)** and **Sampled-100 (NCF 2017 paper protocol)**.
6. Interactive, production-ready web dashboard built with **Streamlit** and **Plotly** for live demonstrations and real-time testing.

---

## 🏗️ Modular System Architecture

The repository structure is organized around the **Separation of Concerns** principle:

```text
Phase1_Classical_Neural_RecSys/
├── .gitignore                   # Excludes large dataset files and virtual environments
├── LICENSE                      # MIT License
├── README.md                    # Technical and educational documentation (English)
├── README.fa.md                 # Technical and educational documentation (Persian)
├── requirements.txt             # Python dependencies (pandas, torch, streamlit, plotly, ...)
├── app/                         # Web UI module
│   ├── __init__.py
│   └── main.py                  # Interactive Streamlit application with 4 dedicated pages
├── data/
│   ├── raw/                     # Raw dataset files (download and place here)
│   └── processed/               # Preprocessed splits (train.csv, val.csv, test.csv, meta.pkl)
├── src/                         # Core Python source code
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocessor.py      # K-Core filtering and temporal splitting pipeline
│   ├── models/
│   │   ├── __init__.py
│   │   ├── classical.py         # Popularity, Item-KNN, and SVD implementations
│   │   └── ncf.py               # Deep NeuMF implementation in PyTorch
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py           # Mathematical formulas and RankingEvaluator class
│   │   ├── benchmark_classical.py # Classical baselines benchmark script
│   │   ├── benchmark_ncf.py       # NCF benchmark script
│   │   └── benchmark_sampled100.py# Sampled-100 protocol benchmark script
│   └── utils/
└── tests/                       # Data integrity and metric unit tests
    ├── test_data_integrity.py
    └── test_metrics.py
```

---

## 📊 Experimental Results & Benchmark Leaderboard

### 1. Sampled-100 Protocol Evaluation (Following the NCF 2017 Paper)
*(For each test user: 1 target movie + 99 randomly sampled negative movies)*

| Rank | Recommender Model | $HR@5$ | $HR@10$ | $HR@20$ | $NDCG@10$ | $MRR@10$ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Matrix Factorization (SVD)** | **70.56%** | **80.98%** | **88.14%** | **0.5810** | **0.5084** |
| 🥈 | **Item-KNN (CF)** | 62.22% | 77.18% | 89.38% | 0.5109 | 0.4296 |
| 🥉 | **Popularity** | 52.60% | 68.40% | 83.60% | 0.4293 | 0.3506 |

---

### 2. All-Item Ranking Protocol Evaluation (Strict Full Catalog: 13,130 Movies)
*(Recommending from the entire item catalog — ACM KDD reference standard)*

| Rank | Recommender Model | $HR@5$ | $HR@10$ | $HR@20$ | $NDCG@10$ | $MRR@10$ | Training Time |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Matrix Factorization (SVD)** | **5.40%** | **9.06%** | **14.92%** | **0.0464** | **0.0330** | 9.2s |
| 🥈 | **Item-KNN (CF)** | 3.96% | 6.96% | 11.64% | 0.0347 | 0.0243 | 5.9s |
| 🥉 | **Popularity** | 3.00% | 5.12% | 8.38% | 0.0255 | 0.0178 | 4.2s |
| 4 | **NCF (NeuMF)** | 2.74% | 4.92% | 7.98% | 0.0244 | 0.0170 | 548s |

> 💡 **Scientific Note:** Random guessing across a 13,130-movie catalog yields an expected hit rate of **0.076%**. By achieving $HR@10 = 9.06\%$, our implemented SVD model delivers **119x higher accuracy than random guessing**.
>
> 🔍 **Methodology & Computational Analysis Note:**
> * **Structural Difference Between Evaluation Protocols:** Under the All-Item protocol, the system must rank the target item across the entire catalog (13,130 movies), whereas under the Sampled-100 protocol (standard in the NCF 2017 paper), ranking is performed among only 100 candidates (1 positive + 99 negative items). Therefore, the substantial numerical gap reflects the difference in search-space scale, and both protocols are official benchmarks in recommender systems literature.
> * **NCF Performance Analysis in Phase 1 (Rapid Prototyping):** Given the massive scale of MovieLens 20M, the Phase 1 NeuMF model was trained on CPU using a 3% data subsample to validate end-to-end neural pipeline integrity. This initial implementation represents controlled underfitting on CPU and justifies upgrading to GPU-accelerated infrastructure (CUDA) and injecting deep semantic knowledge—fully implemented and finalized in **Phase 2 with an RTX 3070 GPU and a KAR+MoE architecture**.

---

## 🚀 Quick Start Guide

### 1. Create Virtual Environment & Install Dependencies
```bash
# Create and activate venv:
python -m venv .venv

# On Windows:
.\.venv\Scripts\activate

# On Linux / macOS:
# source .venv/bin/activate

# Install packages:
pip install -r requirements.txt
```

### 2. Download & Set Up the Dataset
Due to the large size of the raw files (over 800 MB), dataset files are excluded from the Git repository (`.gitignore`).
* Download the **MovieLens 20M** archive from one of the following sources:
  * **[GroupLens Official Dataset Link](https://grouplens.org/datasets/movielens/20m/)**
  * **[Kaggle MovieLens 20M Dataset](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset)**
* Extract the contents directly into the following directory (the pipeline automatically supports both `rating.csv`/`movie.csv` and `ratings.csv`/`movies.csv` naming conventions):
  ```text
  data/raw/
  ```

### 3. Run the Data Preprocessing Pipeline
```bash
python src/data/preprocessor.py
```

### 4. Run Unit & Verification Tests
```bash
# Data integrity and pipeline tests:
python tests/test_data_integrity.py

# Mathematical verification of ranking evaluation metrics:
python tests/test_metrics.py
```

### 5. Run Benchmarks
```bash
# Classical models benchmark (All-Item Full Space):
python src/evaluation/benchmark_classical.py

# Sampled-100 protocol benchmark (NCF 2017 protocol):
python src/evaluation/benchmark_sampled100.py

# Neural NCF benchmark (optional):
python src/evaluation/benchmark_ncf.py
```

### 6. Launch the Interactive Streamlit Dashboard
```bash
streamlit run app/main.py
```
Then open your browser and navigate to:
👉 **`http://localhost:8501`**

---

## 📄 License
This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 GitHub
* Profile: [@AmirGhz-2030](https://github.com/AmirGhz-2030)
* Repository: `Phase1_Classical_Neural_RecSys`
