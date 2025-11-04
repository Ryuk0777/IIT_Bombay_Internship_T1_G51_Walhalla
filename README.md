# Multimodal Cognitive Response Time Prediction

**Predicting students’ response times using EEG, eye-tracking, motion, and psychometric data.**

---

## 📖 Table of Contents

1. [Project Overview](#-project-overview)
2. [Dataset Description](#-dataset-description)
3. [Motivation](#-motivation)
4. [Methodology](#-methodology)
5. [Data Processing & Feature Engineering](#-data-processing--feature-engineering)
6. [Model Development](#-model-development)
7. [Results & Performance](#-results--performance)
8. [Project Pipeline Diagram](#-project-pipeline-ascii-diagram)
9. [Hardware & Training Environment](#️-hardware--training-environment)
10. [Repository Structure](#-repository-structure)
11. [How to Run Locally](#-how-to-run-locally)
12. [Challenges & Learnings](#-challenges--learnings)
13. [Future Work](#-future-work)
14. [ References](#-references)
15. [License](#-license)

---

## 🧩 Project Overview

This project focuses on building an artificial intelligence system to **predict student response times** during cognitive tasks using **multimodal sensor data** — including EEG (brainwave), eye-tracking, and psychometric logs.

The goal is to model the relationship between neural activity, behavioral signals, and task difficulty to understand cognitive effort and reaction patterns.


---

## 🧠 Dataset Description

**Dataset Title:** *A Multisensor Dataset of South Asian Post-Graduate Students Working on Mental Rotation Tasks*
**Authors:** Ashwin T. S., Suraj Ranganath, Kabyashree Khanikar, Karishma Khan, Ramkumar Rajendran, Ritayan Mitra
**DOI:** [https://doi.org/10.6084/m9.figshare.28120670.v1](https://doi.org/10.6084/m9.figshare.28120670.v1)

### Summary:

* **Subjects:** 38 post-graduate students
* **Sensors Used:** EEG, Eye-Tracking, IVT, Psychometric, Accelerometer, Gyroscope
* **Task:** Mental rotation cognitive activity

Each trial corresponds to a unique `QuestionKey` with features describing neural activity, gaze behavior, motion patterns, and subjective difficulty.

---

## 🎯 Motivation

Traditional learning analytics systems focus on grades, scores, or completion time — which do not reflect real cognitive effort or mental workload.

This project explores **how physiological and behavioral signals can predict response times**, offering insights for adaptive learning systems that react to students’ cognitive states.

---

## 🧮 Methodology

### Type-1: Question-level Aggregation

Data aggregated by `QuestionKey` (one record per question).
Baseline models (Random Forest, XGBoost) used to estimate response time from averaged features.

* **Mean R²:** 0.51 – 0.65

---

### Type-2: Second-level Temporal Features

Data aggregated by `Second`.
Created time-based and temporal summary features such as:

* Per-second gaze averages
* Fixation/saccade ratios
* EEG band averages (Δ, θ, α, β, γ)
* Per-question spectral mean features

Saved the new engineered dataset as *Type-2 features*, later used for deep learning.

* **Mean R² (XGBoost GPU):** 0.78

---

### Type-3: Deep Temporal Modeling

Used sequence modeling to capture second-wise temporal dependencies.

* Input shape: `(samples, sequence_length, features)`
* Sequence length (`seq_len`): 10–30 seconds
* Architectures: **Temporal CNN** and **BiLSTM**
* Validation: **Group K-Fold** (5 splits to prevent cross-student leakage)

**Best Model:** BiLSTM

* **Mean R²:** 0.98
* **MAE:** ~1.6 seconds

---

## 🧰 Data Processing & Feature Engineering

### Preprocessing Steps:

* Removed trials with missing or invalid `ResponseTime`
* Filtered out idle seconds and missing `QuestionKey`
* Aligned EEG, EYE, IVT, and PSY using `UnixTime`
* Handled NaN / -1 sentinel values
* Scaled features using `StandardScaler`

### Engineered Features:

| Category            | Features                                               |
| ------------------- | ------------------------------------------------------ |
| EEG Bands           | Delta, Theta, Alpha, Beta, Gamma (raw + averaged)      |
| Eye                 | Gaze X/Y variance, fixation/saccade ratios, dispersion |
| Motion              | Accelerometer X/Y/Z, Gyroscope X/Y/Z                   |
| Psychometric        | Difficulty, QuestionKey, ResponseTime                  |
| Temporal Aggregates | Per-second and per-question means                      |

---

## 🧠 Model Development

| Model         | Type             | Key Features            | Mean R²   | Notes                                    |
| ------------- | ---------------- | ----------------------- | --------- | ---------------------------------------- |
| Random Forest | Tree-based       | QuestionKey Aggregation | 0.51      | Baseline                                 |
| XGBoost       | Tree-based (GPU) | Temporal + Difficulty   | 0.65–0.78 | Tuned via RandomizedSearchCV             |
| Temporal CNN  | Deep Learning    | Sequential Second Data  | 0.94      | Captures local time patterns             |
| BiLSTM        | Deep Learning    | Sequential Second Data  | **0.98**  | Captures long-term temporal dependencies |

---

## 📊 Results & Performance

### Overall Results:

| Model         | Mean R²    | MAE    | Validation Type |
| ------------- | ---------- | ------ | --------------- |
| BiLSTM        | **0.9829** | ~1.6 s | GroupKFold (5)  |
| Temporal CNN  | 0.945      | ~1.8 s | GroupKFold (5)  |
| XGBoost (GPU) | 0.656      | ~2.9 s | GroupKFold (5)  |

---

## 📊 Project Pipeline (ASCII Diagram)

```
                                                ┌─────────────────────────┐
                                                │        Raw Sensor Data  │
                                                │  EEG | EYE | IVT | PSY  │
                                                └─────────────────────────┘
                                                              │
                                                              ▼
                                             ┌──────────────────────────────┐
                                             │  Data Cleaning & Alignment   │
                                             │ (UnixTime Sync, Idle Removal)│
                                             └──────────────────────────────┘
                                                              │
                                                              ▼
                              ┌──────────────────────────────────────────────────────────────┐
                              │                   FEATURE AGGREGATION                        │
                              ├──────────────────────────────────────────────────────────────┤
                              │  TYPE-1: Question-Level Aggregation                          │
                              │   - Aggregate per QuestionKey                                │
                              │   - Features: EEG bands, gaze stats, difficulty              │
                              │   - Models: Linear, RF, XGBoost                              │
                              │   - R² ≈ 0.51-0.65                                           │
                              ├──────────────────────────────────────────────────────────────┤
                              │  TYPE-2: Second-Level Temporal Features                      │
                              │   - Aggregate per Second                                     │
                              │   - Add temporal features: gaze averages, band means         │
                              │   - Models: RF, XGBoost (GPU)                                │
                              │   - R² ≈ 0.65-0.78                                           │
                              ├──────────────────────────────────────────────────────────────┤
                              │  TYPE-3: Deep Temporal Modeling                              │
                              │   - Sequence creation (seq_len = 10-30 s)                    │
                              │   - Models: Temporal CNN, BiLSTM                             │
                              │   - R² ≈ 0.94-0.98                                           │
                              └──────────────────────────────────────────────────────────────┘
                                                            │
                                                            ▼
                                             ┌──────────────────────────────┐
                                             │   Response Time Prediction   │
                                             │  (Regression Output, MAE≈1.6)│
                                             └──────────────────────────────┘
```

---

## ⚙️ Hardware & Training Environment

| Component                     | Specification                                            |
| ----------------------------- | -------------------------------------------------------- |
| **CPU**                       | Intel Core i5-14600K (14 cores, 20 threads)              |
| **GPU**                       | NVIDIA RTX 5070 (12 GB  GDDR7 VRAM)                            |
| **RAM**                       | 32 GB DDR5 (6000 MHz)                                    |
| **OS**                        | Ubuntu 22.04 LTS                                         |
| **Frameworks**                | Python 3.12, TensorFlow 2.x, Scikit-learn, XGBoost (GPU) |
| **RAM Usage During Training** | ~22 GB                                                   |

**GPU Acceleration:**
Used for both **XGBoost (tree_method = hist)** and **deep learning** training to reduce training time and handle large-scale feature tensors efficiently.

---

## 🧱 Repository Structure

```
├── data/
│   ├── raw/                        # Raw sensor data (EEG, EYE, IVT, PSY)
│   ├── processed/                  # Cleaned and merged data
│   └── feature_engineered/         # Type-1 and Type-2 feature data
│
├── notebooks/
│   ├── 01_type1_baseline.ipynb
│   ├── 02_type2_feature_engineering.ipynb
│   └── 03_type3_deep_models.ipynb
│
├── models/
│   ├── xgb_model_tuned.joblib
│   ├── temporal_cnn_model.h5
│   └── bilstm_model.h5
│
├── requirements.txt
└── README.md
```

---

## 🧪 How to Run Locally

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Ryuk0777/IIT_Bombay_Internship_T1_G51_Walhalla.git
cd IIT_Bombay_Internship_T1_G51_Walhalla
```

### 2️⃣ Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

open notebooks to explore step-by-step:

* Type-1 Baseline (Classical ML)
* Type-2 Feature Engineering
* Type-3 Deep Models

---

## 🚧 Challenges & Learnings

### Challenges:

* High memory consumption during aggregation (~22 GB RAM)
* Long CPU-based training times
* Synchronizing multimodal signals with missing timestamps
* Avoiding feature leakage across students

### Learnings:

* Second-level aggregation revealed strong temporal patterns
* GPU acceleration (RTX 5070) drastically reduced model training time
* Proper grouping by students (GroupKFold) is essential for realistic evaluation

---

## 🚀 Future Work

* Real-time inference pipeline for live EEG and eye-tracking data
* Deploy as an interactive FastAPI service
* Model optimization using ONNX Runtime and TensorRT
* Cognitive analytics dashboard for educators

---


## 📘 References

1. T. S. Ashwin et al., *A Multisensor Dataset of South Asian Post-Graduate Students Working on Mental Rotation Tasks*, Scientific Data (2025). [DOI:10.1038/s41597-025-04865-5](https://doi.org/10.1038/s41597-025-04865-5)
2. Dataset DOI: [https://doi.org/10.6084/m9.figshare.28120670.v1](https://doi.org/10.6084/m9.figshare.28120670.v1)


---

## 📜 License

Released under the **MIT License**.
You are free to use, modify, and distribute with attribution.









