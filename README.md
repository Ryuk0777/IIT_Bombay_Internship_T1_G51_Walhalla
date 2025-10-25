# Multimodal EEG + Eye Tracking + IVT Behavioral Response Prediction

This repository contains code and data processing pipelines for predicting **response times** (and related behavioral metrics) using multimodal physiological and eye-tracking data (EEG, Eye, IVT) integrated with psychometrics (PSY).  

---

## 🚀 How to Run

### 1. Create environment & install dependencies

````bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
````
---

## 🧪 Key Design Decisions & Tips

* **Merge using `UnixTime`** (numeric timestamp) rather than human-readable `Timestamp`.
* **Handle sentinel values**: convert `-1` to `NaN`, treat continuous zeros carefully (only drop rows where *all* features are zero).
* **Drop redundant sensor-level timestamp / QuestionKey columns** after merge.
* **Feature selection**: use top-N features from ensemble feature importance, but validate with neural models.
* **Memory management**: downcast data types, drop unused columns, process in chunks if necessary.
* **Cross-subject evaluation**: use GroupKFold with student-based splits to avoid leakage.

---

## 📊 Model Architectures

1. **ANN (feedforward MLP)**
    Input: (num_features,) — no temporal dimension
    Hidden layers → regression output

2. **Temporal CNN**
    Input: (seq_length, num_features) — 1D convolutions across time
    Global pooling + dense head → prediction

3. **BiLSTM / RNN**
    Input: (seq_length, num_features)
    Bidirectional LSTM layers → dense regression output

You can ensemble these or stack them.

---

## 📈 Evaluation Metrics

* **R² (coefficient of determination)**
* **RMSE (root mean squared error)**
* **MAE (mean absolute error)**
* **Per-student cross-validation metrics**
* **Ablation studies / permutation importance** to verify feature contributions

---

## 🧾 Tips & Troubleshooting

* Watch for **memory issues** with full merged data (~10+ million rows). Use downcasting, sparse types, or chunk processing.
* Use **early stopping, dropout, and regularization** to avoid overfitting.
* Always **normalize / scale features** (per student or globally).
* Validate that **QuestionKey filling** doesn’t misassign across trial boundaries.
* Use **model ensembling or stacking** for robustness.

---

## 📂 Requirements

````text
pandas
numpy
scikit-learn
tensorflow
matplotlib
seaborn
````
---
