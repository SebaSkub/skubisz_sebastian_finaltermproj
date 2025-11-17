# skubisz_sebastian_finaltermproj
CS634 Final Project

# Water Quality Classification - Final Project  
**Author:** Sebastian Skubisz  

---

## 1) Folder Layout
```
water_quality_classification/
│
├─ datasets/                              # input CSV files
│   ├─ waterQuality1.csv
│
├─ outputs/                               # algorithm results
│   ├─ KNN/
│   ├─ RandomForest/
│   ├─ LSTM/
│
├─ water_quality_classification.py         # main script
├─ README.md                              # this file
└─ requirements.txt                       # dependencies
```

### Why this layout?  
All datasets are stored under `datasets/`. The main script processes the dataset and generates model evaluation results inside the `outputs/` folder. This layout ensures everything stays organized.

---

## 2) Dataset (Input Format)
The dataset `waterQuality1.csv` includes data for water quality classification. The structure is as follows:

| Column         | Type  | Description |
|----------------|-------|-------------|
| aluminium      | float | Amount of aluminium in water |
| ammonia        | float | Ammonia concentration |
| arsenic        | float | Arsenic concentration |
| ...            | ...   | ... |
| is_safe        | int   | Target variable (1 for safe, 0 for unsafe) |

---

## 3) Environment Setup
Use **Python 3.9–3.12**

### 🪟 Windows PowerShell
```bash
python -m venv .venv
.venv\Scriptsctivate
pip install --upgrade pip
pip install -r requirements.txt
```

### 🐧 macOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### requirements.txt
```
pandas
seaborn
matplotlib
scikit-learn
keras
imblearn
```

---

## 4) How to Run (CLI)

Run the main script interactively from the project root:
```bash
python water_quality_classification.py
```

The script will:

- Load the dataset
- Preprocess the data (handle missing values, normalize features)
- Train three models: **KNN**, **Random Forest**, and **LSTM**.
- Evaluate models using confusion matrix, AUC, and other metrics
- Plot and save ROC curves and other evaluation results

### Example Output
```
===== KNN =====
AUC: 0.88

===== Random Forest =====
AUC: 0.92

===== LSTM =====
AUC: 0.94
```

---

## 5) Example Output

The following evaluation results are stored in the `outputs/` folder:

```
outputs/
├─ KNN/
│  ├─ confusion_matrix.png
│  └─ roc_curve.png
├─ RandomForest/
│  ├─ confusion_matrix.png
│  └─ roc_curve.png
├─ LSTM/
│  ├─ confusion_matrix.png
│  └─ roc_curve.png
```

---

## 6) Defensive Validation
- Detects missing or invalid datasets
- Re-prompts if support/confidence ≤ 0  
- Handles multiple CSV structures and separators gracefully

---

## 7) Key Concepts (Summary)
- **Binary Classification**: Classifying water as "safe" or "unsafe" based on various water quality parameters.
- **Evaluation Metrics**: Confusion matrix, accuracy, precision, recall, F1 score, and AUC.

| Algorithm     | Description | Strengths | Weaknesses |
|---------------|-------------|-----------|------------|
| **KNN**       | Instance-based classifier | Simple, interpretable | Computationally expensive for large datasets |
| **RandomForest** | Ensemble model with decision trees | Robust to overfitting, handles imbalance well | Less interpretable |
| **LSTM**      | Deep learning model | Good for sequential data | Requires large dataset and computational power |

---

**End of README**  
