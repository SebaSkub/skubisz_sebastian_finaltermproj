
# Water Quality Classification Project  
**Author:** Sebastian Skubisz  

---

## 1) Folder Layout
```
skubisz_sebastian_finaltermproj/
│
├─ training.py                          # main script
├─ datasets/                            # input dataset
│   └─ waterQuality1.csv                # the dataset file
├─ images/                              # output images
│   ├─ class_distribution_full.png      # Class distribution plot
│   ├─ correlation_heatmap.png          # Correlation heatmap plot
│   ├─ feature_histograms.png           # Feature histograms plot
│   ├─ pairplot_is_safe.png             # Pairplot of features
│   ├─ smote_class_balance.png          # SMOTE class balance plot
│   ├─ roc_knn.png                      # KNN ROC curve plot
│   ├─ roc_rf.png                       # Random Forest ROC curve plot
│   └─ roc_lstm.png                     # LSTM ROC curve plot
├─ requirements.txt                    # dependencies
└─ README.md                            # this file
```

---

## 2) Dataset
The dataset used is the **Water Quality dataset**, which is created from imaginary data of water quality in an urban environment. The dataset contains 21 columns and 8000 rows.

Dataset source: [Water Quality Dataset - Kaggle](https://www.kaggle.com/datasets/mssmartypants/water-quality)  

---

## 3) Environment Setup

Use **Python 3.9–3.13**.

### 🪟 Windows PowerShell
```bash
Install python from Windows Store 3.13
python -m venv .venv
.venv\Scriptsctivate
pip install --upgrade pip
pip install -r requirements.txt
```

### 🐧 macOS/Linux
```bash
sudo apt install python3.13
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### requirements.txt
```
pandas
numpy
matplotlib
seaborn
sklearn
imblearn
keras
tensorflow
```

---

## 4) How to Run

Run the main script interactively from the project root:
```bash
python training.py
```

### Example Session

The script will:

- Run **KNN**, **Random Forest**, and **LSTM**
- Print evaluation metrics such as accuracy, precision, recall, F1 score, and more
- Save all results and plots in the folder

---

## 5) Output Structure

All results and images are automatically saved to:
```
skubisz_sebastian_finaltermproj/
├─ class_distribution_full.png
├─ correlation_heatmap.png
├─ feature_histograms.png
├─ pairplot_is_safe.png
├─ smote_class_balance.png
├─ roc_knn.png
├─ roc_rf.png
└─ roc_lstm.png
```

---

## 6) Running the Script
The script processes the `waterQuality1.csv` dataset, applies SMOTE for class balancing, trains models using **KNN**, **Random Forest**, and **LSTM**, and then evaluates the models based on various metrics. It also plots ROC curves to compare model performance.

---

## 7) Challenges Encountered
- **Imbalanced Dataset**: Used SMOTE to balance the class distribution.
- **Missing/Invalid Data**: Implemented data preprocessing steps to handle missing values and outliers.
- **Model Hyperparameters**: Fine-tuned KNN and Random Forest models using GridSearchCV for optimal performance.

---

## 8) Key Concepts
### Random Forest
A versatile machine learning algorithm that combines multiple decision trees to improve prediction accuracy.

### LSTM (Long Short-Term Memory)
A type of recurrent neural network (RNN) ideal for sequence prediction tasks, used here to capture temporal relationships in the data.

### KNN (K-Nearest Neighbors)
A simple, instance-based learning algorithm used for classification tasks based on proximity.

---

**End of README**
