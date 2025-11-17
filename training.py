#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from collections import Counter

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import brier_score_loss
from sklearn.metrics import auc

from keras.models import Sequential
from keras.layers import Dense, LSTM

from imblearn.over_sampling import SMOTE  # SMOTE integration

# 1. Loading Data And Preprocessing

# For quick tests, you can switch this to 'waterQuality1_sample.csv'
diab = pd.read_csv('waterQuality1.csv')
print("\nLoaded dataset with shape:", diab.shape)
print("\nDataset info:")
diab.info()

feature_cols = [
    'aluminium', 'ammonia', 'arsenic', 'barium', 'cadmium',
    'chloramine', 'chromium', 'copper', 'flouride', 'bacteria',
    'viruses', 'lead', 'nitrates', 'nitrites', 'mercury',
    'perchlorate', 'radium', 'selenium', 'silver', 'uranium'
]

print("\nConverting feature columns to numeric...")
diab[feature_cols] = diab[feature_cols].apply(pd.to_numeric, errors='coerce')

print("Converting 'is_safe' to numeric and dropping invalid rows...")
diab['is_safe'] = pd.to_numeric(diab['is_safe'], errors='coerce')
before_drop = diab.shape[0]
diab = diab.dropna(subset=['is_safe'])
after_drop = diab.shape[0]
print(f"Dropped {before_drop - after_drop} rows with invalid 'is_safe' values.")
diab['is_safe'] = diab['is_safe'].astype(int)

def impute_missing_values(dataframe):
    print("\nImputing zeros/NaNs in feature columns with median values...")
    for column in feature_cols:
        zero_count = (dataframe[column] == 0).sum()
        nan_count = dataframe[column].isna().sum()
        if zero_count > 0 or nan_count > 0:
            print(f" - Column '{column}': {zero_count} zeros, {nan_count} NaNs before imputation")
        dataframe.loc[dataframe[column] == 0, column] = np.nan
        dataframe[column].fillna(dataframe[column].median(), inplace=True)
    return dataframe

diab = impute_missing_values(diab)
print("Finished preprocessing. New dataset shape:", diab.shape)
print("\nPreview of preprocessed data:")
print(diab.head())

# 2. Separating Dataset into Features and Output Label

features = diab.iloc[:, :-1]
labels = diab.iloc[:, -1]

print("Features shape:", features.shape)
print("Labels shape:", labels.shape)

print("\nClass distribution (is_safe):")
print(labels.value_counts())
positive_outcomes, negative_outcomes = labels.value_counts()
total_samples = labels.count()
print('\n----------Checking for Data Imbalance------------')
print('Number of Positive Outcomes: ', positive_outcomes)
print('Percentage of Positive Outcomes: {}%'.format(round((positive_outcomes / total_samples) * 100, 2)))
print('Number of Negative Outcomes : ', negative_outcomes)
print('Percentage of Negative Outcomes: {}%'.format(round((negative_outcomes / total_samples) * 100, 2)))
print('-------------------------------------------------\n')

sns.countplot(x=labels, label="Count")
plt.title("Class Distribution (is_safe) - Full Dataset")
plt.savefig("class_distribution_full.png", bbox_inches="tight")
plt.show()


# 3. Checking Correlation Between Attributes

correlation_matrix = features.corr()
fig, axis = plt.subplots(figsize=(12, 12))
sns.heatmap(correlation_matrix, annot=True, linewidths=.5, fmt='.2f', ax=axis)
plt.title("Correlation Heatmap (Features Only)")
plt.savefig("correlation_heatmap.png", bbox_inches="tight")
plt.show()

full_corr = diab.corr()
corr_with_target = full_corr['is_safe'].drop('is_safe')
max_corr_feat = corr_with_target.abs().idxmax()
print("\nCorrelation of each feature with 'is_safe':")
print(corr_with_target.sort_values(ascending=False))
print(f"\nHighest absolute correlation with 'is_safe': {max_corr_feat} "
      f"(corr = {corr_with_target[max_corr_feat]:.4f})")

# 4. Visualizing Distributions (Histograms) & Symmetry

features.hist(figsize=(10, 10))
plt.suptitle("Histograms of Features", y=1.02)
plt.tight_layout()
plt.savefig("feature_histograms.png", bbox_inches="tight")
plt.show()

print("Histograms plotted for all feature columns.")

skewness = features.skew()
print("\nSkewness of each feature:")
print(skewness.sort_values())

sym_threshold = 0.5
approx_symmetric = skewness[skewness.abs() < sym_threshold].index.tolist()
print(f"\nFeatures with approximate symmetry (|skew| < {sym_threshold}):")
print(approx_symmetric if approx_symmetric else "None")

# 5. Pairwise Relationships (Pairplot)

sns.pairplot(diab, hue='is_safe')
plt.suptitle("Pairplot of Features Colored by is_safe", y=1.02)
plt.savefig("pairplot_is_safe.png", bbox_inches="tight")
plt.show()

print("Pairplot generated.")

# 6. Train-Test Split and Normalization (70/30 Split)

features_train_all, features_test_all, labels_train_all, labels_test_all = train_test_split(
    features, labels, test_size=0.3, random_state=42, stratify=labels
)

for dataset in [features_train_all, features_test_all, labels_train_all, labels_test_all]:
    dataset.reset_index(drop=True, inplace=True)

print("Training set shape:", features_train_all.shape)
print("Test set shape:", features_test_all.shape)

features_train_all_std = (features_train_all - features_train_all.mean()) / features_train_all.std()
features_test_all_std = (features_test_all - features_test_all.mean()) / features_test_all.std()

print("\nTraining data (standardized) summary:")
print(features_train_all_std.describe())

n_features = features_train_all_std.shape[1]
print("Number of features (for LSTM):", n_features)

# 7. Visualizing Effect of SMOTE on Training Data

smote = SMOTE(random_state=42)

print("\nOriginal training label distribution:")
orig_counts = Counter(labels_train_all)
print(orig_counts)

# Apply SMOTE to the training data only
features_train_bal_all, labels_train_bal_all = smote.fit_resample(features_train_all_std, labels_train_all)

print("\nBalanced training label distribution after SMOTE:")
bal_counts = Counter(labels_train_bal_all)
print(bal_counts)

# Print class distribution before and after SMOTE
print("\nClass distribution before and after SMOTE:")
print(f"Original training set class distribution: {orig_counts}")
print(f"Balanced training set class distribution after SMOTE: {bal_counts}")

# Visualize the effect of SMOTE
smote_df = pd.DataFrame({
    'Class': ['0', '1', '0', '1'],
    'Count': [
        orig_counts.get(0, 0), orig_counts.get(1, 0),
        bal_counts.get(0, 0), bal_counts.get(1, 0)
    ],
    'Dataset': ['Original Train', 'Original Train',
                'SMOTE Train', 'SMOTE Train']
})

plt.figure(figsize=(6, 4))
sns.barplot(data=smote_df, x='Class', y='Count', hue='Dataset')
plt.title("Class Distribution Before and After SMOTE (Train Set)")
plt.savefig("smote_class_balance.png", bbox_inches="tight")
plt.show()

# 8. Defining Metric Calculation and Model Evaluation Functions

def calc_metrics(confusion_matrix_):
    TP, FN = confusion_matrix_[0][0], confusion_matrix_[0][1]
    FP, TN = confusion_matrix_[1][0], confusion_matrix_[1][1]

    TPR = TP / (TP + FN) if TP + FN > 0 else 0
    TNR = TN / (TN + FP) if TN + FP > 0 else 0
    FPR = FP / (TN + FP) if TN + FP > 0 else 0
    FNR = FN / (TP + FN) if TP + FN > 0 else 0
    Precision = TP / (TP + FP) if TP + FP > 0 else 0
    F1_measure = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
    Accuracy = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0
    Error_rate = 1 - Accuracy
    BACC = (TPR + TNR) / 2
    TSS = TPR - FPR
    denom_hss = ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN))
    HSS = 2 * (TP * TN - FP * FN) / denom_hss if denom_hss != 0 else 0

    metrics = [
        TP, TN, FP, FN, TPR, TNR, FPR, FNR,
        Precision, F1_measure, Accuracy, Error_rate,
        BACC, TSS, HSS
    ]
    return metrics

def get_metrics(model, features_train, features_test, labels_train, labels_test, LSTM_flag):

    metrics = []

    if LSTM_flag == 1:
        Xtrain, Xtest, ytrain, ytest = map(
            np.array, [features_train, features_test, labels_train, labels_test]
        )

        shape = Xtrain.shape
        Xtrain_reshaped = Xtrain.reshape(len(Xtrain), shape[1], 1)
        Xtest_reshaped = Xtest.reshape(len(Xtest), shape[1], 1)

        model.fit(
            Xtrain_reshaped, ytrain,
            epochs=50,
            validation_data=(Xtest_reshaped, ytest),
            verbose=0
        )

        lstm_scores = model.evaluate(Xtest_reshaped, ytest, verbose=0)
        predict_prob = model.predict(Xtest_reshaped, verbose=0)
        pred_labels = (predict_prob > 0.5).astype(int)

        matrix = confusion_matrix(ytest, pred_labels, labels=[1, 0])
        brier = brier_score_loss(ytest, predict_prob)
        roc_auc = roc_auc_score(ytest, predict_prob)

        metrics.extend(calc_metrics(matrix))
        metrics.extend([brier, roc_auc, lstm_scores[1]])

    else:
        model.fit(features_train, labels_train)
        predicted = model.predict(features_test)
        matrix = confusion_matrix(labels_test, predicted, labels=[1, 0])
        proba = model.predict_proba(features_test)[:, 1]
        brier = brier_score_loss(labels_test, proba)
        roc_auc = roc_auc_score(labels_test, proba)

        metrics.extend(calc_metrics(matrix))
        metrics.extend([brier, roc_auc, model.score(features_test, labels_test)])

    return metrics

# 9. Hyperparameter Tuning (KNN & Random Forest)

knn_parameters = {"n_neighbors": list(range(1, 16))}
knn_model = KNeighborsClassifier(n_jobs=-1)
knn_cv = GridSearchCV(knn_model, knn_parameters, cv=10, n_jobs=-1)
knn_cv.fit(features_train_all_std, labels_train_all)
best_n_neighbors = knn_cv.best_params_["n_neighbors"]
print("Best KNN n_neighbors:", best_n_neighbors)

param_grid_rf = {
    "n_estimators": list(range(10, 101, 10)),
    "min_samples_split": [2, 4, 6, 8, 10]
}
rf_classifier = RandomForestClassifier(n_jobs=-1)
grid_search_rf = GridSearchCV(
    estimator=rf_classifier,
    param_grid=param_grid_rf,
    cv=10,
    n_jobs=-1
)
grid_search_rf.fit(features_train_all_std, labels_train_all)
best_rf_params = grid_search_rf.best_params_
min_samples_split = best_rf_params["min_samples_split"]
n_estimators = best_rf_params["n_estimators"]
print("Best RF params:", best_rf_params)

# 10. 10-fold Stratified Cross-Validation with SMOTE

cv_stratified = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

metric_columns = [
    'TP', 'TN', 'FP', 'FN', 'TPR', 'TNR', 'FPR', 'FNR',
    'Precision', 'F1_measure', 'Accuracy', 'Error_rate',
    'BACC', 'TSS', 'HSS', 'Brier_score', 'AUC', 'Acc_by_package_fn'
]

knn_metrics_list = []
rf_metrics_list = []
lstm_metrics_list = []

for iter_num, (train_index, test_index) in enumerate(
    cv_stratified.split(features_train_all_std, labels_train_all), start=1
):
    print(f"\n--- Cross-Validation Iteration {iter_num} ---")

    knn_model = KNeighborsClassifier(n_neighbors=best_n_neighbors, n_jobs=-1)
    rf_model = RandomForestClassifier(
        min_samples_split=min_samples_split,
        n_estimators=n_estimators,
        n_jobs=-1
    )

    lstm_model = Sequential()
    lstm_model.add(
        LSTM(
            64,
            activation='relu',
            input_shape=(n_features, 1),
            return_sequences=False
        )
    )
    lstm_model.add(Dense(1, activation='sigmoid'))
    lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    features_train = features_train_all_std.iloc[train_index, :]
    features_test = features_train_all_std.iloc[test_index, :]
    labels_train = labels_train_all[train_index]
    labels_test = labels_train_all[test_index]

    features_train_bal, labels_train_bal = smote.fit_resample(features_train, labels_train)

    knn_metrics = get_metrics(knn_model, features_train_bal, features_test, labels_train_bal, labels_test, 0)
    rf_metrics = get_metrics(rf_model, features_train_bal, features_test, labels_train_bal, labels_test, 0)
    lstm_metrics = get_metrics(lstm_model, features_train_bal, features_test, labels_train_bal, labels_test, 1)

    knn_metrics_list.append(knn_metrics)
    rf_metrics_list.append(rf_metrics)
    lstm_metrics_list.append(lstm_metrics)

    iter_df = pd.DataFrame(
        {
            "KNN": knn_metrics,
            "RF": rf_metrics,
            "LSTM": lstm_metrics
        },
        index=metric_columns
    )

    print(f"\nIteration {iter_num}:")
    print(f"----- Metrics for all Algorithms in Iteration {iter_num} -----")
    print(iter_df.round(2).to_string())

# 11. Average Performance & Per-Algorithm Iteration Tables

metric_index_df = [
    'iter1', 'iter2', 'iter3', 'iter4', 'iter5',
    'iter6', 'iter7', 'iter8', 'iter9', 'iter10'
]

knn_metrics_df = pd.DataFrame(knn_metrics_list, columns=metric_columns, index=metric_index_df)
rf_metrics_df = pd.DataFrame(rf_metrics_list, columns=metric_columns, index=metric_index_df)
lstm_metrics_df = pd.DataFrame(lstm_metrics_list, columns=metric_columns, index=metric_index_df)

print("\nMetrics for Algorithm 1 (KNN):")
print(knn_metrics_df.T.round(2).to_string())

print("\nMetrics for Algorithm 2 (RF):")
print(rf_metrics_df.T.round(2).to_string())

print("\nMetrics for Algorithm 3 (LSTM):")
print(lstm_metrics_df.T.round(2).to_string())

knn_avg_df = knn_metrics_df.mean()
rf_avg_df = rf_metrics_df.mean()
lstm_avg_df = lstm_metrics_df.mean()

avg_performance_df = pd.DataFrame(
    {"KNN": knn_avg_df, "RF": rf_avg_df, "LSTM": lstm_avg_df},
    index=metric_columns
)

print("\nAverage performance over 10 folds (rounded):")
print(avg_performance_df.round(3))

# 12. Final ROC Curves on Held-out Test Set

print("\nTraining final KNN on balanced data and plotting ROC...")
knn_model = KNeighborsClassifier(n_neighbors=best_n_neighbors, n_jobs=-1)
knn_model.fit(features_train_bal_all, labels_train_bal_all)
y_score = knn_model.predict_proba(features_test_all_std)[:, 1]
fpr, tpr, _ = roc_curve(labels_test_all, y_score)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, label=f"KNN AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("KNN ROC Curve")
plt.legend()
plt.savefig("roc_knn.png", bbox_inches="tight")
plt.show()

print(f"KNN Test AUC: {roc_auc:.4f}")

print("\nTraining final Random Forest on balanced data and plotting ROC...")
rf_model = RandomForestClassifier(
    min_samples_split=min_samples_split,
    n_estimators=n_estimators,
    n_jobs=-1
)
rf_model.fit(features_train_bal_all, labels_train_bal_all)
y_score_rf = rf_model.predict_proba(features_test_all_std)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(labels_test_all, y_score_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)
plt.figure()
plt.plot(fpr_rf, tpr_rf, label=f"RF AUC = {roc_auc_rf:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("Random Forest ROC Curve")
plt.legend()
plt.savefig("roc_rf.png", bbox_inches="tight")
plt.show()

print(f"Random Forest Test AUC: {roc_auc_rf:.4f}")

print("\nTraining final LSTM on balanced data and plotting ROC...")
lstm_model = Sequential()
lstm_model.add(
    LSTM(
        64,
        activation='relu',
        input_shape=(n_features, 1),
        return_sequences=False
    )
)
lstm_model.add(Dense(1, activation='sigmoid'))
lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

X_train_array = np.array(features_train_bal_all)
X_test_array = features_test_all_std.to_numpy()
y_train_array = np.array(labels_train_bal_all)
y_test_array = labels_test_all.to_numpy()

input_train = X_train_array.reshape(len(X_train_array), n_features, 1)
input_test = X_test_array.reshape(len(X_test_array), n_features, 1)

lstm_model.fit(input_train, y_train_array, epochs=50,
               validation_data=(input_test, y_test_array), verbose=0)

predict_lstm = lstm_model.predict(input_test, verbose=0)
fpr_lstm, tpr_lstm, _ = roc_curve(labels_test_all, predict_lstm)
roc_auc_lstm = auc(fpr_lstm, tpr_lstm)

plt.figure()
plt.plot(fpr_lstm, tpr_lstm, label=f"LSTM AUC = {roc_auc_lstm:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("LSTM ROC Curve")
plt.legend()
plt.savefig("roc_lstm.png", bbox_inches="tight")
plt.show()

print(f"LSTM Test AUC: {roc_auc_lstm:.4f}")

