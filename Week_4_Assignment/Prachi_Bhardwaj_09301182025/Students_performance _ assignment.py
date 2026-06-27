# Student Name: Prachi Bhardwaj
# Enrollment Number: 09301182025
# College Name: IGDTUW




# Import Libraries

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    GridSearchCV,
    learning_curve
)

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# Load Dataset
# ============================================================

df = pd.read_csv("StudentsPerformance.csv")

print("="*70)
print("First 10 Records")
print("="*70)

print(df.head(10))

print("\nDataset Shape")
print(df.shape)

print("\nColumns")
print(df.columns)

print("\nMissing Values")
print(df.isnull().sum())

print("\nStatistical Summary")
print(df.describe())

# ============================================================
# Create Pass / Fail Target
# ============================================================

# Average Marks

df["Average Score"] = (
    df["math score"] +
    df["reading score"] +
    df["writing score"]
) / 3

# Pass if average >= 50

df["Result"] = df["Average Score"].apply(
    lambda x: "Pass" if x >= 50 else "Fail"
)

print("\nResult Distribution")
print(df["Result"].value_counts())

# ============================================================
# Encode Target Variable
# ============================================================

target_encoder = LabelEncoder()

df["Result"] = target_encoder.fit_transform(df["Result"])

# ============================================================
# Encode Categorical Features
# ============================================================

categorical_columns = [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course"
]

encoder = LabelEncoder()

for column in categorical_columns:
    df[column] = encoder.fit_transform(df[column])

# ============================================================
# Feature Matrix and Target Variable
# ============================================================

X = df.drop(
    ["Result", "Average Score"],
    axis=1
)

y = df["Result"]

# ============================================================
# Feature Scaling
# ============================================================

scaler = StandardScaler()

X = scaler.fit_transform(X)

print("\nData Preprocessing Completed Successfully!")

# ============================================================
# Part 2 : Exploratory Data Analysis (EDA)
# ============================================================

# Reload original dataset for plotting
eda_df = pd.read_csv("StudentsPerformance.csv")

# Create Average Score
eda_df["Average Score"] = (
    eda_df["math score"] +
    eda_df["reading score"] +
    eda_df["writing score"]
) / 3

# Create Pass/Fail Target
eda_df["Result"] = eda_df["Average Score"].apply(
    lambda x: "Pass" if x >= 50 else "Fail"
)

# ------------------------------------------------------------
# 1. Pass / Fail Distribution
# ------------------------------------------------------------

plt.figure(figsize=(6,5))
sns.countplot(x="Result", data=eda_df)
plt.title("Pass / Fail Distribution")
plt.xlabel("Result")
plt.ylabel("Count")
plt.show()

# ------------------------------------------------------------
# 2. Gender vs Average Score
# ------------------------------------------------------------

plt.figure(figsize=(6,5))
sns.boxplot(x="gender", y="Average Score", data=eda_df)
plt.title("Gender vs Average Score")
plt.show()

# ------------------------------------------------------------
# 3. Lunch Type vs Average Score
# ------------------------------------------------------------

plt.figure(figsize=(6,5))
sns.boxplot(x="lunch", y="Average Score", data=eda_df)
plt.title("Lunch Type vs Average Score")
plt.xticks(rotation=10)
plt.show()

# ------------------------------------------------------------
# 4. Test Preparation Course vs Average Score
# ------------------------------------------------------------

plt.figure(figsize=(6,5))
sns.boxplot(
    x="test preparation course",
    y="Average Score",
    data=eda_df
)
plt.title("Test Preparation Course vs Average Score")
plt.xticks(rotation=10)
plt.show()

# ------------------------------------------------------------
# 5. Correlation Heatmap
# ------------------------------------------------------------

plt.figure(figsize=(8,6))

corr = eda_df[[
    "math score",
    "reading score",
    "writing score",
    "Average Score"
]].corr()

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.show()

# ------------------------------------------------------------
# 6. Score Distribution
# ------------------------------------------------------------

plt.figure(figsize=(8,5))

sns.histplot(
    eda_df["Average Score"],
    bins=20,
    kde=True
)

plt.title("Distribution of Average Scores")
plt.xlabel("Average Score")
plt.ylabel("Frequency")
plt.show()

# ============================================================
# EDA Observations
# ============================================================

print("\n" + "="*70)
print("EDA Observations")
print("="*70)

print("1. Most students passed the examination.")
print("2. Students completing the test preparation course")
print("   generally scored higher.")
print("3. Students receiving standard lunch performed")
print("   better on average.")
print("4. Reading and writing scores are highly correlated.")
print("5. Average score has a strong positive relationship")
print("   with all three subject scores.")

# ============================================================
# Part 3 : Model Training and Evaluation
# ============================================================

# Split Dataset (80:20)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Dictionary to store results

results = {}

# ============================================================
# Logistic Regression
# ============================================================

print("\n" + "="*70)
print("Logistic Regression")
print("="*70)

lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

lr_acc = accuracy_score(y_test, lr_pred)
lr_pre = precision_score(y_test, lr_pred)
lr_rec = recall_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)

results["Logistic Regression"] = [
    lr_acc,
    lr_pre,
    lr_rec,
    lr_f1
]

print("Accuracy :", lr_acc)
print("Precision:", lr_pre)
print("Recall   :", lr_rec)
print("F1 Score :", lr_f1)

print("\nClassification Report")
print(classification_report(y_test, lr_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, lr_pred))

# ============================================================
# Decision Tree
# ============================================================

print("\n" + "="*70)
print("Decision Tree")
print("="*70)

dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

dt_acc = accuracy_score(y_test, dt_pred)
dt_pre = precision_score(y_test, dt_pred)
dt_rec = recall_score(y_test, dt_pred)
dt_f1 = f1_score(y_test, dt_pred)

results["Decision Tree"] = [
    dt_acc,
    dt_pre,
    dt_rec,
    dt_f1
]

print("Accuracy :", dt_acc)
print("Precision:", dt_pre)
print("Recall   :", dt_rec)
print("F1 Score :", dt_f1)

print("\nClassification Report")
print(classification_report(y_test, dt_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, dt_pred))

# ============================================================
# Random Forest
# ============================================================

print("\n" + "="*70)
print("Random Forest")
print("="*70)

rf_model = RandomForestClassifier(random_state=42)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred)
rf_pre = precision_score(y_test, rf_pred)
rf_rec = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

results["Random Forest"] = [
    rf_acc,
    rf_pre,
    rf_rec,
    rf_f1
]

print("Accuracy :", rf_acc)
print("Precision:", rf_pre)
print("Recall   :", rf_rec)
print("F1 Score :", rf_f1)

print("\nClassification Report")
print(classification_report(y_test, rf_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, rf_pred))

# ============================================================
# Model Comparison
# ============================================================

comparison = pd.DataFrame(
    results,
    index=[
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
).T

print("\n" + "="*70)
print("Model Comparison")
print("="*70)

print(comparison)

# ============================================================
# Best Model
# ============================================================

best_model = comparison["Accuracy"].idxmax()

print("\nBest Performing Model:", best_model)

# ============================================================
# Part 4 : Cross Validation and Hyperparameter Tuning
# ============================================================

print("\n" + "="*70)
print("5-Fold Cross Validation")
print("="*70)

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# -----------------------------
# Logistic Regression
# -----------------------------

lr_cv = cross_val_score(
    lr_model,
    X,
    y,
    cv=skf,
    scoring="accuracy"
)

print("\nLogistic Regression")
print("Fold Accuracies :", lr_cv)
print("Mean Accuracy :", lr_cv.mean())
print("Standard Deviation :", lr_cv.std())

# -----------------------------
# Decision Tree
# -----------------------------

dt_cv = cross_val_score(
    dt_model,
    X,
    y,
    cv=skf,
    scoring="accuracy"
)

print("\nDecision Tree")
print("Fold Accuracies :", dt_cv)
print("Mean Accuracy :", dt_cv.mean())
print("Standard Deviation :", dt_cv.std())

# -----------------------------
# Random Forest
# -----------------------------

rf_cv = cross_val_score(
    rf_model,
    X,
    y,
    cv=skf,
    scoring="accuracy"
)

print("\nRandom Forest")
print("Fold Accuracies :", rf_cv)
print("Mean Accuracy :", rf_cv.mean())
print("Standard Deviation :", rf_cv.std())

# ============================================================
# Hyperparameter Tuning
# Decision Tree
# ============================================================

print("\n" + "="*70)
print("Decision Tree Hyperparameter Tuning")
print("="*70)

dt_parameters = {
    "criterion": ["gini", "entropy"],
    "max_depth": [3, 5, 7, 10, None],
    "min_samples_split": [2, 5, 10]
}

dt_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    dt_parameters,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

dt_grid.fit(X_train, y_train)

best_dt = dt_grid.best_estimator_

print("Best Parameters")
print(dt_grid.best_params_)

print("\nBest Cross Validation Score")
print(dt_grid.best_score_)

dt_best_pred = best_dt.predict(X_test)

print("\nAccuracy After Tuning")
print(accuracy_score(y_test, dt_best_pred))

# ============================================================
# Hyperparameter Tuning
# Random Forest
# ============================================================

print("\n" + "="*70)
print("Random Forest Hyperparameter Tuning")
print("="*70)

rf_parameters = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10]
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_parameters,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)

best_rf = rf_grid.best_estimator_

print("Best Parameters")
print(rf_grid.best_params_)

print("\nBest Cross Validation Score")
print(rf_grid.best_score_)

rf_best_pred = best_rf.predict(X_test)

print("\nAccuracy After Tuning")
print(accuracy_score(y_test, rf_best_pred))

# ============================================================
# Part 4 : Cross Validation and Hyperparameter Tuning
# ============================================================

print("\n" + "="*70)
print("5-Fold Cross Validation")
print("="*70)

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# -----------------------------
# Logistic Regression
# -----------------------------

lr_cv = cross_val_score(
    lr_model,
    X,
    y,
    cv=skf,
    scoring="accuracy"
)

print("\nLogistic Regression")
print("Fold Accuracies :", lr_cv)
print("Mean Accuracy :", lr_cv.mean())
print("Standard Deviation :", lr_cv.std())

# -----------------------------
# Decision Tree
# -----------------------------

dt_cv = cross_val_score(
    dt_model,
    X,
    y,
    cv=skf,
    scoring="accuracy"
)

print("\nDecision Tree")
print("Fold Accuracies :", dt_cv)
print("Mean Accuracy :", dt_cv.mean())
print("Standard Deviation :", dt_cv.std())

# -----------------------------
# Random Forest
# -----------------------------

rf_cv = cross_val_score(
    rf_model,
    X,
    y,
    cv=skf,
    scoring="accuracy"
)

print("\nRandom Forest")
print("Fold Accuracies :", rf_cv)
print("Mean Accuracy :", rf_cv.mean())
print("Standard Deviation :", rf_cv.std())

# ============================================================
# Hyperparameter Tuning
# Decision Tree
# ============================================================

print("\n" + "="*70)
print("Decision Tree Hyperparameter Tuning")
print("="*70)

dt_parameters = {
    "criterion": ["gini", "entropy"],
    "max_depth": [3, 5, 7, 10, None],
    "min_samples_split": [2, 5, 10]
}

dt_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    dt_parameters,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

dt_grid.fit(X_train, y_train)

best_dt = dt_grid.best_estimator_

print("Best Parameters")
print(dt_grid.best_params_)

print("\nBest Cross Validation Score")
print(dt_grid.best_score_)

dt_best_pred = best_dt.predict(X_test)

print("\nAccuracy After Tuning")
print(accuracy_score(y_test, dt_best_pred))

# ============================================================
# Hyperparameter Tuning
# Random Forest
# ============================================================

print("\n" + "="*70)
print("Random Forest Hyperparameter Tuning")
print("="*70)

rf_parameters = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10]
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_parameters,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)

best_rf = rf_grid.best_estimator_

print("Best Parameters")
print(rf_grid.best_params_)

print("\nBest Cross Validation Score")
print(rf_grid.best_score_)

rf_best_pred = best_rf.predict(X_test)

print("\nAccuracy After Tuning")
print(accuracy_score(y_test, rf_best_pred))
