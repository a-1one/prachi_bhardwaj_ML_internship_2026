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
    StratifiedKFold,
    cross_val_score,
    GridSearchCV
)

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# =====================================================
# Question 1
# =====================================================

# Load Dataset
df = pd.read_csv("Loan prediction.csv")

print("="*60)
print("First 10 Records")
print("="*60)
print(df.head(10))

print("\nColumns:")
print(df.columns)

print("\nShape of Dataset:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

# =====================================================
# Question 2 : Data Preprocessing
# =====================================================

# Drop Loan_ID because it is not useful
df.drop("Loan_ID", axis=1, inplace=True)

# Separate categorical and numerical columns
cat_cols = df.select_dtypes(include="object").columns

num_cols = df.select_dtypes(include=np.number).columns

# Fill missing numerical values with median
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Fill missing categorical values with mode
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Encode categorical variables
le = LabelEncoder()

for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Separate Features and Target
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# Standardize Features
scaler = StandardScaler()

X = scaler.fit_transform(X)

print("\nPreprocessing Completed Successfully!")

# =====================================================
# Question 3 : Exploratory Data Analysis (EDA)
# =====================================================

# Reload dataset for plotting with original categorical values
eda_df = pd.read_csv("Loan prediction.csv")

# Fill missing values
for col in eda_df.select_dtypes(include=np.number).columns:
    eda_df[col].fillna(eda_df[col].median(), inplace=True)

for col in eda_df.select_dtypes(include="object").columns:
    eda_df[col].fillna(eda_df[col].mode()[0], inplace=True)

# -------------------------------
# 1. Loan Approval Distribution
# -------------------------------
plt.figure(figsize=(6,4))
sns.countplot(x="Loan_Status", data=eda_df)
plt.title("Loan Approval Distribution")
plt.show()

# -------------------------------
# 2. Applicant Income vs Loan Approval
# -------------------------------
plt.figure(figsize=(8,5))
sns.boxplot(x="Loan_Status", y="ApplicantIncome", data=eda_df)
plt.title("Applicant Income vs Loan Approval")
plt.show()

# -------------------------------
# 3. Credit History vs Loan Approval
# -------------------------------
plt.figure(figsize=(6,4))
sns.countplot(x="Credit_History", hue="Loan_Status", data=eda_df)
plt.title("Credit History vs Loan Approval")
plt.show()

# -------------------------------
# 4. Education vs Loan Approval
# -------------------------------
plt.figure(figsize=(6,4))
sns.countplot(x="Education", hue="Loan_Status", data=eda_df)
plt.title("Education Level vs Loan Approval")
plt.show()

# -------------------------------
# 5. Property Area vs Loan Approval
# -------------------------------
plt.figure(figsize=(7,4))
sns.countplot(x="Property_Area", hue="Loan_Status", data=eda_df)
plt.title("Property Area vs Loan Approval")
plt.show()

print("\n========== EDA Observations ==========")
print("1. Credit History appears to be the most influential feature.")
print("2. Applicants with Credit_History = 1 have much higher loan approval.")
print("3. Higher income generally improves approval chances.")
print("4. Graduates have slightly higher approval rates.")
print("5. Semiurban and Urban applicants receive more approvals than Rural applicants.")

# =====================================================
# Question 4 : Train-Test Split and Model Training
# =====================================================

# Split the dataset into training and testing sets (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Dictionary to store model results
results = {}

# =====================================================
# Logistic Regression
# =====================================================

log_model = LogisticRegression(max_iter=1000)

log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)

results["Logistic Regression"] = {
    "Accuracy": accuracy_score(y_test, y_pred_log),
    "Precision": precision_score(y_test, y_pred_log),
    "Recall": recall_score(y_test, y_pred_log),
    "F1 Score": f1_score(y_test, y_pred_log)
}

# =====================================================
# Decision Tree
# =====================================================

dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)

results["Decision Tree"] = {
    "Accuracy": accuracy_score(y_test, y_pred_dt),
    "Precision": precision_score(y_test, y_pred_dt),
    "Recall": recall_score(y_test, y_pred_dt),
    "F1 Score": f1_score(y_test, y_pred_dt)
}

# =====================================================
# Random Forest
# =====================================================

rf_model = RandomForestClassifier(random_state=42)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

results["Random Forest"] = {
    "Accuracy": accuracy_score(y_test, y_pred_rf),
    "Precision": precision_score(y_test, y_pred_rf),
    "Recall": recall_score(y_test, y_pred_rf),
    "F1 Score": f1_score(y_test, y_pred_rf)
}

# =====================================================
# Comparison Table
# =====================================================

comparison = pd.DataFrame(results).T

print("\n" + "="*70)
print("Model Comparison")
print("="*70)
print(comparison)

# =====================================================
# Classification Reports
# =====================================================

print("\nClassification Report - Logistic Regression")
print(classification_report(y_test, y_pred_log))

print("\nClassification Report - Decision Tree")
print(classification_report(y_test, y_pred_dt))

print("\nClassification Report - Random Forest")
print(classification_report(y_test, y_pred_rf))

# =====================================================
# Question 5 : Best Performing Model
# =====================================================

best_model = comparison["Accuracy"].idxmax()

print("\n" + "="*70)
print("Question 5")
print("="*70)

print(f"Best Performing Model : {best_model}")

print("\nStrengths and Limitations")

print("\nLogistic Regression")
print("- Simple and fast.")
print("- Easy to interpret.")
print("- Works well for linear relationships.")

print("\nDecision Tree")
print("- Easy to visualize.")
print("- Can capture non-linear patterns.")
print("- May overfit if tree becomes deep.")

print("\nRandom Forest")
print("- More accurate than a single tree.")
print("- Reduces overfitting.")
print("- Computationally more expensive.")

# =====================================================
# Question 6 : Stratified 5-Fold Cross Validation
# =====================================================

print("\n" + "="*70)
print("Question 6 : Stratified 5-Fold Cross Validation")
print("="*70)

# Create Stratified K-Fold object
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Logistic Regression Cross Validation
log_cv = cross_val_score(log_model, X, y, cv=skf, scoring='accuracy')

# Decision Tree Cross Validation
dt_cv = cross_val_score(dt_model, X, y, cv=skf, scoring='accuracy')

# Random Forest Cross Validation
rf_cv = cross_val_score(rf_model, X, y, cv=skf, scoring='accuracy')

# -----------------------------------------------------
# Print Fold Accuracies
# -----------------------------------------------------

print("\nLogistic Regression Fold Accuracies")
for i, score in enumerate(log_cv):
    print(f"Fold {i+1}: {score:.4f}")

print(f"Mean Accuracy : {log_cv.mean():.4f}")
print(f"Standard Deviation : {log_cv.std():.4f}")

print("\nDecision Tree Fold Accuracies")
for i, score in enumerate(dt_cv):
    print(f"Fold {i+1}: {score:.4f}")

print(f"Mean Accuracy : {dt_cv.mean():.4f}")
print(f"Standard Deviation : {dt_cv.std():.4f}")

print("\nRandom Forest Fold Accuracies")
for i, score in enumerate(rf_cv):
    print(f"Fold {i+1}: {score:.4f}")

print(f"Mean Accuracy : {rf_cv.mean():.4f}")
print(f"Standard Deviation : {rf_cv.std():.4f}")

# -----------------------------------------------------
# Comparison Table
# -----------------------------------------------------

cv_results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "Train-Test Accuracy": [
        results["Logistic Regression"]["Accuracy"],
        results["Decision Tree"]["Accuracy"],
        results["Random Forest"]["Accuracy"]
    ],
    "CV Mean Accuracy": [
        log_cv.mean(),
        dt_cv.mean(),
        rf_cv.mean()
    ],
    "CV Std Dev": [
        log_cv.std(),
        dt_cv.std(),
        rf_cv.std()
    ]
})

print("\n" + "="*70)
print("Cross Validation Comparison")
print("="*70)
print(cv_results)

# -----------------------------------------------------
# Theory Answers
# -----------------------------------------------------

print("\nAnswers")

print("\n1. Why is Stratified K-Fold preferred?")
print("Stratified K-Fold maintains the same class distribution")
print("in every fold. Since Loan_Status is imbalanced,")
print("it provides a more reliable evaluation than normal K-Fold.")

most_consistent = cv_results.loc[
    cv_results["CV Std Dev"].idxmin(),
    "Model"
]

print(f"\n2. Most Consistent Model : {most_consistent}")
print("Reason : It has the lowest standard deviation among all models.")

# =====================================================
# Question 7 : Hyperparameter Tuning using GridSearchCV
# =====================================================

print("\n" + "="*70)
print("Question 7 : Hyperparameter Tuning")
print("="*70)

# Parameter Grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5, 10]
}

# Grid Search
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Best Model
best_rf = grid_search.best_estimator_

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross Validation Score:")
print(grid_search.best_score_)

# Evaluate Tuned Model
y_pred_best = best_rf.predict(X_test)

before_accuracy = accuracy_score(y_test, y_pred_rf)
after_accuracy = accuracy_score(y_test, y_pred_best)

print("\nAccuracy Before Tuning :", round(before_accuracy, 4))
print("Accuracy After Tuning :", round(after_accuracy, 4))

print("\nClassification Report After Tuning")
print(classification_report(y_test, y_pred_best))

# =====================================================
# Question 8 : Bias-Variance Tradeoff
# =====================================================

print("\n" + "="*70)
print("Question 8 : Bias-Variance Tradeoff")
print("="*70)

depths = [2, 5, 15]

bias_variance = []

for depth in depths:

    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    bias_variance.append([
        depth,
        train_acc,
        test_acc
    ])

# Create Table
bias_df = pd.DataFrame(
    bias_variance,
    columns=[
        "Max Depth",
        "Training Accuracy",
        "Testing Accuracy"
    ]
)

print("\nBias-Variance Comparison")
print(bias_df)

# -----------------------------------------------------
# Answers
# -----------------------------------------------------

print("\nAnswers")

print("\n1. Underfitting Model")
print("Max Depth = 2")
print("Reason: Tree is too simple and cannot capture enough patterns.")

print("\n2. Overfitting Model")
print("Max Depth = 15")
print("Reason: Training accuracy is much higher than testing accuracy.")

print("\n3. Best Bias-Variance Balance")
print("Max Depth = 5")
print("Reason: Provides a good balance between training and testing accuracy.")

print("\nAssignment Completed Successfully!")