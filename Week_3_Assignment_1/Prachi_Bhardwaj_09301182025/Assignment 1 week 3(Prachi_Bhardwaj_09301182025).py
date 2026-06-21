# Student Name: Prachi Bhardwaj
# Enrollment Number: 09301182025
# College Name: IGDTUW

#ASSIGNMENT 1 WEEK 3

# ==========================
# IMPORT LIBRARIES
# ==========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

# ==========================
# LOAD DATASET
# ==========================
df = pd.read_csv(r"Assignment_1\Prachi_bhardwaj_09301182025\Dataset 3.csv")

# =====================================================
# Q1. DATASET OVERVIEW
# =====================================================
print("Rows and Columns:", df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 10 Records:")
print(df.head(10))

# =====================================================
# Q2. DATA TYPES AND MISSING VALUES
# =====================================================
print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

missing_cols = df.columns[df.isnull().sum() > 0]
print("\nColumns with Missing Values:")
print(missing_cols.tolist())

# =====================================================
# Q3. DESCRIPTIVE STATISTICS
# =====================================================
print("\nSummary Statistics:")
print(df.describe())

numeric_df = df.select_dtypes(include=np.number)

highest_mean = numeric_df.mean().idxmax()
highest_std = numeric_df.std().idxmax()

print("\nFeature with Highest Mean:", highest_mean)
print("Feature with Highest Std Dev:", highest_std)

# =====================================================
# Q4. DISTRIBUTION ANALYSIS
# =====================================================
columns = [
    "rainfall_mm",
    "temperature_c",
    "fertilizer_kg",
    "yield_ton_per_hectare"
]

for col in columns:
    plt.figure(figsize=(6,4))
    plt.hist(df[col], bins=20)
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

# =====================================================
# Q5. CROP TYPE ANALYSIS
# =====================================================
print("\nCrop Type Counts:")
print(df["crop_type"].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(x="crop_type", data=df)
plt.title("Crop Type Count")
plt.show()

print("Most Frequent Crop:")
print(df["crop_type"].mode()[0])

# =====================================================
# Q6. SOIL TYPE ANALYSIS
# =====================================================
print("\nSoil Type Frequency:")
print(df["soil_type"].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(x="soil_type", data=df)
plt.title("Soil Type Count")
plt.show()

print("Most Common Soil Type:")
print(df["soil_type"].mode()[0])

# =====================================================
# Q7. YIELD DISTRIBUTION
# =====================================================
plt.figure(figsize=(6,4))
plt.hist(df["yield_ton_per_hectare"], bins=20)
plt.title("Yield Distribution")
plt.xlabel("Yield")
plt.ylabel("Frequency")
plt.show()

# =====================================================
# Q8. SCATTER PLOT ANALYSIS
# =====================================================
plt.figure(figsize=(6,4))
plt.scatter(df["rainfall_mm"], df["yield_ton_per_hectare"])
plt.xlabel("Rainfall")
plt.ylabel("Yield")
plt.title("Rainfall vs Yield")
plt.show()

plt.figure(figsize=(6,4))
plt.scatter(df["fertilizer_kg"], df["yield_ton_per_hectare"])
plt.xlabel("Fertilizer")
plt.ylabel("Yield")
plt.title("Fertilizer vs Yield")
plt.show()

# =====================================================
# Q9. CORRELATION ANALYSIS
# =====================================================
corr_matrix = numeric_df.corr()

print("\nCorrelation Matrix:")
print(corr_matrix)

plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

yield_corr = corr_matrix["yield_ton_per_hectare"].sort_values(
    ascending=False
)

print("\nTop Correlations with Yield:")
print(yield_corr)

# =====================================================
# Q10. GROUP BASED ANALYSIS
# =====================================================
crop_yield = df.groupby("crop_type")["yield_ton_per_hectare"].mean()

soil_yield = df.groupby("soil_type")["yield_ton_per_hectare"].mean()

print("\nAverage Yield by Crop:")
print(crop_yield)

print("\nAverage Yield by Soil:")
print(soil_yield)

print("\nCrop with Highest Average Yield:")
print(crop_yield.idxmax())

print("\nSoil with Highest Average Yield:")
print(soil_yield.idxmax())

# =====================================================
# Q11. FEATURE ENCODING
# =====================================================
categorical_cols = df.select_dtypes(include="object").columns

print("\nCategorical Columns:")
print(categorical_cols.tolist())

df_encoded = pd.get_dummies(
    df,
    columns=categorical_cols,
    drop_first=True
)

print("\nFirst Five Rows of Encoded Dataset:")
print(df_encoded.head())

# =====================================================
# Q12. FEATURE SELECTION
# =====================================================
target_column = "yield_ton_per_hectare"

X = df_encoded.drop(target_column, axis=1)
y = df_encoded[target_column]

print("\nTarget Variable:", target_column)

# =====================================================
# Q13. TRAIN TEST SPLIT
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nShapes:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

# =====================================================
# Q14. LINEAR REGRESSION
# =====================================================
model = LinearRegression()

model.fit(X_train, y_train)

print("\nIntercept:")
print(model.intercept_)

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\nCoefficients:")
print(coefficients)

highest_positive = coefficients.loc[
    coefficients["Coefficient"].idxmax()
]

print("\nFeature with Highest Positive Coefficient:")
print(highest_positive) 