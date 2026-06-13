# Student Name: Prachi Bhardwaj
# Enrollment Number: 09301182025
# College Name: IGDTUW

#ASSIGNMENT 1 WEEK 2

import pandas as pd

# Load dataset
df = pd.read_csv("Dataset 2.csv")

# Q1: Display first 5 records
print("First 5 Records:")
print(df.head())

# Q2: Number of rows and columns
print("\nShape of Dataset:")
print(df.shape)

# Q3: Column names
print("\nColumn Names:")
print(df.columns.tolist())

# Q4: Numerical and categorical features
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()

print("\nNumerical Features:")
print(numerical_cols)

print("\nCategorical Features:")
print(categorical_cols)

# Q5: Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Q6: Average age
print("\nAverage Age:")
print(df['Age'].mean())

# Q7: Average watch hours per week
print("\nAverage Watch Hours Per Week:")
print(df['WatchHoursPerWeek'].mean())

# Q8: Average monthly spending
print("\nAverage Monthly Spending:")
print(df['MonthlySpend'].mean())

# Q9: Users in each subscription category
print("\nSubscription Type Counts:")
print(df['SubscriptionType'].value_counts())

# Q10: Percentage of renewed subscriptions
renewed_percent = (df['SubscriptionRenewed'] == 'Yes').mean() * 100
print("\nRenewal Percentage:")
print(f"{renewed_percent:.2f}%")


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Make a copy
data = df.copy()

# Q11: Encode categorical columns
le = LabelEncoder()

for col in data.select_dtypes(include='object').columns:
    data[col] = le.fit_transform(data[col])

# Q12: Features and target
X = data.drop(['SubscriptionRenewed', 'MonthlySpend'], axis=1)
y = data['SubscriptionRenewed']

# Q13: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Q14: Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Q15: Accuracy
y_pred_dt = dt.predict(X_test)
dt_accuracy = accuracy_score(y_test, y_pred_dt)

print("Decision Tree Accuracy:", dt_accuracy)

# Q16: Confusion Matrix
print("\nDecision Tree Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt))

# Q17: KNN (K=5)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Q18: KNN Accuracy
y_pred_knn = knn.predict(X_test)
knn_accuracy = accuracy_score(y_test, y_pred_knn)

print("\nKNN Accuracy:", knn_accuracy)

# Q19: Linear Regression for Monthly Spend
X_reg = data.drop('MonthlySpend', axis=1)
y_reg = data['MonthlySpend']

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

lr = LinearRegression()
lr.fit(Xr_train, yr_train)

# Q20: Predict Monthly Spending for a new user
new_user = [1, 25, 1, 2, 20, 3, 1, 5, 1]  # Example values
prediction = lr.predict([new_user])

print("\nPredicted Monthly Spend:", prediction[0])


# Business Reflection Questions

# 1. Which factors appear to influence subscription renewal the most?
# Factors such as Subscription Type, Watch Hours Per Week,
# Monthly Spending, Age, and Devices Used may have a strong
# influence on whether a user renews their subscription.

# 2. Why is subscription renewal a classification problem?
# Subscription renewal is a classification problem because the
# target variable has discrete categories: Yes or No.
# The model predicts which category a user belongs to.

# 3. Why is monthly spending a regression problem?
# Monthly spending is a regression problem because it is a
# continuous numerical value. The model predicts an amount
# rather than a category.

# 4. Which algorithm performed better for renewal prediction?
# Compare the accuracy scores of Decision Tree and KNN.
# The algorithm with the higher accuracy performed better.
# (Replace this comment with your actual result after running the models.)

# Example:
# Decision Tree Accuracy = 0.82
# KNN Accuracy = 0.78
# Therefore, Decision Tree performed better.

# 5. How could the platform use these predictions to improve customer retention?
# Netflix can identify users who are likely not to renew their
# subscriptions and target them with personalized recommendations,
# discounts, special offers, or engagement campaigns to improve retention.