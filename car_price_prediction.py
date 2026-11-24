
# CodeAlpha Internship - Task 3
# Car Price Prediction using Machine Learning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load Dataset
df = pd.read_csv("car data.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# 2. Data Cleaning
df = df.dropna()
df = df.drop_duplicates()

# 3. Exploratory Data Analysis
print("\nDataset Description:")
print(df.describe())

# Selling Price Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Selling_Price"], kde=True)
plt.title("Car Selling Price Distribution")
plt.xlabel("Selling Price (Lakhs)")
plt.ylabel("Frequency")
plt.show()

# Fuel Type Analysis
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="Fuel_Type", y="Selling_Price")
plt.title("Selling Price by Fuel Type")
plt.show()

# 4. Prepare Data
X = df.drop(columns=["Selling_Price", "Car_Name"])
y = df["Selling_Price"]

categorical_features = ["Fuel_Type", "Selling_type", "Transmission"]
numerical_features = ["Year", "Present_Price", "Driven_kms", "Owner"]

# 5. Preprocessing and Model
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"),
         categorical_features)
    ],
    remainder="passthrough"
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ))
])

# 6. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Train Model
model.fit(X_train, y_train)

# 8. Predictions
predictions = model.predict(X_test)

# 9. Model Evaluation
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("\nModel Evaluation:")
print("Mean Absolute Error:", round(mae, 2))
print("Root Mean Squared Error:", round(rmse, 2))
print("R2 Score:", round(r2, 2))

# 10. Actual vs Predicted Graph
plt.figure(figsize=(8, 5))
plt.scatter(y_test, predictions)
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Car Prices")
plt.show()

# 11. Example Prediction
sample_car = pd.DataFrame([{
    "Year": 2015,
    "Present_Price": 5.5,
    "Driven_kms": 30000,
    "Fuel_Type": "Petrol",
    "Selling_type": "Dealer",
    "Transmission": "Manual",
    "Owner": 0
}])

predicted_price = model.predict(sample_car)

print("\nExample Car Price Prediction:")
print("Predicted Selling Price:", round(predicted_price[0], 2), "Lakhs")
