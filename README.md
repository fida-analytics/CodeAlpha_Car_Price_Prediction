# Car Price Prediction using Machine Learning

## CodeAlpha Data Science Internship – Task 3

### Project Overview
This project focuses on predicting the selling price of used cars using Machine Learning techniques.

The model learns from historical car data and predicts the estimated selling price based on different features such as manufacturing year, present price, kilometers driven, fuel type, seller type, transmission, and previous owners.

### Dataset
**Dataset Name:** `car data.csv`

The dataset contains the following features:

- Car_Name
- Year
- Selling_Price
- Present_Price
- Driven_kms
- Fuel_Type
- Selling_type
- Transmission
- Owner

### Technologies & Libraries Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

### Project Workflow
1. Loading the dataset
2. Data cleaning and preprocessing
3. Exploratory Data Analysis (EDA)
4. Visualizing car selling prices
5. Analyzing prices based on fuel type
6. Preparing features and target variable
7. Encoding categorical variables
8. Training a Random Forest Regression model
9. Evaluating model performance
10. Predicting car selling prices

### Machine Learning Algorithm
**Random Forest Regressor**

The Random Forest Regression algorithm is used to predict car selling prices based on the available dataset features.

### Model Evaluation
The model performance is evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

### Visualizations
The project includes:
- Car Selling Price Distribution
- Selling Price by Fuel Type
- Actual vs Predicted Car Prices

### How to Run the Project

**1. Install Required Libraries**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

**2. Run the Python File**

```bash
python car_price_prediction.py
```

### Project Files
- `car_price_prediction.py` – Python source code
- `car data.csv` – Dataset
- `README.md` – Project documentation

### Learning Outcomes
- Understanding Machine Learning regression
- Data cleaning and preprocessing
- Exploratory Data Analysis
- Data visualization
- Model training and evaluation
- Making predictions using Random Forest Regression

### Internship
**CodeAlpha Data Science Internship**

Task 3: Car Price Prediction with Machine Learning