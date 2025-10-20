# ✈️ Airfare Model Deployment

## 🧩 Project Overview
This project focuses on predicting **airfare yield** (`cur_yield`) using a variety of airline and route performance indicators.  
It demonstrates a **complete end-to-end machine learning workflow**, from data preprocessing and feature selection to model evaluation and deployment readiness.

---

## 📊 Dataset Description
The dataset includes performance metrics from multiple airline routes and time periods.  
Each observation represents a specific route and includes both historical and current fare and yield measures.

### Key Features Used:
- `cur_passengers` — Current passenger count  
- `ly_yield` — Last year’s yield value  
- `passenger_growth_rate` — Growth rate in passenger volume  
- `cur_fare` — Current average fare  
- `citymarketid` — Market or route identifier  
- `ly_distance` — Last year’s flight distance  
- `distance_efficiency` — Efficiency ratio of current vs. historical distance  
- `ly_fare` — Last year’s average fare  
- `yield_change` — Change in yield compared to previous year  
- `fare_per_passenger` — Fare distributed per passenger  
- `distance` — Current flight distance  
- `fare_difference` — Difference between current and last year’s fares  

---

## ⚙️ Methodology

### 1. Data Cleaning & Preprocessing
- Handled missing values and outliers  
- Scaled and normalized numerical columns  
- Encoded categorical variables such as `citymarketid`  
- Split data into **training (80%)** and **testing (20%)** sets  

### 2. Feature Selection
A **comprehensive hybrid approach** was applied combining:
- **Random Forest Importance** – identified top-performing features  
- **Correlation Filter** – removed redundant predictors  
- **Lasso Regression (L1 Regularization)** – kept variables with meaningful linear coefficients  

Final Selected Features:  
`['cur_passengers', 'ly_yield', 'passenger_growth_rate', 'cur_fare', 'citymarketid', 'ly_distance', 'distance_efficiency', 'ly_fare', 'yield_change', 'fare_per_passenger', 'distance', 'fare_difference']`

### 3. Model Training & Evaluation
Trained and compared four models:
- Linear Regression  
- Decision Tree Regressor  
- Random Forest Regressor  
- Gradient Boosting Regressor  

| Model | MSE | R² |
|--------|------|------|
| **Linear Regression** | **0.000013** | **1.000000** |
| Gradient Boosting | 0.010251 | 0.999676 |
| Random Forest | 0.014575 | 0.999539 |
| Decision Tree | 0.018031 | 0.999430 |

✅ **Best Model:** Linear Regression  
It achieved near-perfect accuracy with an R² of 1.0 and extremely low error values.

---

## 🧠 Model Files

### `best_linear_regression_model.pkl`
This is the **final deployed model**, trained using `LinearRegression()` from Scikit-learn and saved with `joblib`.  
It is stored in binary format, so GitHub will only show a **“View raw”** link — clicking it downloads the file.

To load and use it in Python:

```python
import joblib
model = joblib.load('best_linear_regression_model.pkl')
```

> ⚠️ Note: The `.pkl` file cannot be viewed directly in GitHub or text editors.  
> It must be loaded using Python to perform predictions.

---

## 📈 Results & Insights
- Linear Regression perfectly captured the relationship between fare and yield-related features.  
- The results indicate strong linear dependence between key pricing factors such as fare, passenger growth, and yield change.  
- Gradient Boosting and Random Forest models provided strong but slightly more complex alternatives.

---

## 🧰 Tools & Libraries
- **Python 3.x**  
- **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**  
- **Scikit-learn** (`LinearRegression`, `RandomForestRegressor`, `DecisionTreeRegressor`, `GradientBoostingRegressor`, `train_test_split`, `joblib`)  
- **Jupyter Notebook / VS Code**  
- **Git & GitHub** for version control  

---

## 📁 Repository Structure
```
Airfare_Model_Deployment/
│
├── README.md
├── mynewnotebook17.ipynb
├── best_linear_regression_model.pkl
├── best_random_forest_model.pkl
├── airfare_cleaned_data.csv
└── requirements.txt
```

---

## 👩‍💻 Author
**Martin**  
Data Science & Machine Learning Enthusiast  
📧 [Insert your email or LinkedIn profile link here]  

---

## 🚀 Future Work
- Integrate a **Streamlit** or **Flask** app for real-time airfare prediction  
- Extend the dataset to include **seasonal and macroeconomic variables**  
- Apply **model explainability (SHAP or LIME)** to interpret feature influence  
- Automate model retraining for dynamic fare trend updates  

---

### ⭐ Summary
This project demonstrates a **professional-grade regression pipeline**, emphasizing data preprocessing, hybrid feature selection, and reproducible model deployment.  
It highlights both the **technical workflow** and the **business insight** behind predicting airfare yield performance.
