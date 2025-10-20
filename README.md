# ✈️ Airfare Model Deployment  

## 🎯 Project Overview  
This project applies a full **machine learning workflow** to analyze and predict airfare yield performance using real-world data.  
It covers data cleaning, exploratory data analysis (EDA), feature engineering, model training, evaluation, and deployment readiness — following a structured and professional workflow.

---

## 🧩 Project Objectives  
1. Clean and preprocess the dataset to ensure data quality and consistency.  
2. Explore data distributions, correlations, and outliers through structured EDA.  
3. Engineer relevant numerical and categorical features to enhance model performance.  
4. Train and compare multiple regression algorithms.  
5. Validate and export the best-performing model for deployment.

---

## 🧠 Workflow Summary  
**Step 1 – Data Cleaning:**  
- Removed duplicates, handled missing values, and standardized column types.  

**Step 2 – Exploratory Data Analysis (EDA):**  
- Conducted univariate and bivariate analyses using histograms, scatter plots, and correlation heatmaps.  

**Step 3 – Feature Engineering:**  
- Created derived metrics such as `fare_per_passenger`, `passenger_growth_rate`, and `distance_efficiency`.  

**Step 4 – Model Training:**  
- Trained and compared four regression models:  
  - Linear Regression  
  - Decision Tree Regressor  
  - Random Forest Regressor  
  - Gradient Boosting Regressor  

**Step 5 – Model Validation:**  
- The **Linear Regression** model achieved the best results:  
  - **R² = 1.0000**  
  - **MSE = 0.000013**  
- The model was saved as `best_linear_regression_model.pkl` using `joblib`.

**Step 6 – Deployment Preparation:**  
- Exported model and notebook files into a single deployment folder (`Airfare_Model_Deployment`).  
- Uploaded final project files to GitHub for version control and portfolio presentation.

---

## 📊 Model Performance Comparison  

| Model | MSE | R² |
|:--|--:|--:|
| Linear Regression | 0.000013 | 1.0000 |
| Gradient Boosting | 0.010251 | 0.999676 |
| Random Forest | 0.014575 | 0.999539 |
| Decision Tree | 0.018031 | 0.999430 |

---

## 🧾 Files in This Repository  
| File | Description |
|:--|:--|
| **mynewnotebook17.ipynb** | Complete Jupyter Notebook workflow containing all steps from data cleaning to model validation. |
| **best_linear_regression_model.pkl** | Trained Linear Regression model serialized using Joblib. |
| *(optional)* `Clean_Airfare_Dataset.csv` | Cleaned dataset used for training and testing. |

---

## 💡 Key Learnings  
- End-to-end ML project execution using Scikit-learn.  
- Importance of structured workflow (EDA → Feature Engineering → Model Evaluation).  
- Model validation and reproducibility through saving and reloading techniques.  
- Version control and deployment using Git & GitHub.

---

## 👩‍💻 Author  
**Benedette Ogochukwu Chukwu**  
📍 Data Science Enthusiast | Machine Learning Developer  
🔗 [GitHub Profile](https://github.com/martystats)

---

## 🧭 Next Steps  
- Extend project to include real-time Streamlit deployment for yield prediction.  
- Perform hyperparameter tuning (GridSearchCV) for further model optimization.  
- Integrate dashboard visuals to support analytical insights.

---

### 🏁 Project Status: ✅ Completed  
This project demonstrates a full machine learning pipeline from raw data to deployable model — built, validated, and published professionally.
