# Titanic Survival Prediction using Machine Learning

Predicting passenger survival on the Titanic using supervised machine learning techniques. This project demonstrates an end-to-end machine learning workflow, including data cleaning, exploratory data analysis (EDA), feature engineering, preprocessing, model comparison, hyperparameter tuning, and Kaggle submission.

---

## Project Overview

The goal of this project is to build a classification model that predicts whether a passenger survived the Titanic disaster based on passenger information such as age, gender, ticket class, fare, and family relationships.

The project follows a complete machine learning pipeline from raw data to final prediction.

---

## Dataset

**Source:** Kaggle Titanic - Machine Learning from Disaster

https://www.kaggle.com/competitions/titanic/data

### Features

- PassengerId
- Pclass
- Name
- Sex
- Age
- SibSp
- Parch
- Ticket
- Fare
- Cabin
- Embarked

### Target Variable

- **Survived**
  - `0` → Did Not Survive
  - `1` → Survived

---

## Project Workflow

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Data Preprocessing
- Model Training
- Model Evaluation
- Hyperparameter Tuning (GridSearchCV)
- Final Prediction
- Kaggle Submission

---

## Feature Engineering

The following features were created to improve model performance:

- Family Size
- IsAlone
- Passenger Title
- Deck Information
- Log Transformation of Fare

---

## Data Preprocessing

- Missing Value Imputation
- One-Hot Encoding
- Standard Scaling
- ColumnTransformer
- Pipeline

---

## Models Evaluated

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest

---

## Evaluation Metrics

The models were compared using:

- Cross Validation Score
- Accuracy Score
- Confusion Matrix
- Classification Report

---

## Hyperparameter Tuning

GridSearchCV was used to optimize the following models:

- Support Vector Machine (SVM)
- Logistic Regression
- K-Nearest Neighbors (KNN)

---

## Final Model

**Support Vector Machine (SVM)**

### Best Parameters

```python
kernel = "rbf"
C = 1
gamma = "scale"
```

---

## Kaggle Results

| Metric | Value |
|--------|-------|
| Public Leaderboard Score | **0.78229** |
| Final Model | Support Vector Machine (SVM) |

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

## Repository Structure

```
Titanic-Survival-Prediction/
│
├── Titanic_Survival_Prediction.ipynb
├── train.csv
├── test.csv
├── submission.csv
├── README.md
└── requirements.txt
```

---

## How to Run

1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Titanic-Survival-Prediction.git
```

2. Install the required libraries

```bash
pip install -r requirements.txt
```

3. Open the notebook

```bash
jupyter notebook Titanic_Survival_Prediction.ipynb
```

---

## Future Improvements

- Experiment with XGBoost, LightGBM, and CatBoost
- Perform advanced feature engineering
- Try ensemble learning methods
- Explore additional hyperparameter optimization techniques

---

## Author

**Tushar Panging**

If you found this project useful, feel free to ⭐ the repository.