# Titanic Survival Prediction

A Machine Learning web application that predicts whether a passenger would have survived the Titanic disaster using a **Support Vector Machine (SVM)**. The project includes feature engineering, preprocessing, hyperparameter tuning with **GridSearchCV**, and an interactive **Streamlit** interface.

---

## Live Demo

**Streamlit App:** *(Add your deployed app link here)*

---

## Features

- Predicts passenger survival in real time
- Interactive Streamlit web interface
- User-friendly sidebar for passenger information
- Displays prediction confidence
- Visualizes prediction using charts
- Shows trained model performance metrics
- Uses a saved machine learning pipeline

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Joblib

---

## Machine Learning Pipeline

### Data Preprocessing

- Missing value handling
- Feature engineering
- Categorical encoding
- Feature scaling

### Feature Engineering

- Passenger Title extraction
- Deck extraction from Cabin
- Family Size creation

### Model

- Support Vector Machine (SVC)
- GridSearchCV hyperparameter tuning
- Pipeline + ColumnTransformer

---

## Model Performance

| Metric | Value |
|--------|------:|
| Model | Support Vector Classifier (SVC) |
| Training Accuracy | **81.56%** |
| Cross Validation Score | **82.45%** |

### Best Hyperparameters

| Parameter | Value |
|-----------|-------|
| C | 1 |
| Kernel | RBF |
| Gamma | scale |

---

## Project Structure

```text
Titanic-Survival-Prediction/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── model/
│   ├── titanic_pipeline.pkl
│   └── metrics.json
│
├── src/
│   ├── app.py
│   ├── train.py
│   ├── predict.py
│   └── preprocessing.py
│
├── requirements.txt
└── README.md
```

---

## Run Locally

Clone the repository

```bash
git clone https://github.com/Panwithaging/Titanic-Survival-Prediction.git
```

Go to the project folder

```bash
cd Titanic-Survival-Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
streamlit run src/app.py
```

---

## Application Preview

Add screenshots of:

- Home page
- User input sidebar
- Prediction result
- Charts and confidence visualization

---

## Dataset

Kaggle Titanic: Machine Learning from Disaster dataset.

---

## Author

**Tushar Panging**

GitHub: https://github.com/Panwithaging
**Tushar Panging**

GitHub: https://github.com/Panwithaging
