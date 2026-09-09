from preprocessing import preprocess_data
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,train_test_split
import joblib
from sklearn.metrics import accuracy_score
import time
import json

train=pd.read_csv("data/train.csv")
train=preprocess_data(train)


x=train.drop(columns=["Survived","PassengerId"])
y=train["Survived"]

x_train,x_val,y_train,y_val=train_test_split(x,y,test_size=0.2,random_state=42, stratify=y)

preprocessor=ColumnTransformer(transformers=[("cat",OneHotEncoder(handle_unknown='ignore',sparse_output=False),["Sex","Embarked","Title","Deck"]),
                                             ("num",StandardScaler(),["Age","Fare","Family Size"])],remainder='passthrough')

clf=Pipeline(steps=[("preprocessor",preprocessor),
                    ("model",SVC(probability=True))])

params={
        "model__C": [0.1, 1, 10, 100],
        "model__kernel": ["linear", "rbf"],
        "model__gamma": ["scale", "auto"]
}

grid=GridSearchCV(estimator=clf,
                  param_grid=params,
                  cv=5,
                  scoring='accuracy',
                  n_jobs=-1)
start_train=time.perf_counter()
grid.fit(x_train,y_train)

pred=grid.predict(x_val)
score=accuracy_score(y_val,pred)

training_time=time.perf_counter()-start_train

joblib.dump(grid.best_estimator_,"model/titanic_pipeline.pkl")

metrics={
    "Model":"Support Vector Classifier",
    "Accuracy": score,
    "Training_time":training_time,
    "Best parameters":{
        "C": grid.best_params_["model__C"],
        "Kernel": grid.best_params_["model__kernel"],
        "Gamma": grid.best_params_["model__gamma"]
        },
    "cv score":grid.best_score_
}

with open("model/metrics.json","w") as f:
    json.dump(metrics,f,indent=4)

print("model saved successfully")