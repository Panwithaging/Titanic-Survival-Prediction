import joblib
import pathlib as path

from preprocessing import preprocess_data

import time
BASE_DIR=path.Path(__file__).resolve().parent.parent
MODEL_PATH=BASE_DIR/"model"/"titanic_pipeline.pkl"

model=joblib.load(MODEL_PATH)

def predict(data):
    data=preprocess_data(data)
    start=time.perf_counter()
    prediction=model.predict(data)
    prob=model.predict_proba(data)
    prediction_time=time.perf_counter()-start

    return prediction,prediction_time,prob
