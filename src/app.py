import streamlit as  st
import pathlib as path
import json
import pandas as pd
from predict import predict
import matplotlib.pyplot as plt

BASE_DIR=path.Path(__file__).resolve().parent.parent
MODEL_PATH=BASE_DIR/"model"/"metrics.json"
with open(MODEL_PATH,"r") as f:
    metrics=json.load(f)   

st.set_page_config(page_title="Titanic Survival Prediction",page_icon="🚢",layout="wide") 
st.header("Titanic Survival Prediction") 
st.write("This application predicts Titanic passenger survival using a **Support Vector Machine (SVM)**. The model was trained on the Kaggle Titanic dataset with feature engineering, preprocessing, and hyperparameter tuning using **GridSearchCV** to improve prediction performance.") 
params = metrics["Best parameters"]

col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

with col1:
    st.metric("Model", "SVC")

with col2:
    st.metric("Accuracy", f"{metrics['Accuracy']*100:.2f}%")

with col3:
    st.metric("Training Time", f"{metrics['Training_time']:.2f} sec")

with col4:
    st.metric("CV Score", f"{metrics['cv score']*100:.2f}%")

with col5:
    st.metric("C", params["C"])

with col6:
    st.metric("Kernel", params["Kernel"])

with col7:
    st.metric("Gamma", params["Gamma"])

st.sidebar.title("User Input Parameters")
st.sidebar.write("Please provide the following information to predict the survival of a passenger on the Titanic.")
title=st.sidebar.selectbox("Enter your Title",["Mr","Mrs","Miss","Master","rare"],index=None,placeholder="Select Title")
st.sidebar.text_input("Enter your name:")
sex=st.sidebar.selectbox("Select your gender",["male","female"],index=None,placeholder="Select your gender")
age=st.sidebar.slider("Age",min_value=0,max_value=100,value=30)
sibsp = st.sidebar.number_input("Siblings/Spouses Aboard",min_value=0,max_value=8,value=0,help="Enter the total number of siblings and/or spouse traveling with the passenger.")
parch=st.sidebar.number_input("Parents/Children Aboard",min_value=0,max_value=9,value=0,help="Enter total number of parents and/or children travelling with you ")
fare=st.sidebar.number_input("Enter Fare(in usd) of ticket:",min_value=0.0,value=0.0,step=1.0)
pclass=st.sidebar.selectbox("Select your passenger  class",[1,2,3],index=None,placeholder="Pls select your passenger class")
deck=st.sidebar.selectbox("Select your Deck",["A","B","C","E","F"],index=None,placeholder="Select your deck")
deck_no=st.sidebar.number_input("Enter your Deck number:",min_value=1,value=1,step=1)
embarked=st.sidebar.selectbox("Embarked",["S","C","Q"],index=None,placeholder="Select Emabarked")

press=st.sidebar.button("Predict")
valid=True
if press:
    if title is None:
        st.error("Pls select your title")
        valid=False
    if sex is None:
        st.error("Pls select your gender")
        valid=False
    if deck is None:
        st.error("Pls select your deck")
        valid=False
    if embarked is None:
        st.error("Pls select your embarked")
        valid=False
    if deck is None:
        st.error("Pls select your deck")
        valid=False    
    if pclass is None:
        st.error("Pls select your passenger class")
        valid=False    

    if valid:
        cabin=deck+str(deck_no)
        user_data=pd.DataFrame({"Pclass":[pclass],
            "Title":[title],
            "Sex":[sex],
            "Age":[age],
            "SibSp":[sibsp],
            "Parch":[parch],
            "Fare":[fare],
            "Cabin":[cabin],
            "Embarked":[embarked]})   

        prediction,prediction_time,prob=predict(user_data)

        confidence=prob[0].max()*100
        

        #pre="Survived" if prediction[0]==1 else "Not survived"

        time=prediction_time
          
graph1, graph2, graph3 = st.columns(3)

with graph1:
    graph1_placeholder = st.empty()

with graph2:
    graph2_placeholder = st.empty()

with graph3:
    graph3_placeholder = st.empty()

if press and valid:

    fig1, ax1 = plt.subplots(figsize=(4,3))
    labels = ["Not Survived", "Survived"]
    values = [prob[0][0]*100, prob[0][1]*100]
    ax1.bar(labels, values)
    ax1.set_xlim(0,100)
    ax1.set_xlabel("Probability (%)")
    ax1.set_title("Prediction Probability")
    graph1_placeholder.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(4,3))
    st.dataframe(user_data,use_container_width=True)
    features = ["Age","Family Size"]
    values = [age, sibsp + parch + 1]
    ax2.bar(features, values)
    ax2.set_title("Passenger Profile")
    graph2_placeholder.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=(4,3))
    sizes = [prob[0][0], prob[0][1]]
    labels = ["Not Survived", "Survived"]
    ax3.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90)
    ax3.set_title("Prediction Confidence")
    graph3_placeholder.pyplot(fig3)

    out1,out2,out3=st.columns(3)
    out1.success(f"Survived" if prediction[0]==1 else "Not survived")
        
    with out2:
        st.metric ("Confidence",f"{confidence:.2f}%") 
        
    with out3:
        st.metric("prediction time",f"{time:.4f}sec")

else:

    graph1_placeholder.info("Prediction graph will appear here.")
    graph2_placeholder.info("Passenger graph will appear here.")
    graph3_placeholder.info("Dataset graph will appear here.")  