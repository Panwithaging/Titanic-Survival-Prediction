import pandas as pd

import numpy as np

def preprocess_data(data):

    data["Age"]=data["Age"].fillna(data["Age"].median())

    data["Family Size"]=data["SibSp"]+data["Parch"]+1

    data["IsAlone"]=(data["Family Size"]==1).astype(int)
    if "Name" in data.columns:
        data["Title"]=data["Name"].str.extract(r",s\*([^.]+)",expand=False)

    rare_titles={"Lady", "Countess", "Capt", "Col", "Don",
    "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona","the Countess"}
    data["Title"]=data["Title"].replace({"Mlle":"Miss",
                                         "Mme":"Mrs",
                                         "Ms":"Miss"})

    data["Title"]=data["Title"].replace(rare_titles,"rare")

    data["Fare"]=data["Fare"].fillna(data["Fare"].median())

    data["Fare"]=np.log1p(data["Fare"])

    data["Deck"]=data["Cabin"].fillna("No Info")

    data["Deck"]=data["Deck"].apply(lambda x: x[0] if x != "No Info"
                                    else "No Info")

    data["Embarked"]=data["Embarked"].fillna(data["Embarked"].mode()[0])

    data.drop(columns=["SibSp","Parch","Cabin"],inplace=True)

    if "Name" in data.columns and "Ticket" in data.columns:
        data.drop(columns=["Name","Ticket"],inplace=True)

    return data