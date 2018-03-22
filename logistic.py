import numpy as np
from statistics import mean
import random
import pandas as pd
import statsmodels.formula.api as sm
from patsy import dmatrices

def get_data():
    df = pd.read_csv("https://raw.githubusercontent.com/BigDataGal/Python-for-Data-Science/master/titanic-train.csv")
    return df

def test_mode():
    df = get_data()
    data = df.drop(columns = ['PassengerId','Ticket','Name','Cabin'])
    y, X = dmatrices("Survived ~ Pclass+Sex+Age+SibSp+Parch+Fare+Embarked", df, return_type = 'dataframe')
    
    #logit = sm.Logit(data["Survived"],data[])
    logit = sm.Logit(y,X)
    result = logit.fit()
    print result.summary2()
test_mode()
