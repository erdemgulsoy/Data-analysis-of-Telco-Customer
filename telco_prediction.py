import joblib
import pandas as pd
from telco_pipeline import telco_data_prep


df = pd.read_csv("datasets/telco_churn2.csv")

new_model = joblib.load("telco_voting_clf.pkl")
train_col = joblib.load('train_col.pkl')

X, y = telco_data_prep(df)
X = X[train_col]

new_model.predict(X) # %75
