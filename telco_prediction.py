import joblib
import pandas as pd

df = pd.read_csv("datasets/telco_churn2.csv")

from telco_pipeline import telco_data_prep

new_model = joblib.load("telco_voting_clf.pkl")
train_col = joblib.load('train_col.pkl')


X, y = telco_data_prep(df)
X = X[train_col]


random_user = X.sample(1, random_state=50)
new_model.predict(random_user)
new_model.predict(X)

