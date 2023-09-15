import joblib
import pandas as pd

from telco_research import telco_data_prep, hyperparameter_optimization, voting_classifier

def main () :
    df = pd.read_csv("datasets/Telco-Customer-Churn.csv")
    X, y = telco_data_prep(df)
    train_columns = X.columns.tolist()
    joblib.dump(train_columns, 'train_col.pkl')
    best_models = hyperparameter_optimization(X, y)
    voting_clf = voting_classifier(best_models, X, y)
    joblib.dump(voting_clf, "telco_voting_clf.pkl")
    return voting_clf


if __name__ == "__main__":
    main()

