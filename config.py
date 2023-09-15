from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

knn_params = {"n_neighbors": range(2, 50)}

cart_params = {'max_depth': range(1, 25),
               "min_samples_split": range(25, 40)}

rf_params = {"max_depth": [8, 15, 22],
             "max_features": [5, 7, "sqrt", "auto"],
             "min_samples_split": [20, 29, 39],
             "n_estimators": [100, 200]}

xgboost_params = {"learning_rate": [0.01, 0.05, 0.1],
                  "max_depth": range(1, 10),
                  "n_estimators": [100, 200],
                  "colsample_bytree": [0.3, 0.5, 1]}

lightgbm_params = {"learning_rate": [0.01, 0.05, 0.1],
                   "n_estimators": [250, 300, 350],
                   "colsample_bytree": [0.8, 1]}

classifiers = [('KNN', KNeighborsClassifier(), knn_params),
               ("CART", DecisionTreeClassifier(), cart_params),
               ("RF", RandomForestClassifier(), rf_params),
               ('XGBoost', XGBClassifier(eval_metric='logloss'), xgboost_params),
               ('LightGBM', LGBMClassifier(force_col_wise=True, verbose=-1), lightgbm_params)]

