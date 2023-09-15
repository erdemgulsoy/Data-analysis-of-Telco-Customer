# author : Mustafa Erdem Gülsoy
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

def grab_col_names(dataframe, cat_th=5, car_th=20) :
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe : dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th : int, float
        numerik fakat kategorik olan değişkenlerdir.
    car_th : int, float
        kategorik fakat kardinal olan değişkenler için sınıf eşik değeri.

    Returns
    -------
        cat_cols : list
            Kategorik değişken listesi.
        num_cols : list
            Numerik değişkeen listesi.
        cat_but_car : list
            Kategorik görünümlü kardinal değişken listesi.

    Notes
    ------
        cat_cols + num_cols +cat_but_car = toplam değişken sayısı.
        num_but_cat, cat_cols'un içerisinde.

    """

    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].dtypes in ["int64", "float64", "uint8"] and dataframe[col].nunique() < cat_th]

    cat_but_car = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object"] and dataframe[col].nunique() > car_th]

    cat_cols += num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if col not in cat_cols and dataframe[col].dtypes in ["int64", "float64", "uint8"]]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car
def check_df (dataframe, head=5) :
    print("#################### Shape ##################")
    print(dataframe.shape)
    print("#################### Type ##################")
    print(dataframe.dtypes)
    print("#################### Head ##################")
    print(dataframe.head(head))
    print("#################### Tail ##################")
    print(dataframe.tail(head))
    print("#################### NA ##################")
    print(dataframe.isnull().sum())
    print("#################### Quantiles ##################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
def cat_summary2(dataframe, col_name, plot=False) :
    print(pd.DataFrame({col_name : dataframe[col_name].value_counts(),
                        "Ratio"  : 100 * dataframe[col_name].value_counts() / len(dataframe) }))
    print("###############################################")

    if plot :
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)
def num_summary2(dataframe, numerical_col, plot=False) :
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot :
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)
def outlier_thresholds(dataframe, col_name, q1=0.15, q3=0.85) :

    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)

    iqr = quartile3 - quartile1 # InterQuartileRange

    up = quartile3 + iqr * 1.5  # formüle göre üst sınır.
    down = quartile1 - iqr * 1.5  # formüke göre alt sınır.

    return up, down
def check_outlier(df, col) :
    up, down = outlier_thresholds(df, col)
    print(col, " : ", df[(df[col] < down) | (df[col] > up)].any(axis=None))
def missing_values_table(dataframe, na_name = False) :
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0 or (dataframe[col] == " ").sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_df, end="\n")

    if na_name :
        return na_columns
def missing_vs_target(dataframe, target, na_columns) :
    temp_df = dataframe.copy()

    for col in na_columns :
        temp_df[col + "_NA_FLAG"] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags :
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(), "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe
# Rare Encoding ;
def rare_analyser(dataframe, target, cat_cols) :
    for col in cat_cols :
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"Count": dataframe[col].value_counts(),
                            "Ratio": 100 * (dataframe[col].value_counts() / len(dataframe)),
                            "Target_Mean": dataframe.groupby(col)[target].mean()}), end="\n\n\n")
def rare_encoder(dataframe, rare_perc) :
    temp_df = dataframe.copy()
    # burada ilk önce kategorik ve içinde 0.01'in altında oran olanlar varsa bu KOLONLARI rare_columns olarak alıyoruz.
    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == "O"
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    # içinde 0.01'den az oran olan kategoriler olduğunu bildiğimiz kolonların içinde gezerek bu kategorileri seçiyoruz ve bunları rare_labels içine atıyoruz. unutmayalım ; rare_columsn = içinde 0.01'den az oranda kategori olduğunu bildiğimiz kolonlar, rare_labels = bu kolonlardaki 0.01 oranındaki kategoriler.
    for var in rare_columns :
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), "Rare", temp_df[var])


    return temp_df
def one_hot_encoder(dataframe, categorial_cols, drop_first=True) :
    dataframe = pd.get_dummies(dataframe, columns=categorial_cols, drop_first=drop_first)
    return dataframe


def telco_data_prep(dataframe) :
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)

    dataframe["TotalCharges"] = pd.to_numeric(dataframe["TotalCharges"], errors='coerce')
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)

    dataframe['Churn'] = dataframe['Churn'].map({'Yes': 1, 'No': 0})

    zero_cols = ["MonthlyCharges", "customerID", "TotalCharges"]
    dataframe[zero_cols] = dataframe[zero_cols].replace(0, np.nan)

    na_cols = missing_values_table(dataframe, True)

    dataframe["TotalCharges"].fillna(dataframe["MonthlyCharges"] * (dataframe["tenure"]+1), inplace=True)


    dataframe.loc[(dataframe["tenure"] < 10) & (dataframe["Contract"] == "Month-to-month"), "NEW_ten_Cont_NOM"] = "very_shortMonth"
    dataframe.loc[((dataframe["tenure"] >= 10) & (dataframe["tenure"] < 20)) & (dataframe["Contract"] == "Month-to-month"), "NEW_ten_Cont_NOM"] = "shortMonth"
    dataframe.loc[((dataframe["tenure"] >= 20) & (dataframe["tenure"] < 60)) & (dataframe["Contract"] == "Month-to-month"), "NEW_ten_Cont_NOM"] = "normalMonth"
    dataframe.loc[(dataframe["tenure"] >= 60) & (dataframe["Contract"] == "Month-to-month"), "NEW_ten_Cont_NOM"] = "longMonth"
    dataframe.loc[(dataframe["tenure"] < 10) & (dataframe["Contract"] == "One year"), "NEW_ten_Cont_NOM"] = "very_short1Year"
    dataframe.loc[((dataframe["tenure"] >= 10) & (dataframe["tenure"] < 20)) & (dataframe["Contract"] == "One year"), "NEW_ten_Cont_NOM"] = "short1Year"
    dataframe.loc[((dataframe["tenure"] >= 20) & (dataframe["tenure"] < 60)) & (dataframe["Contract"] == "One year"), "NEW_ten_Cont_NOM"] = "normal1Year"
    dataframe.loc[(dataframe["tenure"] >= 60) & (dataframe["Contract"] == "One year"), "NEW_ten_Cont_NOM"] = "long1Year"
    dataframe.loc[(dataframe["tenure"] < 10) & (dataframe["Contract"] == "Two year"), "NEW_ten_Cont_NOM"] = "very_short2Year"
    dataframe.loc[((dataframe["tenure"] >= 10) & (dataframe["tenure"] < 20)) & (dataframe["Contract"] == "Two year"), "NEW_ten_Cont_NOM"] = "short2Year"
    dataframe.loc[((dataframe["tenure"] >= 20) & (dataframe["tenure"] < 60)) & (dataframe["Contract"] == "Two year"), "NEW_ten_Cont_NOM"] = "normal2Year"
    dataframe.loc[(dataframe["tenure"] >= 60) & (dataframe["Contract"] == "Two year"), "NEW_ten_Cont_NOM"] = "long2Year"

    dataframe.loc[(dataframe["MonthlyCharges"] < 30) & (dataframe["PaymentMethod"] == "Electronic check"), "NEW_Char_Pay_NOM"] = "very_cheapElectro"
    dataframe.loc[((dataframe["MonthlyCharges"] >= 30) & (dataframe["MonthlyCharges"] < 60)) & (dataframe["PaymentMethod"] == "Electronic check"), "NEW_Char_Pay_NOM"] = "cheapElectro"
    dataframe.loc[((dataframe["MonthlyCharges"] >= 60) & (dataframe["MonthlyCharges"] < 90)) & (dataframe["PaymentMethod"] == "Electronic check"), "NEW_Char_Pay_NOM"] = "normalElectro"
    dataframe.loc[(dataframe["MonthlyCharges"] >= 90) & (dataframe["PaymentMethod"] == "Electronic check"), "NEW_Char_Pay_NOM"] = "expensiveElectro"
    dataframe.loc[(dataframe["MonthlyCharges"] < 30) & (dataframe["PaymentMethod"] == "Mailed check"), "NEW_Char_Pay_NOM"] = "very_cheapMail"
    dataframe.loc[((dataframe["MonthlyCharges"] >= 30) & (dataframe["MonthlyCharges"] < 60)) & (dataframe["PaymentMethod"] == "Mailed check"), "NEW_Char_Pay_NOM"] = "cheapMail"
    dataframe.loc[((dataframe["MonthlyCharges"] >= 60) & (dataframe["MonthlyCharges"] < 90)) & (dataframe["PaymentMethod"] == "Mailed check"), "NEW_Char_Pay_NOM"] = "normalMail"
    dataframe.loc[(dataframe["MonthlyCharges"] >= 90) & (dataframe["PaymentMethod"] == "Mailed check"), "NEW_Char_Pay_NOM"] = "expensiveMail"
    dataframe.loc[(dataframe["MonthlyCharges"] < 30) & (dataframe["PaymentMethod"] == "Bank transfer (automatic)"), "NEW_Char_Pay_NOM"] = "very_cheapBank"
    dataframe.loc[((dataframe["MonthlyCharges"] >= 30) & (dataframe["MonthlyCharges"] < 60)) & (dataframe["PaymentMethod"] == "Bank transfer (automatic)"), "NEW_Char_Pay_NOM"] = "cheapBank"
    dataframe.loc[((dataframe["MonthlyCharges"] >= 60) & (dataframe["MonthlyCharges"] < 90)) & (dataframe["PaymentMethod"] == "Bank transfer (automatic)"), "NEW_Char_Pay_NOM"] = "normalBank"
    dataframe.loc[(dataframe["MonthlyCharges"] >= 90) & (dataframe["PaymentMethod"] == "Bank transfer (automatic)"), "NEW_Char_Pay_NOM"] = "expensiveBank"
    dataframe.loc[(dataframe["MonthlyCharges"] < 30) & (dataframe["PaymentMethod"] == "Credit card (automatic)"), "NEW_Char_Pay_NOM"] = "very_cheapCredit"
    dataframe.loc[((dataframe["MonthlyCharges"] >= 30) & (dataframe["MonthlyCharges"] < 60)) & (dataframe["PaymentMethod"] == "Credit card (automatic)"), "NEW_Char_Pay_NOM"] = "cheapCredit"
    dataframe.loc[((dataframe["MonthlyCharges"] >= 60) & (dataframe["MonthlyCharges"] < 90)) & (dataframe["PaymentMethod"] == "Credit card (automatic)"), "NEW_Char_Pay_NOM"] = "normalCredit"
    dataframe.loc[(dataframe["MonthlyCharges"] >= 90) & (dataframe["PaymentMethod"] == "Credit card (automatic)"), "NEW_Char_Pay_NOM"] = "expensiveCredit"

    dataframe['NEW_TotalServices'] = (dataframe[['PhoneService', 'InternetService', 'OnlineSecurity','OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']] == 'Yes').sum(axis=1)

    dataframe["NEW_AVG_Charges"] = dataframe["TotalCharges"] / (dataframe["tenure"] + 1)

    dataframe["NEW_Increase"] = dataframe["NEW_AVG_Charges"] / dataframe["MonthlyCharges"]

    dataframe["NEW_AVG_Service_Fee"] = dataframe["MonthlyCharges"] / (dataframe['NEW_TotalServices'] + 1)


    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)

    binary_cols = [col for col in cat_cols if dataframe[col].dtype not in ["int64", "float64"] and dataframe[col].nunique() == 2]
    for col in binary_cols:
        dataframe = label_encoder(dataframe, col)

    dataframe = rare_encoder(dataframe, 0.01)

    ohe_cols = [col for col in dataframe.columns if dataframe[col].nunique() > 2 and dataframe[col].nunique() <= 20]

    dataframe = one_hot_encoder(dataframe, ohe_cols, True)

    all_possible_values = ["very_shortMonth", "shortMonth", "normalMonth", "longMonth",
                           "very_short1Year", "short1Year", "normal1Year", "long1Year",
                           "very_short2Year", "short2Year", "normal2Year", "long2Year"]

    for val in all_possible_values:
        if f'NEW_ten_Cont_NOM_{val}' not in dataframe.columns:
            dataframe[f'NEW_ten_Cont_NOM_{val}'] = 0


    all_possible_values = ["very_cheapElectro", "cheapElectro", "normalElectro", "expensiveElectro",
                           "very_cheapMail", "cheapMail", "normalMail", "expensiveMail",
                           "very_cheapBank", "cheapBank", "normalBank", "expensiveBank",
                           "very_cheapCredit", "cheapCredit", "normalCredit", "expensiveCredit"]

    for val in all_possible_values:
        if f'NEW_Char_Pay_NOM{val}' not in dataframe.columns:
            dataframe[f'NEW_Char_Pay_NOM{val}'] = 0


    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)

    scaler = StandardScaler()
    dataframe[num_cols] = scaler.fit_transform(dataframe[num_cols])

    y = dataframe["Churn"]
    X = dataframe.drop(["Churn", "customerID"], axis=1)



    return X, y

def base_models(X, y, scoring="roc_auc"):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier(force_col_wise=True, verbose=-1)),
                   ]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")

from config import classifiers
def hyperparameter_optimization(X, y, cv=3, scoring="roc_auc"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=-1).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models

def voting_classifier(best_models, X, y):
    print("Voting Classifier...")

    voting_clf = VotingClassifier(estimators=[('KNN', best_models["KNN"]),
                                              ('RF', best_models["RF"]),
                                              ('LightGBM', best_models["LightGBM"])],
                                  voting='soft').fit(X, y)

    cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc"])
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1Score: {cv_results['test_f1'].mean()}")
    print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
    return voting_clf


