# author : Mustafa Erdem Gülsoy
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_curve



pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = pd.read_csv("datasets/Telco-Customer-Churn.csv")


#######################################################
# Genel Resim ;
#######################################################


# DataFrame için ilk olarak kategorik ve numerik değişkenleri ayırıyoruz ;
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

    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]

    num_but_cat = [col for col in df.columns if df[col].dtypes in ["int64", "float64", "uint8"] and df[col].nunique() < cat_th]

    cat_but_car = [col for col in df.columns if str(df[col].dtypes) in ["category", "object"] and df[col].nunique() > car_th]

    cat_cols += num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if col not in cat_cols and df[col].dtypes in ["int64", "float64", "uint8"]]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# TotalCharges sayısal bir değişken olmalı
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
cat_cols, num_cols, cat_but_car = grab_col_names(df)


# DataFrame için ön bilgi alma işlemi ;
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
check_df(df)

def cat_summary2(dataframe, col_name, plot=False) :
    print(pd.DataFrame({col_name : dataframe[col_name].value_counts(),
                        "Ratio"  : 100 * dataframe[col_name].value_counts() / len(dataframe) }))
    print("###############################################")

    if plot :
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)
for col in cat_cols :
    cat_summary2(df, col, True)

def num_summary2(dataframe, numerical_col, plot=False) :
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot :
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)
for col in num_cols :
    num_summary2(df, col, True)


# Kategorik değişkenlere göre hedef değişkenin ortalaması ;
# Bunun için ilk olarak yes ve no kategorilerini 1 ve 0 olarak değiştirmeliyiz ;
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

for col in cat_cols :
    print(df.groupby(col).agg({"Churn": "mean"}))

# Hedef değişkene göre numerik değişkenlerin ortalaması ;
for col in num_cols :
    print(df.groupby("Churn").agg({col: "mean"}))



#################################################
# Outliers Analizi ;
#################################################


# Aykırı değer kontrolü için up ve down oranlarını belirliyoruz.
def outlier_thresholds(dataframe, col_name, q1=0.15, q3=0.85) :

    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)

    iqr = quartile3 - quartile1 # InterQuartileRange

    up = quartile3 + iqr * 1.5  # formüle göre üst sınır.
    down = quartile1 - iqr * 1.5  # formüke göre alt sınır.

    return up, down
# Değişkenlerin aykırı değerleri varsa True, yoksa False veriyor. (check outlier)
def check_outlier(df, col) :
    up, down = outlier_thresholds(df, col)
    print(col, " : ", df[(df[col] < down) | (df[col] > up)].any(axis=None))

for col in num_cols:
    check_outlier(df, col)

# Aykırı Değer yok.



#################################################
# Missing Value Analizi ;
#################################################


# Eksik değer var mı ?
def missing_values_table(dataframe, na_name = False) :
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0 or (dataframe[col] == " ").sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_df, end="\n")

    if na_name :
        return na_columns
missing_values_table(df)

# Veri setinde eksik gözlem bulunmamakta ama bazı değişkenlerde 0 değeri içeren gözlem birimleri eksik değeri ifade ediyor olabilir.
zero_cols = ["MonthlyCharges", "customerID", "TotalCharges"]
df[zero_cols] = df[zero_cols].replace(0, np.nan)

na_cols = missing_values_table(df, True)
# TotalCharges değişkeninde 11 tane eksik değer var.


# Eksik Değerlerin Bağımlı Değişken Analizi;
def missing_vs_target(dataframe, target, na_columns) :
    temp_df = dataframe.copy()

    for col in na_columns :
        temp_df[col + "_NA_FLAG"] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags :
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(), "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")
missing_vs_target(df, "Churn", na_cols)
# çok da göze çarpan bir değişken yok.


# Eksik değerler sadece toplam ödeme miktarı(TotalCharges) olduğu için. Müşteriden aylık tahsis edilen miktar(MonthlyCharges) ile kaldığı ay sayısını(tenure) çarparak yerine yerleştirebiliriz.
df["TotalCharges"].fillna(df["MonthlyCharges"] * (df["tenure"]+1), inplace=True)


# Tekrar eksik değer var mı diye bakıyoruz ;
missing_values_table(df) # Yok.

# Tüm eksik Değerleri en yakın 5 komşusuna göre doldurmuş bulunuyoruz.


#################################################
# KORELASYON ANALİZİ
#################################################


# Korelasyon, olasılık kuramı ve istatistikte iki rassal değişken arasındaki doğrusal ilişkinin yönünü ve gücünü belirtir

df.corr()

# Korelasyon Matrisi
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show(block=True)

# Eksik Değerlerin Korelasyon Analizi ;
msno.bar(df)
plt.show(block=True)

msno.matrix(df)
plt.show(block=True)

msno.heatmap(df)
plt.show(block=True)


#################################################
# DEĞİŞKEN ÜRETME
#################################################

# şirkette kaldığı ay - sözleşme süresi
df.loc[(df["tenure"] < 10) & (df["Contract"] == "Month-to-month") , "NEW_ten_Cont_NOM"] = "very_shortMonth"
df.loc[((df["tenure"] >= 10) & (df["tenure"] < 20)) & (df["Contract"] == "Month-to-month"), "NEW_ten_Cont_NOM"] = "shortMonth"
df.loc[((df["tenure"] >= 20) & (df["tenure"] < 60)) & (df["Contract"] == "Month-to-month"), "NEW_ten_Cont_NOM"] = "normalMonth"
df.loc[(df["tenure"] >= 60) & (df["Contract"] == "Month-to-month") , "NEW_ten_Cont_NOM"] = "longMonth"
df.loc[(df["tenure"] < 10) & (df["Contract"] == "One year") , "NEW_ten_Cont_NOM"] = "very_short1Year"
df.loc[((df["tenure"] >= 10) & (df["tenure"] < 20)) & (df["Contract"] == "One year"), "NEW_ten_Cont_NOM"] = "short1Year"
df.loc[((df["tenure"] >= 20) & (df["tenure"] < 60)) & (df["Contract"] == "One year"), "NEW_ten_Cont_NOM"] = "normal1Year"
df.loc[(df["tenure"] >= 60) & (df["Contract"] == "One year") , "NEW_ten_Cont_NOM"] = "long1Year"
df.loc[(df["tenure"] < 10) & (df["Contract"] == "Two year") , "NEW_ten_Cont_NOM"] = "very_short2Year"
df.loc[((df["tenure"] >= 10) & (df["tenure"] < 20)) & (df["Contract"] == "Two year"), "NEW_ten_Cont_NOM"] = "short2Year"
df.loc[((df["tenure"] >= 20) & (df["tenure"] < 60)) & (df["Contract"] == "Two year"), "NEW_ten_Cont_NOM"] = "normal2Year"
df.loc[(df["tenure"] >= 60) & (df["Contract"] == "Two year") , "NEW_ten_Cont_NOM"] = "long2Year"


# ödeme miktarı - ödeme şekli
df.loc[(df["MonthlyCharges"] < 30) & (df["PaymentMethod"] == "Electronic check") , "NEW_Char_Pay_NOM"] = "very_cheapElectro"
df.loc[((df["MonthlyCharges"] >= 30) & (df["MonthlyCharges"] < 60)) & (df["PaymentMethod"] == "Electronic check"), "NEW_Char_Pay_NOM"] = "cheapElectro"
df.loc[((df["MonthlyCharges"] >= 60) & (df["MonthlyCharges"] < 90)) & (df["PaymentMethod"] == "Electronic check"), "NEW_Char_Pay_NOM"] = "normalElectro"
df.loc[(df["MonthlyCharges"] >= 90) & (df["PaymentMethod"] == "Electronic check") , "NEW_Char_Pay_NOM"] = "expensiveElectro"
df.loc[(df["MonthlyCharges"] < 30) & (df["PaymentMethod"] == "Mailed check") , "NEW_Char_Pay_NOM"] = "very_cheapMail"
df.loc[((df["MonthlyCharges"] >= 30) & (df["MonthlyCharges"] < 60)) & (df["PaymentMethod"] == "Mailed check"), "NEW_Char_Pay_NOM"] = "cheapMail"
df.loc[((df["MonthlyCharges"] >= 60) & (df["MonthlyCharges"] < 90)) & (df["PaymentMethod"] == "Mailed check"), "NEW_Char_Pay_NOM"] = "normalMail"
df.loc[(df["MonthlyCharges"] >= 90) & (df["PaymentMethod"] == "Mailed check") , "NEW_Char_Pay_NOM"] = "expensiveMail"
df.loc[(df["MonthlyCharges"] < 30) & (df["PaymentMethod"] == "Bank transfer (automatic)") , "NEW_Char_Pay_NOM"] = "very_cheapBank"
df.loc[((df["MonthlyCharges"] >= 30) & (df["MonthlyCharges"] < 60)) & (df["PaymentMethod"] == "Bank transfer (automatic)"), "NEW_Char_Pay_NOM"] = "cheapBank"
df.loc[((df["MonthlyCharges"] >= 60) & (df["MonthlyCharges"] < 90)) & (df["PaymentMethod"] == "Bank transfer (automatic)"), "NEW_Char_Pay_NOM"] = "normalBank"
df.loc[(df["MonthlyCharges"] >= 90) & (df["PaymentMethod"] == "Bank transfer (automatic)") , "NEW_Char_Pay_NOM"] = "expensiveBank"
df.loc[(df["MonthlyCharges"] < 30) & (df["PaymentMethod"] == "Credit card (automatic)") , "NEW_Char_Pay_NOM"] = "very_cheapCredit"
df.loc[((df["MonthlyCharges"] >= 30) & (df["MonthlyCharges"] < 60)) & (df["PaymentMethod"] == "Credit card (automatic)"), "NEW_Char_Pay_NOM"] = "cheapCredit"
df.loc[((df["MonthlyCharges"] >= 60) & (df["MonthlyCharges"] < 90)) & (df["PaymentMethod"] == "Credit card (automatic)"), "NEW_Char_Pay_NOM"] = "normalCredit"
df.loc[(df["MonthlyCharges"] >= 90) & (df["PaymentMethod"] == "Credit card (automatic)") , "NEW_Char_Pay_NOM"] = "expensiveCredit"


# Kişinin toplam aldığı servis sayısı
df['NEW_TotalServices'] = (df[['PhoneService', 'InternetService', 'OnlineSecurity',
                                       'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                       'StreamingTV', 'StreamingMovies']]== 'Yes').sum(axis=1)


# Kişinin ortalama aylık gideri
df["NEW_AVG_Charges"] = df["TotalCharges"] / (df["tenure"] + 1)


# Güncel Fiyatın ortalama fiyata göre artışı
df["NEW_Increase"] = df["NEW_AVG_Charges"] / df["MonthlyCharges"]


# Servis başına ücret
df["NEW_AVG_Service_Fee"] = df["MonthlyCharges"] / (df['NEW_TotalServices'] + 1)

df.head()

# Yeni değişkenler türettik. şimdi de bu yeni değişkenleri kolonları güncellemeliyiz ;
cat_cols, num_cols, cat_but_car = grab_col_names(df)



#################################################
# Encoding işlemlerini gerçekleştirelim;
#################################################

# Label Encoding  ;
binary_cols = [col for col in cat_cols if df[col].dtype not in ["int64", "float64"] and df[col].nunique() == 2]
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe
for col in binary_cols:
    df = label_encoder(df, col)


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
rare_analyser(df, "Churn",cat_cols)
df = rare_encoder(df, 0.01)


# One-Hot Encoder ;
# Kategori sayısı 2 ile 10 arasında olanları ohe_cols olarak toplayalım ve işlem uygulayalım ;
ohe_cols = [col for col in df.columns if df[col].nunique() > 2 and df[col].nunique() <= 20]
def one_hot_encoder(dataframe, categorial_cols, drop_first=True) :
    dataframe = pd.get_dummies(dataframe, columns=categorial_cols, drop_first=drop_first)
    return dataframe
df = one_hot_encoder(df, ohe_cols, True)

df.head()


# Kategorileri güncelliyoruz.
cat_cols, num_cols, cat_but_car = grab_col_names(df)


# Gereksiz, kullanışsız değişken var mı diye baktık. Eğer varsa ya rare yapacaktık ya da silecektik. Ama burada yok.
useless_cols = [col for col in df.columns if df[col].nunique() == 2 and (df[col].value_counts() / len(df) < 0.01).any(axis=None)]


#################################################
# Numerik değişkenleri Standartlaştırıyoruz ;
#################################################


scaler = StandardScaler()
df[num_cols]  = scaler.fit_transform(df[num_cols])
df[num_cols].head()


#################################################
# Model Oluşturuyoruz ;
#################################################


y = df["Churn"]
X = df.drop(["Churn", "customerID"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)


# Sonuçlar doğrultusunda veri setinin dengesiz olduğunu gözlemledik ;
f,ax = plt.subplots(1,2,figsize=(18,8))
df['Churn'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('dağılım')
ax[0].set_ylabel('')
sns.countplot(x='Churn', data=df, ax=ax[1])
ax[1].set_title('Churn')
plt.show(block=True)

# Dengeli hale getirmek için Random undersampling uygulayacağız.
# rA dan önce eğitim setindeki sınıf sayısı
y_train.value_counts()

# Oversampling
from imblearn.over_sampling import RandomOverSampler
oversample = RandomOverSampler(sampling_strategy='minority')
X_randomover, y_randomover = oversample.fit_resample(X_train, y_train)


# Random undersampling sonra
y_randomover.value_counts()

rf_model = RandomForestClassifier(random_state=46).fit(X_randomover, y_randomover)


# Test
y_pred = rf_model.predict(X_test)
print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}") #  0.79
print(f"Recall: {round(recall_score(y_pred,y_test),3)}") # 0.65
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}") # 0.50
print(f"F1: {round(f1_score(y_pred,y_test), 2)}") # 0.57
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}") # 0.74

# Burada da doğruluk oranımızı analiz ettik...
print(classification_report(y_test, y_pred))


# Genel anlamda accuary, prec. gibi değerlerde bir tutarsızlık gözüküyor (değişkenlik) yani 10 katlı çapraz doğrulmaa şart ;
cv_results = cross_validate(rf_model, X, y, cv=5, scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

# şimdi artık daha kesin sonuçlarla çıkabiliriz ;
cv_results["test_accuracy"].mean() # 0.787
cv_results["test_recall"].mean() # 0.484
cv_results["test_precision"].mean() # 0.63
cv_results["test_f1"].mean() # 0.547
cv_results["test_roc_auc"].mean() # 0.824



##################################################################
# Yeni türettiğimiz değişkenlerin önem ve işe yarama oranını grafikle inceleyelim ;
##################################################################
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')
plot_importance(rf_model, X_train)


##################################################################
# Modellerin ön tanımlı değerlerine bakarak config.py'daki GridSearch edilecek paramsların aralıklarını girelim.
##################################################################


knn_model = KNeighborsClassifier().fit(X, y)
knn_model.get_params()

cart_model = DecisionTreeClassifier(random_state=1).fit(X, y)
cart_model.get_params()

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
rf_model.get_params()

xgboost_model = XGBClassifier(random_state=17).fit(X, y)
xgboost_model.get_params()

lightgbm_model = LGBMClassifier(random_state=17).fit(X, y)
lightgbm_model.get_params()

##################################################################
# Modelin Overfit olup olmadığını test edelim ;
##################################################################

train_accuracy = rf_model.score(X_randomover, y_randomover)
test_accuracy = rf_model.score(X_test, y_test)


print(f"Training Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")
# Olmuş. Bunu düzeltmenin bikaç yolu var. Onlardan biri de modelin ön tanımlı değerleri ile oynamak ;

rf_params = {"max_depth": [8, 15, 22],
             "max_features": [5, 7, "sqrt", "auto"],
             "min_samples_split": [20, 29, 39],
             "n_estimators": [100, 200]}

from sklearn.model_selection import GridSearchCV

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
rf_best_grid.best_params_
rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)


# final modelini test ettiğimizde amacımıza ulaştık. devam ediyoruz ;
train_accuracy = rf_final.score(X_randomover, y_randomover)
test_accuracy = rf_final.score(X_test, y_test)


##################################################################
# Modelimizde f1 skoru düşük. Bunun için modelin en iyi f1 skorunu bulup onun threshold değerini kullandık ;
##################################################################

y_probs = rf_model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
f1_scores = 2*(precision*recall)/(precision+recall)

# En iyi f1 score değerine bakıyoruz ve fikir ediniyoruz.
best_f1_score = np.max(f1_scores)
best_threshold = thresholds[np.argmax(f1_scores)] # 0.37

# Değerleri deneyerek 0.42 değerinin en uygun olduğunu görüyoruz.
y_pred_updated = (y_probs > 0.42).astype(int)

# Yeni değerler ;
print(f"Accuracy: {round(accuracy_score(y_pred_updated, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred_updated,y_test),3)}")
print(f"Precision: {round(precision_score(y_pred_updated,y_test), 2)}")
print(f"F1: {round(f1_score(y_pred_updated,y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred_updated,y_test), 2)}")

df.head()




