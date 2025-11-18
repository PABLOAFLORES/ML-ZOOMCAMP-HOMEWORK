
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.tree import export_text
import xgboost as xgb
import pickle



df = pd.read_csv(r'C:\Users\TALIGENT\git\ML-ZOOMCAMP-HOMEWORK\First Proyect\Dataset\online_shoppers_intention.csv')



def camel_to_snake(name):
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
    return name.lower()

df.columns = [camel_to_snake(col) for col in df.columns]
df.columns = df.columns.str.replace('__', '_', regex=False)

features_categoricas = list(df.dtypes[df.dtypes == 'object'].index)
for c in features_categoricas:
    df[c] = df[c].str.lower().str.replace(' ', '_')


df['revenue'] = df['revenue'].astype(int)
df['weekend'] = df['weekend'].astype(int)

cols_to_str = ['operating_systems', 'browser', 'region', 'traffic_type']
df[cols_to_str] = df[cols_to_str].astype(str)

df_full_train, df_test =train_test_split(df, test_size=0.2, random_state=1)

df_train, df_val =train_test_split(df_full_train, test_size=0.25, random_state=1) 

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test= df_test.reset_index(drop=True)
df_full_train.reset_index(drop= True)

y_train = df_train.revenue.values
y_val = df_val.revenue.values
y_test = df_test.revenue.values
y_full_train = df_full_train.revenue.values


del df_train['revenue']
del df_val['revenue']
del df_test['revenue']
del df_full_train ['revenue']

categoricas = [
    'month',
    'visitor_type',
    'weekend',
    'operating_systems',
    'browser',
    'region',
    'traffic_type',
    
]

numericas = [
    'administrative',
    'administrative_duration',
    'informational',
    'informational_duration',
    'product_related',
    'product_related_duration',
    'bounce_rates',
    'exit_rates',
    'page_values',
    'special_day'
]


# Armo funcion para obtener X_train y X_val
def Dict_Vect (df_train, df_val):
    train_dict = df_train[categoricas + numericas].to_dict(orient = 'records')
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(train_dict)

    val_dict = df_val[categoricas + numericas].to_dict(orient='records')
    X_val = dv.transform(val_dict) 

    return X_train, X_val,dv

X_train, X_val, dv = Dict_Vect(df_train,df_val)


#Parametros
xgb_params = {
    'eta': 0.1, 
    'max_depth': 7,
    'min_child_weight': 5,

    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'nthread': 8,
    'seed': 42,
    'verbosity': 1,
}

t = 0.3466

features = list(dv.get_feature_names_out())

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)

model = xgb.train(xgb_params, dtrain, num_boost_round=35)

y_pred_prob = model.predict(dval)
y_pred = (y_pred_prob >= t).astype(int)
sc_M4 = roc_auc_score(y_val, y_pred_prob)
F1_XGBOOST = f1_score(y_val, y_pred)

print("Evaluacion modelo XGBOOST TRAIN:\n"
      f"ROC-AUC M4: {sc_M4:.4f}\n"
      f"F1 Score M4: {F1_XGBOOST:.4f}")

X_full_train, X_test, dv = Dict_Vect(df_full_train, df_test)

features = list(dv.get_feature_names_out())

d_full_train = xgb.DMatrix(X_full_train, label=y_full_train, feature_names=features)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=features)

model = xgb.train(xgb_params, d_full_train, num_boost_round=35)

y_pred_prob = model.predict(dtest)
y_pred = (y_pred_prob >= t).astype(int)
sc_XGBOOST_final = roc_auc_score(y_test, y_pred_prob)
F1_XGBOOST_final = f1_score(y_test, y_pred)

print( "Evaluacion modelo Final"
      f"ROC-AUC XGBOOST FINAL: {sc_XGBOOST_final:.4f}\n"
      f"F1 XGBOOST FINAL: {F1_XGBOOST_final:.4f}"
)

output_file = f'Model_XGBOOST.bin'

with open(output_file,'wb') as f_out:
    pickle.dump((dv,model),f_out)

print(f'Model es saved in {output_file}')



