# %%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
import pickle



##Parametros###

C = 1.0
n_splits = 5
output_file = f'Model_C={C}.bin'



##Preparacion Datos###
df = pd.read_csv(r"C:\Users\TALIGENT\Downloads\data-week-3.csv.csv")
df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn == 'yes').astype(int)


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values

del df_train['churn']
del df_val['churn']
del df_test['churn']


numerical = ['tenure', 'monthlycharges', 'totalcharges']

categorical = [
    'gender',
    'seniorcitizen',
    'partner',
    'dependents',
    'phoneservice',
    'multiplelines',
    'internetservice',
    'onlinesecurity',
    'onlinebackup',
    'deviceprotection',
    'techsupport',
    'streamingtv',
    'streamingmovies',
    'contract',
    'paperlessbilling',
    'paymentmethod',
]


#Entrenamiento modelo###
def train(df_train,y_train,C=1.0):
    train_dict = df_train[categorical + numerical].to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(train_dict)

    model = LogisticRegression(C=C, max_iter=10000)
    model.fit(X_train, y_train)
    return dv, model


def prediccion(df_val,dv,model):
    val_dict = df_val[categorical + numerical].to_dict(orient='records')
    X_val = dv.transform(val_dict)
    y_pred = model.predict_proba(X_val)[:, 1]
    return y_pred



#Evaluacion modelo#

print(f'Validando con C={C}')
Kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)


scores = []

fold = 0

for(train_idx, val_idx) in Kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.churn.values
    y_val = df_val.churn.values

    dv, model = train(df_train,y_train,C=C)
    y_pred = prediccion(df_val,dv,model)

    auc = roc_auc_score(y_val,y_pred)
    scores.append(auc)

    print(f'auc con fold {fold} es {auc}')
    fold = fold + 1


#Entrenamos modelo#

print('Train final model')
dv, model = train(df_full_train,df_full_train.churn.values,C = 1)
y_pred = prediccion(df_test,dv,model)

auc = roc_auc_score(y_test, y_pred)

print(f'auc = {auc}')

##### SAVE MODEL PICKLE#####

with open(output_file,'wb') as f_out:
    pickle.dump((dv,model),f_out)

print(f'Model es saved in {output_file}')
