import pandas as pd
import numpy as np
import seaborn as sns
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

df = pd.read_csv(r'C:\Users\TALIGENT\git\ML-ZOOMCAMP-HOMEWORK\Midterm Proyect\Dataset\bank.csv')


features_categoricas = list(df.dtypes[df.dtypes == 'object'].index)
for c in features_categoricas:
    df[c] = df[c].str.lower().str.replace(' ', '_')

binary_cols = ['default', 'housing', 'loan', 'deposit'] 

for col in binary_cols:
    df[col] = df[col].astype(str).str.strip().str.lower()
    
    df[col] = df[col].replace({'yes': 1, 'no': 0}).astype(int)

df_full_train, df_test =train_test_split(df, test_size=0.2, random_state=1)

df_train, df_val =train_test_split(df_full_train, test_size=0.25, random_state=1) 

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test= df_test.reset_index(drop=True)
df_full_train.reset_index(drop= True)

y_train = df_train.deposit.values
y_val = df_val.deposit.values
y_test = df_test.deposit.values
y_full_train = df_full_train.deposit.values


del df_train['deposit']
del df_val['deposit']
del df_test['deposit']
del df_full_train ['deposit']


categoricas = [
'default',
'housing',
'loan',
'job',
'marital',
'education',
'contact',
'month',
'poutcome'
    
]

numericas = [
'age',
'balance',
'day', 
'duration',
'campaign',
'pdays',
'previous'
]



def Dict_Vect (df_train, df_val):
    train_dict = df_train[categoricas + numericas].to_dict(orient = 'records')
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(train_dict)

    val_dict = df_val[categoricas + numericas].to_dict(orient='records')
    X_val = dv.transform(val_dict) 

    return X_train, X_val,dv

X_train, X_val, dv = Dict_Vect(df_train,df_val)

# Parametros
xgb_params = {
    'eta': 0.3, 
    'max_depth': 3,
    'min_child_weight': 7,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'nthread': 8,
    'seed': 42,
    'verbosity': 1,
}

t = 0.3366


features = list(dv.get_feature_names_out())

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)

model = xgb.train(xgb_params, dtrain, num_boost_round=105)

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



output_file = f'Model_XGBOOST_bank.bin'

with open(output_file,'wb') as f_out:
    pickle.dump((dv,model),f_out)

print(f'Model es saved in {output_file}')


