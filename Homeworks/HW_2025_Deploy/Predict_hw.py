
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

model_file = 'pipeline_v1.bin'



record = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}



with open(model_file,'rb') as f_in:
    dv,model =pickle.load(f_in)


X = dv.transform([record])
y_pred = model.predict_proba(X)[0,1]
print(y_pred)



