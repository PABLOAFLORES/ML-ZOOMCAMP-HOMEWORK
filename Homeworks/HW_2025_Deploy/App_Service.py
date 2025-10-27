import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from fastapi import FastAPI
from pydantic import BaseModel

#model_file = 'pipeline_v1.bin'
model_file = 'pipeline_v2.bin'

with open(model_file,'rb') as f_in:
    pipeline =pickle.load(f_in)

# Inicio FastApi
app = FastAPI()

# Esquema de entrada
class Client(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

# Endpoint de Prediccion
@app.post("/predict")
def predict(client_data: Client):
    record = client_data.model_dump() # Convierto el obj Cliente a diccionario

    Matriz_Prob = pipeline.predict_proba([record])
    y_pred = Matriz_Prob[0, 1]

    Decision = y_pred>0.5

    return  {
        'decision_probability' : float(y_pred),
        'decision' : bool(Decision)
    }
