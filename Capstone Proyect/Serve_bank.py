import pickle
import pandas as pd
from flask import Flask, request, jsonify
import xgboost as xgb

FINAL_THRESHOLD = 0.3366 
MODEL_FILE = 'Model_XGBOOST_bank.bin'

with open(MODEL_FILE, 'rb') as f_in:
    dv, model = pickle.load(f_in) 

app = Flask('deposit')


@app.route('/predict', methods=['POST'])
def predict():
 
    customer = request.get_json()

    X = dv.transform([customer])

    features = list(dv.get_feature_names_out())

    d_X = xgb.DMatrix(X, feature_names=features)
    y_pred_prob = model.predict(d_X)[0]


    deposit_decision = y_pred_prob >= FINAL_THRESHOLD
    

    # 6. Formatear Resultado
    result = {
        'Deposit_probability': float(y_pred_prob),
        'Deposit_decision': bool(deposit_decision),
        'umbral_usado': FINAL_THRESHOLD
    }
    return jsonify(result)

if __name__ == '__main__':
    # Usar host='0.0.0.0' para que Docker lo exponga correctamente
    app.run(debug=False, host='0.0.0.0', port=9696)


