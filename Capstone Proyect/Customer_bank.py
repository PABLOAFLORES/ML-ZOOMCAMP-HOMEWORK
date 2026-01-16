import requests
import json
import pandas as pd

# URL del servicio web corriendo localmente (puerto 9696)
URL = "http://localhost:8080/predict"



high_intent_data = {
    "age": 32,
    "job": "management",
    "marital": "single",
    "education": "tertiary",
    "default": 0,    
    "balance": 2500, 
    "housing": 0,    
    "loan": 0,       
    "contact": "cellular",
    "day": 15,
    "month": "aug",
    "duration": 400, 
    "campaign": 1,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown"
}

def test_prediction():
    try:
       # print(f"--- 1. Enviando solicitud POST a: {URL} ---")
        
        response = requests.post(URL, json=high_intent_data)
        
        if response.status_code == 200:
            result = response.json()
           # print("\n--- 2. Respuesta del Servicio (Status 200 OK) ---")
           # print(json.dumps(result, indent=4))
            
            prob = result.get('Deposit_probability', 0.0)
            
            if result.get('Deposit_decision') is True:
                 print(f"✅ ¡VERIFICACIÓN EXITOSA! Probabilidad ({prob:.4f}) > Umbral (0.3268).")
                 print("El modelo predice: Deposit.")
            else:
                 print(f"⚠️ VERIFICACIÓN COMPLETADA. Probabilidad ({prob:.4f}) < Umbral (0.3268).")
                 print("El modelo predice: Churn")
                 
        else:
            print(f"\n❌ ERROR DE HTTP: Código {response.status_code}")
            print("Mensaje de error del servidor (Verifique que serve.py esté corriendo):")
            print(response.text)

    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: No se pudo conectar al servicio.")
        print("Asegúrese de que el script 'serve.py' esté corriendo en otra terminal en 'http://0.0.0.0:9696'.")
    except Exception as e:
        print(f"\n❌ ERROR INESPERADO: {e}")

if __name__ == '__main__':
    test_prediction()


