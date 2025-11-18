import requests
import json
import pandas as pd

# URL del servicio web corriendo localmente (puerto 9696)
URL = "http://localhost:9696/predict"



high_intent_data = {
    "month": "nov",
    "visitor_type": "returning_visitor",
    "weekend": 1,
    "operating_systems": 2,
    "browser": 2,
    "region": 3,
    "traffic_type":2,
    "administrative":5,
    "administrative_duration":150.0,
    "informational":0,
    "informational_duration":0.0,
    "product_related":30,
    "product_related_duration":2500.0,
    "bounce_rates":0.005,
    "exit_rates": 0.01,
    "page_values":26.0,
    "special_day":0.0
}

def test_prediction():
    try:
       # print(f"--- 1. Enviando solicitud POST a: {URL} ---")
        
        response = requests.post(URL, json=high_intent_data)
        
        if response.status_code == 200:
            result = response.json()
           # print("\n--- 2. Respuesta del Servicio (Status 200 OK) ---")
           # print(json.dumps(result, indent=4))
            
            prob = result.get('purchase_probability', 0.0)
            
            if result.get('purchase_decision') is True:
                 print(f"✅ ¡VERIFICACIÓN EXITOSA! Probabilidad ({prob:.4f}) > Umbral (0.3268).")
                 print("El modelo predice: COMPRA.")
            else:
                 print(f"⚠️ VERIFICACIÓN COMPLETADA. Probabilidad ({prob:.4f}) < Umbral (0.3268).")
                 print("El modelo predice: NO COMPRA.")
                 
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