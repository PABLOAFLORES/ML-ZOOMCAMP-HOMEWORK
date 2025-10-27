import requests

url = 'http://127.0.0.1:9696/predict'

customer_id = 'XYZ-123'
customer = {
    "gender": "female",
    "seniorcitizen": 0,
    "partner": "yes",
    "dependents": "no",
    "phoneservice": "no",
    "multiplelines": "no_phone_service",
    "internetservice": "dsl",
    "onlinesecurity": "no",
    "onlinebackup": "yes",
    "deviceprotection": "no",
    "techsupport": "no",
    "streamingtv": "no",
    "streamingmovies": "no",
    "contract": "month-to-month",
    "paperlessbilling": "yes",
    "paymentmethod": "electronic_check",
    "tenure": 1,
    "monthlycharges": 29.85,
    "totalcharges": 29.85
}

response = requests.post(url,json= customer).json()
print(response)

if response ['churn'] == True:
    print(f'Enviar promo a cliente ID= {customer_id}')
else:
     print(f'NO Enviar promo a cliente ID= {customer_id}')



