ENG
# 🚀 Final Project: Online Shoppers Intention Prediction

## 📝 1. Problem Description

The objective of this project is to predict whether an e-commerce website visitor will or will not make a revenue transaction (a purchase) based on their navigation behavior within the site.

### Prediction and Utility

* **Business Problem:** The e-commerce needs to identify users with a high probability of converting to optimize the experience and resources.
* **Prediction:** This is a **Binary Classification** task. The model predicts the **`Revenue`** variable, where **1** is purchase and **0** is no purchase.
* **Utility:** Identifying a customer with high purchase intention allows applying personalized strategies in real-time (e.g., discount offers) and optimizing resources.

---

## 💾 2. Dataset

The project uses the **`Online Shoppers Intention`** dataset, which collects session attributes from 12,330 users. The data includes duration metrics, bounce rates, and page values.

### Key Features

Feature analysis was performed:

The numerical variables with the most interaction are "bounce\_rates" and "exit\_rates"; therefore, we need to use a model to eliminate this multicollinearity.

The categorical variables **traffic\_type** and **month** are those that provide the most information about the purchase, which confirms that traffic sources and seasonality (e.g., months of high consumption like Nov or Dec) have the strongest relationship with the target variable.

**EDA Analysis**
Feature names and their content were normalized when they are categorical.
Some variables ['operating\_systems', 'browser', 'region', 'traffic\_type'] are numerical but are categorical.
Boolean features are transformed into binary, including "revenue".
There is a notorious imbalance: 
revenue
0    84.52%
1    15.47%

---

## 🔬 3. Methodology and Model Selection

The modeling phase was focused on binary classification, using **ROC-AUC** as the main metric, given the imbalanced nature of the problem.

### Models Evaluated

Logistic Regression, Decision Tree Classifier, Random Forest, and XGBoost models were evaluated.

### Selected Model

**XGBoost (eXtreme Gradient Boosting)** was selected for providing the best discrimination capacity between classes, resulting in the highest ROC-AUC as well as the highest F1 Score. The final model was optimized for a **decision threshold of "0.3268"**.

---

## 🛠️ 4. Project Structure and Instructions (Continuation)

### 4.1. Key Files

| File | Function |
| :--- | :--- |
| `notebook.ipynb`| Analysis and evaluation of the different models |
| `train.py` | Trains the final model and saves the binary (`Model_XGBOOST.bin`). |
| `serve.py` | Script that loads the model and exposes it as a Flask/Gunicorn web service. |
| `requirements.txt` | Project dependencies, managed with `uv`. |
| **`Dockerfile`** | Instructions for building the container image. |
| **`test.py`** | Client script to test the deployed web service. |

### 4.2. Local Dependency Setup (Optional)
For dependency management, `uv` is used to ensure a fast and consistent installation in the virtual environment.

**Create and Activate the Virtual Environment:**

Bash
uv venv
.\.venv\Scripts\activate  

Install Dependencies:

Bash
uv pip install -r requirements.txt
uv pip install requests # Necessary for the test script


4.3. Model Training
Execute the training script to generate the final model:

Bash
python train.py

Result: The Model_XGBOOST.bin file will be generated.

4.4. Docker Deployment
Ensure Docker Desktop is running. These steps create the service container:

Build the Docker Image:

Bash
docker build -t revenue-predictor .

Run the Container: Launches the service on port 9696. Keep this terminal open.

Bash
docker run -it --rm -p 9696:9696 revenue-predictor


4.5. Prediction Verification
Open a second terminal and run the client script test.py to verify that the container responds correctly.

Bash
python test.py


## 🔗 5. Deployment Evidence

### Service URL

The prediction service is available via **POST** requests at:

http://localhost:9696/predict

### Interaction Proof

A video demonstrating the successful interaction with the deployed service is included https://github.com/PABLOAFLORES/ML-ZOOMCAMP-HOMEWORK/blob/main/First%20Proyect/Video_deploy.mp4. The video shows the Gunicorn server running in the Docker container (Terminal 1) and the successful execution of the `test.py` script (Terminal 2) with a 200 status code and a "BUY" prediction.
----------------------------------------------------------------------------------------------------

SPA
#  🚀 Proyecto Final: Predicción de Intención de Compra en Línea (Online Shoppers Intention)

## 📝 1. Descripción del Problema

El objetivo de este proyecto es predecir si un visitante de un sitio web de comercio electrónico (e-commerce) realizará o no una transacción de ingresos (una compra) basándose en su comportamiento de navegación dentro del sitio.

### Predicción y Utilidad

* **Problema de Negocio:** El e-commerce necesita identificar a los usuarios con alta probabilidad de convertir para optimizar la experiencia y los recursos.
* **Predicción:** Se trata de una tarea de **Clasificación Binaria**. El modelo predice la variable **`Revenue` (Ingresos)**, donde **1** es compra y **0** es no compra.
* **Utilidad:** Identificar a un cliente con alta intención de compra permite aplicar estrategias personalizadas en tiempo real (ej. ofertas de descuento) y optimizar los recursos.


## 💾 2. Dataset

El proyecto utiliza el conjunto de datos **`Online Shoppers Intention`**, que recopila atributos de sesión de 12.330 usuarios. Los datos incluyen métricas de duración, tasas de rebote y valores de página.

### Características Clave

Se realizo el analisis de Features:

Las variables numericas las que tienen mas interaccion son "bounce_rates" y "exit_rates" por lo tanto necesitamos usar un modelo para eliminamos esta multicolinealidad 

Las variables categoricas traffic_type y month son las que aportan mayor información sobre la compra, lo que confirma que las fuentes de tráfico y la estacionalidad (ej., meses de alto consumo como nov o dec) tienen la relación más fuerte con la variable objetivo

**Analisis EDA**
Se normalizaron los nombres de las features y su contenido cuando son categoricas.
Hay algunas variables ['operating_systems', 'browser', 'region', 'traffic_type'] que estan como numericas pero son categoricas.
Se transforman las features booleanas en binarias incluyendo "revenue".
Hay un notorio desblance: 
revenue
0    84.52%
1    15.47%

## 🔬 3. Metodología y Selección del Modelo

La fase de modelado se enfocó en la clasificación binaria, utilizando el **ROC-AUC** como métrica principal, dada la naturaleza desbalanceada del problema.

### Modelos Evaluados

Se evaluaron modelos de Logistic Regression, Desicion Tree Clasiffer, Random Forest y XGBoost.

### Modelo Seleccionado

Se seleccionó **XGBoost (eXtreme Gradient Boosting)** por proporcionar la mejor capacidad de discriminación entre las clases, resultando en el ROC-AUC más alto como tambien con el F1 Score mas alto. El modelo final fue optimizado para un **umbral de decisión de "0.3268".


## 🛠️ 4. Estructura del Proyecto e Instrucciones (Continuación)

### 4.1. Archivos Clave

| Archivo | Función |
| `notebook.ipynb`| Analisis y evaluacion de los distintos modelos |
| `train.py` | Entrena el modelo final y guarda el binario (`Model_XGBOOST.bin`). |
| `serve.py` | Script que carga el modelo y lo expone como un servicio web Flask/Gunicorn. |
| `requirements.txt` | Dependencias del proyecto, gestionadas con `uv`. |
| **`Dockerfile`** | Instrucciones para construir la imagen del contenedor. |
| **`test.py`** | Script cliente para probar el servicio web desplegado. |

### 4.2. Instalación de Dependencias Locales (Opcional)
Para la gestión de dependencias, se utiliza uv para asegurar una instalación rápida y consistente en el entorno virtual.

**Crear y Activar el Entorno Virtual:**


uv venv

***.\.venv\Scripts\activate***  # En Windows PowerShell

Instalar Dependencias:

***uv pip install -r requirements.txt***
***uv pip install requests # Necesario para el script de prueba*** 

### 4.3. Entrenamiento del Modelo
Ejecuta el script de entrenamiento para generar el modelo final:

***python train.py***

Resultado: Se generará el archivo Model_XGBOOST.bin.

### 4.4. Despliegue con Docker
Asegúrate de que Docker Desktop esté corriendo. Estos pasos crean el contenedor de servicio:

Construir la Imagen de Docker:

***docker build -t revenue-predictor .***


Ejecutar el Contenedor: Lanza el servicio en el puerto 9696. Mantén esta terminal abierta.


***docker run -it --rm -p 9696:9696 revenue-predictor***

### 4.5. Verificación de la Predicción
Abre una segunda terminal y ejecuta el script cliente test.py para verificar que el contenedor responde correctamente.

***python test.py***

## 🔗 5. Evidencia del Despliegue

### URL del Servicio Desplegado

El servicio de predicción está disponible mediante solicitudes **POST** en:

$$\text{http://localhost:9696/predict}$$

### Prueba de Interacción

Se incluye un video que demuestra la interacción exitosa con el servicio desplegado https://github.com/PABLOAFLORES/ML-ZOOMCAMP-HOMEWORK/blob/main/First%20Proyect/Video_deploy.mp4. El video muestra el servidor Gunicorn ejecutándose en el contenedor de Docker (Terminal 1) y la ejecución exitosa del script `test.py` (Terminal 2) con un código de estado 200 y una predicción de "COMPRA".