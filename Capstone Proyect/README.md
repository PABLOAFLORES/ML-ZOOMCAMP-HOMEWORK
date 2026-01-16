## Bank Marketing Strategy - Capstone Project

**1. Description of the Problem**
The goal of this project is to predict whether a client will subscribe to a term deposit (target variable y) following a marketing campaign. Using a dataset from a Portuguese banking institution, we built a machine learning pipeline to help the bank identify the most promising customers, optimizing their marketing efforts and increasing conversion rates.


**2. Data**
Source: The dataset is included in the Dataset/ folder of this repository.
Origin: It contains information such as age, job, marital status, education, and previous campaign outcomes.
Target: y (binary: 'yes','no').

**3. Exploratory Data Analysis (EDA)**
During the exploratory phase, the following key steps were performed to understand the underlying patterns of the bank marketing data:
Missing Value Analysis: The dataset was audited for completeness, confirming no significant null values that would require complex imputation.
Target Variable Distribution: A notable class imbalance was identified, as most clients did not subscribe to the deposit. This guided the choice of evaluation metrics, shifting focus toward ROC AUC and F1-score rather than simple accuracy.
Feature Correlation:
The duration variable (last contact duration) showed a very strong positive correlation with the subscription outcome.
Previous campaign success (poutcome) and socioeconomic factors were identified as critical predictors.
Outlier Detection: Numerical features such as age and balance were analyzed to ensure extreme values did not negatively bias the model's learning process.

**4. Model Selection and Training**
The model selection process followed an iterative approach to find the best balance between complexity and performance:
*1. Evaluated Models*
Logistic Regression: Used as a baseline model due to its simplicity and interpretability.
Random Forest: Implemented to capture non-linear relationships and interactions between features.
XGBoost (Final Choice): Selected as the final model because it provided the highest performance on tabular data and offered robust handling of class imbalance through hyperparameter tuning.
*2. Hyperparameter Tuning*
Optimization was performed using Grid Search to fine-tune key XGBoost parameters such as max_depth, eta (learning rate), and n_estimators.
The primary optimization metric was ROC AUC to ensure the model maintains high discriminative power across different decision thresholds.
*3. Final Training and Export*
The final model was trained on the full training set (after cross-validation) and exported to a binary format (Model_XGBOOST_bank.bin) using the pickle library for deployment.


**Prerequisites**
Before running this project, ensure you have the following installed:
*Docker Desktop*: To build the images and run the local cluster.
*Kind*: To create the Kubernetes cluster (kind.exe).
*Kubectl*: To manage Kubernetes resources (kubectl.exe).
*Python 3.13 and uv*: For dependency management and running the local scripts.


**5. Project Structure**
Notebook_bank.ipynb: Contains data cleaning, EDA, feature importance (XGBoost), and hyperparameter tuning.

*Train_bank.py*: Script that trains the final XGBoost model and saves it as Model_XGBOOST_bank.bin.

*Serve_bank.py*: Flask web service that loads the model and serves predictions.

*Customer_bank.py*: Client script to test the service.

*pyproject.toml*: Dependency management using uv.

*Dockerfile*: Containerization instructions for the service.

*deployment.yaml*: Kubernetes manifests for Deployment and Service.


**6. Instructions on How to Run**
Local Environment (with uv)

Install dependencies:
Bash
*uv sync*

Run the service:
Bash
*uv run Serve_bank.py*

Running with Docker
Build the image:
Bash
*docker build -t bank-prediction-app:latest .*

Run the container:
Bash
*docker run -it --rm -p 9696:9696 bank-prediction-app:latest*

Running with Kubernetes (Kind)
Create the cluster and load the image:
Bash
*kind create cluster*
*kind load docker-image bank-prediction-app:latest*

Deploy the app:
Bash
*kubectl apply -f deployment.yaml*

Access the service (Port Forward):
Bash
*kubectl port-forward service/bank-service 8080:80*

Test the prediction:
Bash
*uv run Customer_bank.py*


**7. Deployment Evidence**
You can find the video/images of the interaction with the deployed service in the Evidence/ folder or via the following link: [INSERT YOUR VIDEO LINK HERE].