FROM python:3.11-slim

# Directorio de trabajo
WORKDIR /app

# Instalamos gunicorn y las librerías necesarias directamente
RUN pip install --no-cache-dir \
    gunicorn \
    flask \
    pandas \
    numpy \
    xgboost \
    scikit-learn

# Copiamos los archivos indispensables
COPY Serve_bank.py .
COPY Model_XGBOOST_bank.bin .

# Exponemos el puerto
EXPOSE 9696

# Comando de ejecución 
CMD ["gunicorn", "--bind", "0.0.0.0:9696", "Serve_bank:app"]