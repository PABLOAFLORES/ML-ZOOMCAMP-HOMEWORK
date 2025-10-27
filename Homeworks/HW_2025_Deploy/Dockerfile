FROM agrigorev/zoomcamp-model:2025

WORKDIR /code

#COPY pipeline_v2.bin .

COPY App_Service.py .

RUN pip install fastapi uvicorn scikit-learn==1.6.1


EXPOSE 8000

CMD ["uvicorn", "App_Service:app", "--host", "0.0.0.0", "--port", "8000"]