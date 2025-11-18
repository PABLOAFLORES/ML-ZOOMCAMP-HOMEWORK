FROM python:3.11-slim

WORKDIR /app

RUN pip install gunicorn uv

COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

COPY serve.py .
COPY Model_XGBOOST.bin .

EXPOSE 9696

CMD ["gunicorn", "--bind", "0.0.0.0:9696", "serve:app"]

