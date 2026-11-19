FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY common ./common
COPY services/api ./services/api

CMD ["uvicorn", "services.api.server:app", "--host", "0.0.0.0", "--port", "8090"]
