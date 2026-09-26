FROM python:3.12-slim

WORKDIR /app
COPY services/reranker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY services/reranker ./services/reranker

ENV RERANKER_HOST=0.0.0.0
ENV RERANKER_PORT=8080
EXPOSE 8080
CMD ["python", "-m", "services.reranker.server"]
