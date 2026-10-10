FROM python:3.12-slim

WORKDIR /app
COPY services/completion_evaluator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY services/completion_evaluator ./services/completion_evaluator

ENV COMPLETION_EVALUATOR_HOST=0.0.0.0
ENV COMPLETION_EVALUATOR_PORT=8082
EXPOSE 8082
CMD ["python", "-m", "services.completion_evaluator.server"]
