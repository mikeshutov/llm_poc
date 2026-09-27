FROM python:3.12-slim

WORKDIR /app
COPY services/attribute_classifier/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY services/attribute_classifier ./services/attribute_classifier

ENV ATTRIBUTE_CLASSIFIER_HOST=0.0.0.0
ENV ATTRIBUTE_CLASSIFIER_PORT=8081
EXPOSE 8081
CMD ["python", "-m", "services.attribute_classifier.server"]
