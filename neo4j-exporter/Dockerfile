FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir neo4j>=5.0.0
COPY exporter.py .
CMD ["python", "exporter.py"]
