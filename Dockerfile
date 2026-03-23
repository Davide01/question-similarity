FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN pip install "poetry>=2.0.0,<3.0.0" && \
    poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root --no-interaction

COPY api.py .
COPY question_sim/ ./question_sim/

# Copy the MLflow tracking database and artifact store.
# Patch absolute host paths in mlflow.db to /app (the container's WORKDIR).
COPY mlflow.db .
COPY mlruns/ ./mlruns/
COPY patch_mlflow_db.py .
RUN python3 patch_mlflow_db.py && rm patch_mlflow_db.py

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
