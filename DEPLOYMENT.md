# Deployment & Monitoring Strategy

## CI/CD Pipeline

### Service Deployment Pipeline

Triggered when a new model is registered or when API code changes:

1. **Build** — Build the Docker image (`docker build -t question-sim-api .`).
2. **Integration test** — Spin up the container, run smoke tests against `/health` and `/similarity`.
3. **Push** — Push the image to a container registry
4. **Deploy** — Roll out to the target environment using e.g.,rolling update strategy.

### Tooling Options

- **GitHub Actions / Azure CI/CD** for pipeline orchestration.
- **MLflow** for model versioning and experiment tracking.
- **BigQuery** for dataset storage

## Infrastructure

### Recommended Setup

- **Container orchestration**: For example Kubernetes 
- **Model storage**: S3/GCS bucket behind the MLflow artifact store
- **Autoscaling**: HPA (Horizontal Pod Autoscaler) based on CPU/request latency, or managed autoscaling on Cloud Run/Fargate.
- **Load balancing**: An ingress controller or managed ALB in front of the service replicas.

### Configuration Management

Use environment variables (or a secrets manager) for runtime config:

- `MLFLOW_TRACKING_URI` — MLflow server address.
- `MODEL_RUN_ID` or `MODEL_PATH` — Which model artifact to load at startup.
- `SIMILARITY_THRESHOLD` — Configurable decision threshold.

## Observability

### Logging

- Emit structured JSON logs (e.g. via Python `logging` + `json` formatter).
- Log every prediction request: input lengths, latency, predicted label, and confidence score.
- Ship logs to a centralized platform (ELK, Loki + Grafana, CloudWatch, etc.).

### Metrics

Expose Prometheus-compatible metrics:

- **Request rate** — requests/sec to `/similarity`.
- **Latency** — p50/p95/p99 response times.
- **Error rate** — 4xx/5xx responses.

### Alerting

Set up alerts (Grafana Alerting, Opsgenie) for:

- Error rate spike (e.g. >1% 5xx over 5 min).
- Latency degradation (e.g. p99 > 500ms).
- Container restarts or health check failures.

### Model Monitoring

- **Prediction drift**: Monitor the distribution of similarity scores and label ratios. A sudden shift may indicate degraded model quality.
- **Periodic re-training and re-evaluation**: Schedule retrainings of models after every e.g., 10% data increase or after every 90 days.

### Dashboard

A Grafana monitoring dashboard could be useful to display system metrics.
