# Bitcoin Forecast MLOps Pipeline

Production-style MLOps assignment that fetches BTC-USD data, trains an autoregressive
7-day forecasting model, serves predictions through Flask, orchestrates retraining with
Airflow/Kubeflow, and deploys the API to Kubernetes.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.etl_fetch
python -m src.train_forecast
gunicorn --bind 0.0.0.0:5000 'app.app:create_app()'
```

Test the API:

```bash
curl http://localhost:5000/health
curl -X POST http://localhost:5000/predict \
  -H 'Content-Type: application/json' -d '{"forecast_days": 7}'
curl http://localhost:5000/metrics
```

Run tests with `pytest -q`. See `docs/RUNBOOK.md` for Docker, Airflow, Kubeflow,
Minikube, autoscaling, and evidence-capture commands.

