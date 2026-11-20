# Demonstration Runbook

## 1. Local model and API

```bash
python -m src.etl_fetch
python -m src.train_forecast
MODEL_DIR="$PWD/models" flask --app app.app run --port 5000
pytest -q
```

Capture `/health`, `/predict`, and `/metrics` responses.

## 2. Docker

```bash
docker build -f app/Dockerfile -t btc-forecast-api:local .
docker run --rm -p 5000:5000 -v "$PWD/models:/app/models" btc-forecast-api:local
docker compose up -d
docker compose ps
```

## 3. Kubeflow compilation

```bash
python -m pip install 'kfp>=2.7,<3'
python kubeflow/btc_kfp_pipeline.py
```

The generated `kubeflow/btc_kfp_pipeline.yaml` can be uploaded to a Kubeflow Pipelines UI.

## 4. Minikube deployment

```bash
minikube start --driver=docker --cpus=4 --memory=4096
minikube addons enable metrics-server
minikube addons enable ingress
minikube image load btc-forecast-api:local
kubectl apply -k k8s/
kubectl get pods,svc,pvc,hpa,ingress
kubectl port-forward service/btc-forecast-service 8081:80
curl -X POST http://127.0.0.1:8081/predict \
  -H 'Content-Type: application/json' -d '{"forecast_days":7}'
```

## 5. Autoscaling and resilience evidence

```bash
locust -f loadtest/locustfile.py --headless -u 100 -r 20 \
  --run-time 3m --host http://127.0.0.1:8081
kubectl get hpa --watch
kubectl get pods -l app=btc-forecast-api
kubectl delete pod <one-pod-name>
kubectl get pods -l app=btc-forecast-api --watch
```

## 6. Airflow

```bash
docker compose up airflow-init
docker compose up -d postgres airflow-webserver airflow-scheduler api
docker compose exec airflow-webserver airflow dags trigger btc_daily_forecast_pipeline
docker compose exec airflow-webserver airflow dags list-runs -d btc_daily_forecast_pipeline
```

Open `http://localhost:8080` and capture the DAG graph and successful run.

## Submission evidence checklist

- Passing pytest and lint output
- Model validation MAE/RMSE/MAPE
- Flask API sample JSON
- Docker build and running container
- Airflow DAG graph and task success
- Kubeflow compiled YAML
- GitHub Actions green run and GHCR image
- `kubectl get pods,svc,pvc,hpa,ingress`
- HPA scale-up under Locust traffic
- Deleted pod being recreated

