from locust import HttpUser, between, task


class ForecastUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(4)
    def forecast(self):
        self.client.post("/predict", json={"forecast_days": 7})

    @task(1)
    def health(self):
        self.client.get("/health")

