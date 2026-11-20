PYTHON ?= python3
DOCKER ?= docker

.PHONY: install fetch train test format lint build

install:
	$(PYTHON) -m pip install -r requirements-dev.txt

fetch:
	$(PYTHON) -m src.etl_fetch

train:
	$(PYTHON) -m src.train_forecast

test:
	pytest -q

format:
	black app src tests dags

lint:
	black --check app src tests dags
	flake8 app src tests dags --max-line-length=110 --extend-ignore=E203,W503

build:
	$(DOCKER) build -f app/Dockerfile -t btc-forecast-api:local .

