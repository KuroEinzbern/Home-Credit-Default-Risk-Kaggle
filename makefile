ifeq ($(OS),Windows_NT)
    SHELL := bash.exe
else
    SHELL := bash
endif

.PHONY: data features train model predict optimize all

install:
	python -m venv venv
	@echo "installing dependencies and the project in editable mode..."
	pip install -r requirements.txt
	pip install -e .

docker_up:
	docker compose up -d --build

docker_down:
	docker compose down

setup: install docker-up

data:
	python src/default_risk/data/make_dataset.py

features:
	python src/default_risk/features/build_features.py

train:
	python src/default_risk/models/train_model.py

train-xgbm:
	python src/default_risk/models/train_model.py --only_xgb

train-lgbm:
	python src/default_risk/models/train_model.py -only_lgbm

model: data features train
	echo "Pipeline completo ejecutado"

predict: 
	python src/default_risk/models/predict_model.py

optimize: 
	python src/default_risk/models/optimize_model.py