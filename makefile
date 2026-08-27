ifeq ($(OS),Windows_NT)
    SHELL := bash.exe
    PYTHON := venv/Scripts/python
else
    SHELL := bash
    PYTHON := venv/bin/python
endif

.PHONY: data features train model predict optimize all

install:
	python -m venv venv
	@echo "installing dependencies and the project in editable mode..."
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down

setup: install docker-up

data:
	$(PYTHON) src/default_risk/data/make_dataset.py

features:
	$(PYTHON) src/default_risk/features/build_features.py

train:
	$(PYTHON) src/default_risk/models/train_model.py

train-xgb:
	$(PYTHON) src/default_risk/models/train_model.py --only_xgb

train-lgbm:
	$(PYTHON) src/default_risk/models/train_model.py --only_lgbm

model: data features train
	@echo "Pipeline completo ejecutado"

predict: 
	$(PYTHON) src/default_risk/models/predict_model.py

predict-xgb: 
	$(PYTHON) src/default_risk/models/predict_model.py --only_xgb

predict-lgbm: 
	$(PYTHON) src/default_risk/models/predict_model.py --only_lgbm

optimize: 
	$(PYTHON) src/default_risk/models/optimize_model.py