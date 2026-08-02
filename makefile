ifeq ($(OS),Windows_NT)
    SHELL := bash.exe
else
    SHELL := bash
endif

FLAGS ?=

.PHONY: data features train model predict optimize all

install:
	python -m venv venv
	@echo "Instalando dependencias y el paquete en modo editable..."
	pip install -r requirements.txt
	pip install -e .

docker_up:
	docker compose up -d --build

docker_down:
	docker compose down

setup: venv docker-up

data:
	python src/default_risk/data/make_dataset.py $(FLAGS)

features:
	python src/default_risk/features/build_features.py $(FLAGS)

train:
	python src/default_risk/models/train_model.py $(FLAGS)

model: data features train
	echo "Pipeline completo ejecutado"

predict: 
	python src/default_risk/models/predict_model.py $(FLAGS)

optimize: 
	python src/default_risk/models/optimize_model.py $(FLAGS)