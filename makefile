ifeq ($(OS),Windows_NT)
    SHELL := bash.exe
else
    SHELL := bash
endif
.PHONY: data features all

data:
	python src/default_risk/data/make_dataset.py

features:
	python src/default_risk/features/build_features.py

train:
	python src/default_risk/models/train_model.py

model: data features train
	echo "Pipeline completo ejecutado"