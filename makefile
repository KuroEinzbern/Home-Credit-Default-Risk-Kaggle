.PHONY: data features all

data:
	python src/default_risk/data/make_dataset.py

features: data
	python src/default_risk/features/build_features.py

all: features
	echo "Pipeline completo ejecutado"