
# Default Credit Risk Solution

**Authors:** Alejandro Bastida, Ignacio Pugliese.  
**Competition:**  [https://www.kaggle.com/competitions/home-credit-default-risk/](https://www.kaggle.com/competitions/home-credit-default-risk/)
**Tech Stack:** Python, Pandas, Scikit-Learn, XGBoost, LightGBM, MLflow, Optuna, Docker, GNU Make.

To continue improving our machine learning skills and showcase our work as ML engineers, we solved the iconic Kaggle competition **Home Credit Default Risk**, achieving **0.798+** on the Public Leaderboard and **0.797+** on the Private Leaderboard using a single model with only **130 engineered features**.

The project was built entirely from scratch. No Kaggle notebooks, kernels, or code from other participants were used during development.

The project was designed with production readiness in mind, incorporating the software architecture, engineering practices, and technologies commonly used in the industry today. These design decisions are described in the following sections.

# How to Install the Project

The project includes a Makefile with a command `make setup`. This command starts the Docker Compose services, creates the virtual environment to isolate the dependencies, installs the dependencies listed in `requirements.txt`, and installs the project in editable mode (following the `src` project layout). Therefore, having **Docker and Docker Compose installed** and working correctly on your system, and **GNU Make** installed, simply running `make setup` is enough to have the project ready to use.

## About the Pipeline and How to Recreate the Results

The _Makefile_ provides the following commands: `data`, `features`, `train`, `predict`, `optimize`.

-   Running `make data` downloads the dataset (if the raw data is not present in the project) and executes the entire data engineering pipeline, following these steps:
    1.  **Schema preparation:** Prepares the data before the train–test split (some IDs were missing, so we decided to denormalize to simplify the pipeline).
    2.  **Train–test split:** In order to avoid leakage in the next steps, we split the raw data into training and test sets.  
        (`data/interim/01_Splits`)
    3.  **Data cleaning:** This step cleans the data, creating Parquet files of the Silver layer.  
        (`data/interim/02_Cleans`)
    4.  **Pre-aggregation feature engineering:** Creates additional features that improve the information captured during table aggregation.
    5.  **Table aggregation:** Processes the aggregations for each table.  
        (`root/data/interim/03_Processed`)
-   `make features` creates the Gold layer (`root/data/interim/03_Master`). To do so, it uses the Parquet files stored in `root/data/interim/     03_Processed` and joins the aggregated features into a single table. This step also creates some cross-table features.
-   `make train`: This step takes the Gold layer, performs feature pruning to keep the optimal set of features for the selected model, trains the model, logs the training results in MLflow, and exports the trained model through serialization.  
    (`root/models`)
    
    **Notes:** This command have options. You can use `train-xgb` or `train-lgbm` to trail one of both models. Using just `train`, both models will be trained and exported. Also, this step uses MLflow to track the results; therefore, the corresponding Docker container must be running.
    
-   `make predict`: This step performs inference, takes the serialized model generated in the previous step, and predicts on the test dataset, creating the submission file with the results.  
    (`root/data/master/`)
    
    **Note:** Like the previous step, it have the alternatives: `predict-xgb`, `-predict-lgbm`. Use only `predict` generates predictions using an ensemble of both models.
    
-   `make optimize`: This step is only used to run Optuna and search for hyperparameters. The optimal hyperparameters are already configured.

## About the EDA

We performed an extensive exploratory data analysis. Everything is documented in the notebooks located at `root/notebooks/eda`, with one notebook per table.

The structure of these notebooks is as follows: the first cell contains the context and imports, the second one is a Markdown cell with a summary of all the findings and decisions taken based on the observations, and the third cell runs the script to generate a variable profiling report (details in `src/scripts/variable_profiling.py`). After that, the cells contain the EDA code. The remaining cells are idempotent, so they can be run independently of the rest.

## Architecture, Technologies and Engineering Practices

This project follows and implements:

**Medallion architecture:**  
In order to follow industry best practices, the data engineering process follows the Medallion architecture. It contains the Bronze layer (the Parquet files after the Schema preparation step in our case), the Silver layer (cleaned data), and the Gold layer (the single table used to train the model).

**SRC layout:**  
We structured our project to have reusable code. Every script is located in the `src` folder, and we install the project as an editable dependency. This, paired with `root/src/config.py`, avoids import-related issues and keeps the codebase clean and maintainable.

**MLflow:**  
To track our experimentation, we used a local MLflow server from the very first day, with PostgreSQL running in Docker (`root/docker-compose.yml`). We save the results, parameters, model, feature importance, and dataset signature used to train the model for every run. We created an integration to track nested cross-validation runs and keep everything organized (details in `root/src/scripts/cv_mlflow_integration.py`).

**Documentation:**  
We created a data dictionary and other documents to track data lineage and feature decisions.

**Optuna:**  
For model tuning, we used Optuna with 5-fold cross-validation and between 100–300 trials. We iteratively searched for hyperparameters and performed feature pruning using permutation feature importance until reaching an efficient combination of features and hyperparameters.

**Notebooks:**  

We have EDA notebooks, process notebooks (where we try new features before implementing them in the pipeline), and model experimentation notebooks where we run experiments to try new ideas, document the gains in each table, and prune and tune the models

-   **EDA Notebooks:** For exploratory data analysis.
    
-   **Process Notebooks:** notebooks where we test new features ideas before integrating them into the main pipeline.
    
-   **Model Experimentation:** For running experiments, check gains, feature pruning, and model tuning.
