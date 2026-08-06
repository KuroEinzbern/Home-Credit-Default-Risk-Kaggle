from pathlib import Path

# __file__ es la posición de este script. 
# .parent.parent.parent sube desde 'config.py' hasta la raíz del proyecto.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Definición de carpetas principales
DATA_DIR = PROJECT_ROOT / "data/"
NOTEBOOKS= PROJECT_ROOT / "notebooks"
METADATA_DIR = PROJECT_ROOT / "metadata"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = PROJECT_ROOT / "models"


RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MASTER_DATA_DIR = DATA_DIR / "master"

CANONIC_DIR = INTERIM_DATA_DIR / "00_Canonicals"
SPLITS_DIR = INTERIM_DATA_DIR / "01_Splits"
CLEANS_DIR = INTERIM_DATA_DIR / "02_Cleans"
PROCESSED_DIR = INTERIM_DATA_DIR / "03_Processed"

MODEL_PARAMS = PROJECT_ROOT / "params.yaml"
DUMP_FROM_NOTEBOOKS= NOTEBOOKS / "dumps_from_notebooks"

# Rutas específicas a archivos
APPLICATION_TRAIN = RAW_DATA_DIR / "application_train.csv"
APPLICATION_TEST = RAW_DATA_DIR / "application_test.csv"
BUREAU = RAW_DATA_DIR / "bureau.csv"
BUREAU_BALANCE = RAW_DATA_DIR / "bureau_balance.csv"
CREDIT_CARD_BALANCE= RAW_DATA_DIR / "credit_card_balance.csv"
INSTALLMENTS_PAYMENTS= RAW_DATA_DIR / "installments_payments.csv"
POS_CASH_BALANCE = RAW_DATA_DIR / "POS_CASH_balance.csv"
PREVIOUS_APPLICATION= RAW_DATA_DIR / "previous_application.csv"
SCHEMA_JSON = METADATA_DIR / "schema.json"

# Crear carpetas si no existen
DATA_DIR.mkdir(exist_ok=True)
RAW_DATA_DIR.mkdir(exist_ok=True)
DUMP_FROM_NOTEBOOKS.mkdir(exist_ok=True)
CANONIC_DIR.mkdir(exist_ok=True,parents=True)
SPLITS_DIR.mkdir(exist_ok=True,parents=True)
CLEANS_DIR.mkdir(exist_ok=True,parents=True)
INTERIM_DATA_DIR.mkdir(exist_ok=True,parents=True)
ARTIFACTS_DIR.mkdir(exist_ok=True,parents=True)
PROCESSED_DIR.mkdir(exist_ok=True,parents=True)


