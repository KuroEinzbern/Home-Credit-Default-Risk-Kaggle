from pathlib import Path

# __file__ es la posición de este script. 
# .parent.parent.parent sube desde 'config.py' hasta la raíz del proyecto.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Definición de carpetas principales
DATA_DIR = PROJECT_ROOT / "data"
NOTEBOOKS= PROJECT_ROOT / "notebooks"
METADATA_DIR = PROJECT_ROOT / "metadata"

# Rutas específicas a archivos
DUMP_FROM_NOTEBOOKS= NOTEBOOKS / "dumps_from_notebooks"
APPLICATION_TRAIN = DATA_DIR / "application_train.csv"
APPLICATION_TEST = DATA_DIR / "application_test.csv"
BUREAU = DATA_DIR / "bureau.csv"
BUREAU_BALANCE = DATA_DIR / "bureau_balance.csv"
CREDIT_CARD_BALANCE= DATA_DIR / "credit_card_balance.csv"
INSTALLMENTS_PAYMENTS= DATA_DIR / "installments_payments.csv"
POS_CASH_BALANCE = DATA_DIR / "POS_CASH_balance.csv"
PREVIOUS_APPLICATION= DATA_DIR / "previous_application.csv"
SCHEMA_JSON = METADATA_DIR / "schema.json"

# Crear carpetas si no existen (útil para notebooks nuevos)
DATA_DIR.mkdir(exist_ok=True)