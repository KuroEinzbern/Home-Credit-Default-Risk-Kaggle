
from pathlib import Path
from types import SimpleNamespace
from default_risk.config import CLEANS_DIR, RAW_DATA_DIR, SPLITS_DIR
from default_risk.data.clean import clean_credit_card_balance, clean_installments_payments
from default_risk.data.extract import split_dataset


def main():

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    CLEANS_DIR.mkdir(parents=True, exist_ok=True)

    split_dataset(RAW_DATA_DIR, SPLITS_DIR)

    clean_credit_card_balance(SPLITS_DIR, CLEANS_DIR )
    clean_installments_payments(SPLITS_DIR, CLEANS_DIR)

if __name__ == "__main__":
    main()