from pathlib import Path
from types import SimpleNamespace
from default_risk.config import CLEANS_DIR, RAW_DATA_DIR, SPLITS_DIR
from default_risk.datasets.clean import clean_credit_card_balance, clean_installments_payments, clean_pos_cash_balance
from default_risk.datasets.extract import split_dataset

def generate_cleaner_paths(table_name: str, split: str):
    p_in = SPLITS_DIR / f'{table_name}.{split}.parquet'
    p_out = CLEANS_DIR / f'{table_name}.{split}-cleaned.parquet'
    return p_in, p_out

def main():


    splits = [ "train" ]
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    CLEANS_DIR.mkdir(parents=True, exist_ok=True)

    split_dataset(RAW_DATA_DIR, SPLITS_DIR)

    for split in splits:
        credit_card_in, credit_card_out = generate_cleaner_paths('credit_card_balance', split)
        clean_credit_card_balance( credit_card_in , credit_card_out )

        installment_payments_in, installment_payments_out = generate_cleaner_paths('installments_payments', split)
        clean_installments_payments(installment_payments_in, installment_payments_out)

        pos_cash_in, pos_cash_out = generate_cleaner_paths('POS_CASH_balance', split)
        clean_pos_cash_balance(pos_cash_in, pos_cash_out)



if __name__ == "__main__":
    main()