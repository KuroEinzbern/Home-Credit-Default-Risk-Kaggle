from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from default_risk.config import CLEANS_DIR, DATA_DIR, PROCESSED_DIR, RAW_DATA_DIR, SPLITS_DIR, CANONIC_DIR
from default_risk.data.clean import clean_bureau, clean_credit_card_balance, clean_installments_payments, clean_pos_cash_balance, clean_previous_application, clean_application_train,clean_bureau_balance
from default_risk.data.extract import download_dataset, split_dataset, canonizate
from collections.abc import Callable

from default_risk.data.process import process_application_train, process_bureau, process_bureau_balance, process_credit_card_balance, process_installments_payments, process_pos_cash_balance, process_previous_application
canonizated_tables: dict ={}

def generate_cleaner_paths(table_name: str, split: str):
    p_in = SPLITS_DIR / f'{table_name}_{split}.parquet'
    p_out = CLEANS_DIR / f'{table_name}_{split}-cleaned.parquet'
    return p_in, p_out

def generate_process_paths(table_name: str, split: str):
    p_in = CLEANS_DIR / f'{table_name}_{split}-cleaned.parquet'
    p_out = PROCESSED_DIR / f'{table_name}_{split}-processed.parquet'
    return p_in, p_out

cleaning_dict: dict[str, Callable] = {
        "bureau_balance" : clean_bureau_balance,
        "bureau": clean_bureau,
        "installments_payments" : clean_installments_payments,
        "credit_card_balance" : clean_credit_card_balance,
        "POS_CASH_balance" : clean_pos_cash_balance,
        "previous_application" : clean_previous_application,
        "application" : clean_application_train,

}

procesing_dict: dict[str, Callable] = {
        "bureau_balance": process_bureau_balance,
        "bureau": process_bureau,
        "installments_payments" : process_installments_payments,
        "credit_card_balance": process_credit_card_balance,
        "POS_CASH_balance": process_pos_cash_balance,
        "previous_application" : process_previous_application,
        "application": process_application_train

}

cleaning_tables_list: list= list(cleaning_dict.keys())
procesing_tables_list: list= list(procesing_dict.keys())

@contextmanager
def log_step(table_name: str):
    print(f"Processing aggregations of {table_name}...", end='', flush=True)
    try:
        yield
        print(" Done.")
    except Exception as e:
        print(" FAILED!")
        raise e
    
def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    CLEANS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    if not any(Path(RAW_DATA_DIR).iterdir()):
        download_dataset()
  

    canonizate()

    split_dataset(RAW_DATA_DIR,CANONIC_DIR, SPLITS_DIR)

    splits = ['train', 'test']
    for split in splits:
        print(f'Preparing split {split}')

        for table_name in cleaning_tables_list :
            print(f"Cleaning table {table_name}...", end='', flush=True)
            path_bronze_layer_file, path_silver_layer_file = generate_cleaner_paths(table_name, split)
            cleaner= cleaning_dict[table_name]
            cleaner(path_bronze_layer_file,path_silver_layer_file)
            print("Done.")


        with log_step("bureau_balance"):
            process_bureau_balance(*generate_process_paths("bureau_balance", split))

        with log_step("installments_payments"):
            process_installments_payments(*generate_process_paths("installments_payments", split))

        with log_step("credit_card_balance"):
            process_credit_card_balance(*generate_process_paths("credit_card_balance", split))

        with log_step("POS_CASH_balance"):
            process_pos_cash_balance(*generate_process_paths("POS_CASH_balance", split))

        with log_step("application"):
            process_application_train(*generate_process_paths("application", split))

        with log_step("bureau"):
            process_bureau(
                input_filepath=generate_process_paths("bureau", split)[0],
                bureau_balance_agg_filepath=generate_process_paths("bureau_balance", split)[1],
                output_filepath=generate_process_paths("bureau", split)[1]
            )

        with log_step("previous_application"):
            installments_base_out = generate_process_paths("installments_payments", split)[1]
            credit_card_base_out = generate_process_paths("credit_card_balance", split)[1]
            cash_balance_base_out = generate_process_paths("POS_CASH_balance", split)[1]

            process_previous_application(
                input_filepath=generate_process_paths("previous_application", split)[0],
                installments_filepath=installments_base_out,
                credit_card_filepath=credit_card_base_out,
                installments_time_window_filepath=installments_base_out.parent / f"{installments_base_out.stem}-temporal_window.parquet",
                credit_card_time_window_filepath=credit_card_base_out.parent / f"{credit_card_base_out.stem}-last_six_months_agg.parquet",
                cash_balance_time_window_filepath=cash_balance_base_out.parent / f"{cash_balance_base_out.stem}_time_window.parquet",
                output_filepath=generate_process_paths("previous_application", split)[1]
            )

        
if __name__ == "__main__":
    main()