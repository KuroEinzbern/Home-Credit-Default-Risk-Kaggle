from pathlib import Path
from types import SimpleNamespace
from default_risk.config import CLEANS_DIR, PROCESSED_DIR, RAW_DATA_DIR, SPLITS_DIR, CANONIC_DIR
from default_risk.data.clean import clean_bureau, clean_credit_card_balance, clean_installments_payments, clean_pos_cash_balance, clean_previous_application, clean_application_train,clean_bureau_balance
from default_risk.data.extract import download_dataset, split_dataset, canonizate
from collections.abc import Callable

from default_risk.data.process import process_bureau, process_installments_payments, process_previous_application
canonizated_tables: dict ={}

def generate_cleaner_paths(table_name: str, split: str):
    p_in = SPLITS_DIR / f'{table_name}.{split}.parquet'
    p_out = CLEANS_DIR / f'{table_name}.{split}-cleaned.parquet'
    return p_in, p_out

def generate_process_paths(table_name: str, split: str):
    p_in = CLEANS_DIR / f'{table_name}.{split}-cleaned.parquet'
    p_out = PROCESSED_DIR / f'{table_name}.{split}-processed.parquet'
    return p_in, p_out

cleaning_dict: dict[str, Callable] = {
        "bureau_balance" : clean_bureau_balance,
        "bureau": clean_bureau,
        "installments_payments" : clean_installments_payments,
        "credit_card_balance" : clean_credit_card_balance,
        "POS_CASH_balance" : clean_pos_cash_balance,
        "previous_application" : clean_previous_application,
        "application_train" : clean_application_train,
}

procesing_dict: dict[str, Callable] = {
        "bureau_balance" : lambda *args: None,
        "bureau": process_bureau,
        "installments_payments" : process_installments_payments,
        "credit_card_balance" : lambda *args: None,
        "POS_CASH_balance" : lambda *args: None,
        "previous_application" : process_previous_application,
        "application_train" : lambda *args: None,

}

tables_list: list= list(cleaning_dict.keys())


def main():

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    CLEANS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    if not any(Path(RAW_DATA_DIR).iterdir()):
        download_dataset()
  
    canonizate()

    split_dataset(RAW_DATA_DIR,CANONIC_DIR, SPLITS_DIR)

    for table_name in tables_list :
        path_bronze_layer_file, path_silver_layer_file = generate_cleaner_paths(table_name,"train")
        cleaner= cleaning_dict[table_name]
        cleaner(path_bronze_layer_file,path_silver_layer_file)

    for table_name in tables_list :
        input, output = generate_process_paths(table_name,"train")
        processer= procesing_dict[table_name]
        processer(input, output)

        
if __name__ == "__main__":
    main()