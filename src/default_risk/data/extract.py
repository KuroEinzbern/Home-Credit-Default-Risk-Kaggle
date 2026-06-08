
from pathlib import Path
from collections.abc import Callable
import pandas as pd
from default_risk.config import BUREAU, BUREAU_BALANCE, CANONIC_DIR

def download_dataset():
    NotImplemented



all_tables_name = [
        "bureau",
        "application_train",
        "credit_card_balance",
        "installments_payments",
        "POS_CASH_balance",
        "previous_application",
        "credit_card_balance",
        "bureau_balance"
    ]   


def canonizate_bureau_balance():
    bureau= pd.read_csv(BUREAU)
    bureau_balance=pd.read_csv(BUREAU_BALANCE)
    merged= bureau_balance.merge(bureau[["SK_ID_CURR","SK_ID_BUREAU"]],how="left",on="SK_ID_BUREAU")

    #The original file of bureau_balance is arround of 350 MB and the resultant file of this merge was arround of 700mb.
    #In order to avoid that unnecesary heavy file, we cast explict all the types, don't persist the index and drop the orphan rows.
    #(result 200mb less of file)

    merged = merged.dropna(subset=["SK_ID_CURR"])
    merged["SK_ID_CURR"] = merged["SK_ID_CURR"]
    merged= merged.astype({
    "SK_ID_CURR": "uint32",
    "SK_ID_BUREAU": "uint32",
    "STATUS": "category",
    "MONTHS_BALANCE": "int16"
    })
    merged.to_csv(CANONIC_DIR / "bureau_balance.csv",index=False)
    return



canonization_dict: dict[str, Callable] = {
        "bureau_balance" : canonizate_bureau_balance
}


def split_table(table_path: Path, train_ids: set, test_ids: set, output_dir: Path):

    train_out_path = output_dir / f"{table_path.stem}.train.parquet"
    test_out_path = output_dir / f"{table_path.stem}.test.parquet"

    if train_out_path.exists() and test_out_path.exists():
            print(f"{table_path.name} already processed, skipped")
            return
    
    print(f"Procesing: {table_path.name}...")
    print(table_path)
    df = pd.read_csv(table_path)

    df_train = df[df['SK_ID_CURR'].isin(train_ids)]
    df_test = df[df['SK_ID_CURR'].isin(test_ids)]
    
    df_train.to_parquet(train_out_path, index=False)

    if len(df_test) > 0:
        df_test.to_parquet(test_out_path, index=False)
    
    print(f"{table_path.name} splitted in train and test files")

def split_dataset(dataset_path: Path, canonic_path: Path, output_path: Path):

    output_path.mkdir(parents=True, exist_ok=True)
    df_app_train = pd.read_csv(dataset_path / "application_train.csv", usecols=["SK_ID_CURR"])
    df_app_test = pd.read_csv(dataset_path / "application_test.csv", usecols=["SK_ID_CURR"])
    train_ids = set(df_app_train["SK_ID_CURR"])
    test_ids = set(df_app_test["SK_ID_CURR"])


    for table_name in all_tables_name:
        if(table_name in canonization_dict) :
            file_path = (canonic_path / table_name).with_suffix(".csv")  
        else :
            file_path= (dataset_path / table_name).with_suffix(".csv")
        split_table(file_path, train_ids, test_ids, output_path)



def canonizate():
    for a_table_in_dict,function_to_canonizate in canonization_dict.items():
        print("--applying preprocessing to " + a_table_in_dict + "-----\n")
        function_to_canonizate()
    return


