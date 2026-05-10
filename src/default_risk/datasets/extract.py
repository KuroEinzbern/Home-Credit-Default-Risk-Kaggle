
from pathlib import Path

import pandas as pd


def download_dataset():
    NotImplemented


def split_table(table_path: Path, train_ids: set, test_ids: set, output_dir: Path):

    train_out_path = output_dir / f"{table_path.stem}.train.parquet"
    test_out_path = output_dir / f"{table_path.stem}.test.parquet"

    if train_out_path.exists() and test_out_path.exists():
            print(f"{table_path.name} already processed, skipped")
            return
    
    print(f"Procesing: {table_path.name}...")
    df = pd.read_csv(table_path)

    df_train = df[df['SK_ID_CURR'].isin(train_ids)]
    df_test = df[df['SK_ID_CURR'].isin(test_ids)]
    

    
    df_train.to_parquet(train_out_path, index=False)
    df_test.to_parquet(test_out_path, index=False)
    
    print(f"{table_path.name} splitted in train and test files")

def split_dataset(dataset_path: Path, output_path: Path):
    output_path.mkdir(parents=True, exist_ok=True)
    df_app_train = pd.read_csv(dataset_path / "application_train.csv", usecols=["SK_ID_CURR"])
    df_app_test = pd.read_csv(dataset_path / "application_test.csv", usecols=["SK_ID_CURR"])

    train_ids = set(df_app_train["SK_ID_CURR"])
    test_ids = set(df_app_test["SK_ID_CURR"])

    tables_to_split = [
        "bureau.csv",
        "credit_card_balance.csv",
        "installments_payments.csv",
        "POS_CASH_balance.csv",
        "previous_application.csv",
        "credit_card_balance.csv"
    ]

    for table_name in tables_to_split:
        file_path = dataset_path / table_name
        split_table(file_path, train_ids, test_ids, output_path)

