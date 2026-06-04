
from pathlib import Path

import pandas as pd


def process_bureau(input_filepath: Path, output_filepath: Path):
    print('Starting bureau table processing...')
    print(f'Loading data from: {input_filepath}')
    bureau_df = pd.read_parquet(input_filepath)

    bureau_df.sort_values(["id_curr", "days_credit"],inplace=True,ascending=False)
    last_three = bureau_df.groupby("id_curr").head(3)
    last_three = last_three.copy()
    last_three["loan_order"] = last_three.groupby("id_curr").cumcount() + 1
    last_three_columns = last_three.pivot(index="id_curr", columns="loan_order")
    last_three_columns.columns =[f"{col}_prev_{rank}" for col, rank in last_three_columns.columns]

    def get_first_mode(x):
        mode_series = x.mode()
        return mode_series.iloc[0] if not mode_series.empty else pd.NA

    bureau_aggregattted = bureau_df.groupby("id_curr").agg({
        "id_curr": ["count"],
        "flag_have_credit_day_overdue": ["count"],
        "credit_type": [get_first_mode],
        "amt_credit_sum_limit": ["max"],
        "cnt_credit_prolong": ["max", "min", "mean"],
        "amt_credit_sum": ["max", "min", "mean"],
    })

    bureau_aggregattted.columns = [
        f"{col}_{stat.__name__ if callable(stat) else stat}" 
        for col, stat in bureau_aggregattted.columns
    ]

    combined_rows = pd.concat([bureau_aggregattted, last_three_columns],  axis=1 )
    combined_rows.reset_index(inplace=True)
    combined_rows.to_parquet(output_filepath, index=False)
