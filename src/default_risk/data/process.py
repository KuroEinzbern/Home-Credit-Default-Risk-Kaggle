
from pathlib import Path

import pandas as pd
import numpy as np
import default_risk.config as cfg
from default_risk.config import CLEANS_DIR, PROCESSED_DIR
from default_risk.scripts.auxiliar_eda_function import recreate_and_sort_series_given_rows, recreate_and_sort_the_serie_given_ids


def processed_path(table_name: str, split: str = "train", artifact: str = "") -> Path:
    artifact_suffix = f"-{artifact}" if artifact else ""
    return PROCESSED_DIR / f"{table_name}_{split}-processed{artifact_suffix}.parquet"

def process_bureau_balance(input_filepath: Path, output_filepath: Path):
    bureau_balance_df = pd.read_parquet(input_filepath)

    column_order_reference= "months_balance"
    bureau_balance_df.sort_values(["id_bureau",column_order_reference],inplace=True)
    bureau_balance_df["raw_length"] = bureau_balance_df.groupby("id_bureau").transform("size")
    next_status = bureau_balance_df.groupby("id_bureau")["status"].shift(-1)

    bureau_balance_df["not_last_row_active"] = (bureau_balance_df["status"] != "C") & (next_status != "C")
    bureau_balance_df["amount_rows_with_activity"] = bureau_balance_df.groupby("id_bureau")["not_last_row_active"].transform("sum")
    bureau_balance_df["amount_rows_with_activity"]= bureau_balance_df["amount_rows_with_activity"] + 1

    bureau_balance_df["status"]= bureau_balance_df["status"].str.lower()
    status_dict = {'c': 0, 'x': 0, '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5}
    bureau_balance_df["status_score"] = bureau_balance_df["status"].map(status_dict)
    bureau_balance_df["status"]=bureau_balance_df["status"].astype("category")
    bureau_balance_df=pd.get_dummies(bureau_balance_df,columns=["status"])

    due_all = bureau_balance_df[bureau_balance_df["status_score"].isin([1, 2, 3, 4, 5])]
    most_recents_months_with_dues= due_all.groupby(["id_bureau", "status_score"])["months_balance"].max().unstack()


    most_recents_months_with_dues.columns = [f"months_since_{int(col)}_status" for col in most_recents_months_with_dues.columns]
    recency_matrix = most_recents_months_with_dues.reset_index()
    bureau_balance_df= bureau_balance_df.merge(recency_matrix,how="left",on="id_bureau")
    bureau_balance_df["is_delincuency"] = bureau_balance_df["status_score"] > 0
    bureau_severe= bureau_balance_df [bureau_balance_df["is_delincuency"]]
    months= bureau_severe.groupby("id_bureau")["months_balance"].transform("max")
    bureau_balance_df["months_since_delincuency"] = bureau_balance_df["id_bureau"].map(months)

    agg_from_bureau_balance_dict = {
        "raw_length": ["first"], 
        "amount_rows_with_activity": ["first"],
        "potential_on_going_loan": ["first"],
        "closing_month": ["first"],
        "months_since_delincuency" : ["first"],
        "months_balance" : ["min","max"],
        "status_score": ["max","mean","std"],

        #categoricals
        "status_0": ["mean","sum"],
        "is_delincuency": ["mean","sum"], 
    }

    aggregated_bureau_balance= bureau_balance_df.groupby("id_bureau").agg(agg_from_bureau_balance_dict)
    aggregated_bureau_balance.columns = [
        f"balance_{col[0]}" if col[1] == "first" else f"balance_{col[0]}_{col[1]}"
        for col in aggregated_bureau_balance.columns
    ]
    agg_metrics_df= aggregated_bureau_balance.reset_index()
    agg_metrics_df.to_parquet(output_filepath)

def process_bureau(input_filepath: Path, output_filepath: Path, bureau_balance_agg_filepath: Path):
    bureau_df = pd.read_parquet(cfg.CLEANS_DIR / input_filepath)
    bureau_balance_agg= pd.read_parquet(bureau_balance_agg_filepath)

    bureau_df=bureau_df.merge(bureau_balance_agg,how="left",on="id_bureau")
    bureau_df['has_bureau_balance_data'] = bureau_df['balance_months_balance_min'].notna().astype(int)
    bureau_balance_agg.head()

    bureau_df["ratio_credit_annuity"]= np.where(bureau_df["amt_annuity"] !=0, bureau_df["amt_credit_sum"] / bureau_df["amt_annuity"] , np.nan )  
    bureau_df["completetitud_ratio"] = np.where(bureau_df["amt_credit_sum_debt"]!=0,bureau_df["amt_credit_sum"] / bureau_df["amt_credit_sum_debt"],np.nan)
    bureau_df["ratio_debt_limit"] =  np.where(bureau_df["amt_credit_sum"] !=0, bureau_df["amt_credit_sum_limit"] / bureau_df["amt_credit_sum"], np.nan )  

    bureau_df["credit_active"]=bureau_df["credit_active"].str.lower()
    bureau_df= pd.get_dummies(bureau_df, columns=["credit_active"], dtype=int)

    bureau_df.sort_values(["id_curr", "days_credit"],inplace=True,ascending=False)
    last_two = bureau_df.groupby("id_curr").head(1)
    last_two = last_two.copy()
    last_two["loan_order"] = last_two.groupby("id_curr").cumcount() + 1


    last_two_columns = last_two.drop(columns=["id_bureau"]).pivot(index="id_curr", columns="loan_order")
    last_two_columns.columns = [f"bureau_{col[0]}_loan_{col[1]}" for col in last_two_columns.columns]
    last_two_columns= last_two_columns.reset_index()

    bureau_df["log_amt_credit_sum"] = np.log1p(bureau_df["amt_credit_sum"])

    bureau_df["credit_type"]= bureau_df["credit_type"].str.lower()
    bureau_df= pd.get_dummies(bureau_df,columns= ["credit_type"])



    bureau_agg_dic_active= {
        "id_curr": ["count"],

        #monetary
        "amt_credit_sum": ["max", "mean","sum","std"],
        "amt_credit_sum_limit": ["max","mean","min","std"],
        "amt_annuity" : ["max","mean","min","std"], #
        "amt_credit_sum_debt" : ["max","mean","sum","std"],


        #log_transformed
        "log_amt_credit_sum": ["mean","std"],

        #counters
        "cnt_credit_prolong": ["max","mean"], #,"sum"
        "days_credit_update": ["min","max","mean"], 
        "days_credit": ["min","max","mean"], 
        "days_credit_enddate": ["max","mean"], 

        "ratio_credit_annuity" : ["max","mean","min"],
        "completetitud_ratio" : ["mean","min"],
        "amt_annuity_is_missing" : ["mean","sum"],
        "have_amt_credit_sum_overdue" : ["mean","sum"],
        "amt_credit_max_overdue" :["max","mean","sum"],

        #categorical
        "credit_type_credit card" : ["sum"], # ,"sum"
        "credit_type_mortgage" : ["mean","sum"], # ,"sum"
        "credit_type_microloan" : ["mean","sum"], #  ,"sum"
        "credit_type_consumer credit": ["sum"], #,"sum"
    }

    bureau_agg_dic_closed = {
        "id_curr": ["count"],

        #monetary
        "amt_credit_sum": ["max", "mean","sum","std"],
        "amt_credit_sum_limit": ["max","mean","min"], #,"std"
        "amt_annuity" : ["max","mean","min","std"],
        "amt_credit_sum_debt" : ["max","mean","sum","std"],

        #log_transformed
        "log_amt_credit_sum": ["mean","std"],

        #counters
        #"cnt_credit_prolong": ["max","mean"], #,"sum"
        "days_credit_update": ["min","max","mean"], 
        "days_credit": ["min","max","mean"], 
        "days_enddate_fact": ["max"], 

        "ratio_credit_annuity" : ["max","mean","min"],
        "completetitud_ratio" : ["mean","min"],
        "credit_active_sold" : ["mean","sum"],
        "amt_annuity_is_missing" : ["mean","sum"],
        "amt_credit_max_overdue" :["max","mean","sum"],
        "amt_credit_max_overdue_is_missing" :["mean","sum"],

        #categorical
        "credit_type_credit card" : ["sum"], #,"sum"
        "credit_type_mortgage" : ["mean","sum"], #,"sum"
        "credit_type_microloan" : ["mean","sum"], #,"sum"
        "credit_type_consumer credit": ["sum"], #,"sum"
    }


    agg_from_bureau_balance_dict = {
        "balance_status_score_max": ["max"], #0.76 
        "balance_months_balance_max": ["max"], #0.76 
        "balance_months_balance_min": ["min"], #0.76 
        "balance_months_since_delincuency" : ["max"],#0.76 
        "balance_is_delincuency_sum" : ["max"],#0.76 
        "balance_is_delincuency_mean" : ["mean"],#0.76 
    }

    active_loans= bureau_df [bureau_df["credit_active_active"] == 1]
    closed_but_recent= (bureau_df["credit_active_active"] != 1) & (bureau_df["days_enddate_fact"] > -5080)
    non_active_loans=  bureau_df[closed_but_recent]
    bureau_active_aggregated = active_loans.groupby("id_curr").agg(bureau_agg_dic_active|agg_from_bureau_balance_dict ).add_prefix("active_")
    bureau_non_active_aggregated = non_active_loans.groupby("id_curr").agg(bureau_agg_dic_closed|agg_from_bureau_balance_dict ).add_prefix("closed_")

    bureau_aggregated= bureau_active_aggregated.merge(bureau_non_active_aggregated,how="outer",on="id_curr")

    bureau_aggregated.columns= [f"{col[0]}_{col[1]}" for col in bureau_aggregated.columns]
    bureau_aggregated= bureau_aggregated.reset_index()

    bureau_final_df= last_two_columns.merge(bureau_aggregated,how="left",on="id_curr")

    bureau_final_df.to_parquet(cfg.PROCESSED_DIR / output_filepath)

def process_installments_payments(input_filepath: Path, output_filepath: Path):
    installments_payment_df = pd.read_parquet(input_filepath)

    #we starting catching this because we are probably cutting some parts of the temporal sequence
    installments_payment_df["raw_size_serie"]= installments_payment_df.groupby("id_prev").transform("size")
    installments_payment_df["amount_of_versions_in_sequence"] = installments_payment_df.groupby("id_prev")["num_instalment_version"].transform("nunique")

    #we gonna use the knowlege recolected in the EDA. 
    #For more details look eda_installments_payments.ipynb decisions summary 1#

    #creating auxiliar columns
    installments_payment_df["next_payment_value"] = installments_payment_df.groupby("id_prev")["days_entry_payment"].shift(-1)
    nan_amount= installments_payment_df.groupby("id_prev")["days_entry_payment_is_missing"].transform("sum")

    #defining the mask to separate the differents cases of missings values 
    have_missing_entry_payment_mask= (installments_payment_df["next_payment_value"].isna())
    not_a_deadtail_mask= (installments_payment_df["days_entry_payment"].isna() ) & ( installments_payment_df["next_payment_value"].notna())

    #we want to catch the cases of  dead-tail so we starting filtering that cases with nans but that are not dead tails
    non_dead_tail_nans= installments_payment_df[not_a_deadtail_mask]
    ids_with_nulls_that_are_non_deadtails= non_dead_tail_nans["id_prev"].unique()
    excluding_nans_non_deadtails_mask= ~(installments_payment_df["id_prev"].isin(ids_with_nulls_that_are_non_deadtails))
    series_without_problematic_nans = installments_payment_df[excluding_nans_non_deadtails_mask]

    #now this ID are series where the nans are deadtails, so count nans in "days_entry_payment" o "amt_payment" is calculate
    #the lenght of the deadtail and we save that value in a new column
    ids_with_deadtails= series_without_problematic_nans[series_without_problematic_nans["days_entry_payment"].isna()]["id_prev"].unique()
    installments_payment_df["dead_tail_length"]= np.where(installments_payment_df["id_prev"].isin(ids_with_deadtails),  nan_amount, 0)

    #also in the decision summary of the EDA we define a criteria to incomplete series 
    starting_instalment_number= installments_payment_df.groupby("id_prev")["num_instalment_number"].transform("first")
    starting_date= installments_payment_df.groupby("id_prev")["days_instalment"].transform("first")

    installments_payment_df["is_potentially_incomplete_sequence"] = ((starting_instalment_number >  1)  & (starting_date < -2890 ))

    potentially_on_going_id = installments_payment_df[installments_payment_df["days_instalment"] > (- 33)]["id_prev"].unique()
    installments_payment_df["potentially_on_going"]= installments_payment_df["id_prev"].isin(potentially_on_going_id)   

    installments_payment_df["is_underpayment"]= (installments_payment_df["amt_instalment"] >  installments_payment_df["amt_payment"]) & (installments_payment_df["amt_payment"] != 0)
    rows_with_underpayment= installments_payment_df [installments_payment_df["is_underpayment"] == True]
    installments_payment_df["days_of_underpayment"] = np.where(installments_payment_df["is_underpayment"], installments_payment_df["days_instalment"], np.nan)

    installments_payment_df["diff_expected_received"]= installments_payment_df["amt_instalment"] -  installments_payment_df["amt_payment"]
    installments_payment_df["diff_deadline_factical_payment"]= installments_payment_df["days_entry_payment"] - installments_payment_df["days_instalment"] 
    installments_payment_df["days_in_advance"] = installments_payment_df["days_instalment"] - installments_payment_df["days_entry_payment"]
    installments_payment_df["days_of_delinquency"]= installments_payment_df["diff_deadline_factical_payment"].clip(lower=0)
    installments_payment_df["days_in_advance"]= installments_payment_df["days_in_advance"].clip(lower=0)
    installments_payment_df["is_delinquency"] = installments_payment_df["days_of_delinquency"] > 0 




    def get_time_window_metrics(days_since, installments_payment_df) :    
        last_year= installments_payment_df[installments_payment_df["days_instalment"] > days_since]

        days_since = days_since * -1

        last_year_agg= last_year.groupby("id_curr").agg({
            "amt_instalment": ["max", "min","mean","sum"],
            "amt_payment": ["max", "min","mean","sum","std"],
            "days_of_delinquency":["mean","sum"],
            "days_in_advance":["mean","sum"],
            "diff_expected_received": ["max", "min", "mean", "sum"],
            "is_delinquency" : ["mean","sum"],
            "is_underpayment" : ["mean"],
            "extra_instalament":["sum","mean"],
            })


        last_year_agg.columns = [
        f"last_{days_since}_instalments_{col[0]}_{col[1]}"
        for col in last_year_agg.columns
        ]

        last_year_agg= last_year_agg.reset_index()

        last_year_agg[f"last_{days_since}_instalments_completion_ratio"] = np.where( last_year_agg[f"last_{days_since}_instalments_amt_instalment_sum"] > 0, last_year_agg[f"last_{days_since}_instalments_amt_payment_sum"] / last_year_agg[f"last_{days_since}_instalments_amt_instalment_sum"], 1.0 ) 

        last_year_agg.to_parquet(cfg.PROCESSED_DIR / "installments_payments_last_year_metrics.parquet")
        return last_year_agg



    next_installment_number = installments_payment_df.groupby("id_prev")["num_instalment_number"].shift(-1)
    next_version_number = installments_payment_df.groupby("id_prev")["num_instalment_version"].shift(-1)
    repeated_installment_mask= (installments_payment_df ["num_instalment_number"] == next_installment_number)
    underpayment_mask= (installments_payment_df[ "amt_payment" ] < installments_payment_df ["amt_instalment"])
    full_payment_mask= installments_payment_df[ "amt_payment" ] == installments_payment_df ["amt_instalment"] 
    installments_payment_df[ "repeated_for_underpayment" ] = (repeated_installment_mask) & (underpayment_mask)
    #installments_payment_df[ "repeated_for_reschedule" ] = (repeated_installment_mask) & (full_payment_mask)
    #installments_payment_df["repeated_for_payment_in_advance"] = (repeated_installment_mask) & (installments_payment_df[ "amt_payment" ] == 0) & (installments_payment_df["days_of_delinquency"] == 0)
    installments_payment_df["log_amt_instalment"]= np.log1p(installments_payment_df ["amt_instalment"] )
    installments_payment_df["log_amt_payment"]= np.log1p(installments_payment_df ["amt_payment"] )

    installments_payment_df["extra_instalament"] = (installments_payment_df["amount_of_versions_in_sequence"] > 1) & (installments_payment_df["raw_size_serie"] <95)  & (installments_payment_df ["num_instalment_number"] >= 100)
    interesting_cases= installments_payment_df[installments_payment_df["extra_instalament"] == True ]

    agg_metrics_df= installments_payment_df.groupby("id_prev").agg({

        #static values calculated for the entire squenece
        "raw_size_serie" : ["first"],
        "dead_tail_length" : ["first"],
        "potentially_on_going" : ["first"],
        "amount_of_versions_in_sequence" : ["first"],

        #for log transformated we want to catch the mean and the std (avoiding the impact of the heavy tail from this columns)
        "log_amt_instalment": ["mean","std"],   
        "log_amt_payment": ["mean","std"],

        #natural scale
        "amt_instalment": ["max", "min","median","sum"], #, "count"
        "amt_payment": ["max", "min","median","sum"],
        "extra_instalament":["sum","mean"],
    

        #computed_differences
        "diff_expected_received": ["max", "min", "median", "sum"],
        
        #categoricals
        "repeated_for_underpayment": ["mean","sum"],
        #"repeated_for_reschedule": ["mean","sum"],
        "is_delinquency" : ["mean","sum"],

        #counters
        #"days_instalment":["min"],
        "days_of_delinquency":["mean","max","sum"],
        "days_in_advance":["mean","max","sum"],
        "days_of_underpayment" : ["max"]
    })

    agg_metrics_df.columns = [
        f"instalments_{col[0]}" if col[1] == "first" else f"instalments_{col[0]}_{col[1]}"
        for col in agg_metrics_df.columns
    ]

    agg_metrics_df = agg_metrics_df.reset_index()

    agg_metrics_df["instalments_completion_ratio"] = np.where( agg_metrics_df["instalments_amt_instalment_sum"] > 0, agg_metrics_df["instalments_amt_payment_sum"] / agg_metrics_df["instalments_amt_instalment_sum"], 1.0 )  #if the debt is 0 or negative we assume competitud (1)   


    agg_metrics_last_three_months= get_time_window_metrics(-90, installments_payment_df)
    agg_metrics_last_year = get_time_window_metrics(-365, installments_payment_df)

    temporal_windows_df = agg_metrics_last_year.merge(
        agg_metrics_last_three_months, 
        on="id_curr", 
        how="outer",  
    )

    temporal_windows_df["payment_trend"] = np.where(temporal_windows_df["last_90_instalments_completion_ratio"] != 0 , temporal_windows_df["last_365_instalments_completion_ratio"] / temporal_windows_df["last_90_instalments_completion_ratio"],np.nan)
    temporal_windows_df["delincuency_trend"] = temporal_windows_df["last_365_instalments_days_of_delinquency_mean"] - temporal_windows_df["last_90_instalments_days_of_delinquency_mean"] 
    temporal_windows_df["underpayment_trend"] = temporal_windows_df["last_365_instalments_is_underpayment_mean"] - temporal_windows_df["last_90_instalments_is_underpayment_mean"]

    temporal_windows_df_to_save=pd.DataFrame()
    #temporal_windows_df_to_save["last_365_instalments_days_of_delinquency_mean"]= temporal_windows_df["last_365_instalments_days_of_delinquency_mean"]
    #temporal_windows_df_to_save["delincuency_trend"] = temporal_windows_df["delincuency_trend"]
    temporal_windows_df_to_save["last_365_instalments_days_of_delinquency_sum"]= temporal_windows_df["last_365_instalments_days_of_delinquency_sum"]
    temporal_windows_df_to_save["last_365_instalments_days_of_delinquency_mean"]= temporal_windows_df["last_365_instalments_days_of_delinquency_mean"]
    temporal_windows_df_to_save["last_365_instalments_extra_instalament_mean"]= temporal_windows_df["last_365_instalments_extra_instalament_mean"]
    temporal_windows_df_to_save["last_365_instalments_amt_payment_min"]= temporal_windows_df["last_365_instalments_amt_payment_min"]
    temporal_windows_df_to_save["last_90_instalments_amt_instalment_min"]= temporal_windows_df["last_90_instalments_amt_instalment_min"]
    temporal_windows_df_to_save["last_365_instalments_is_delinquency_mean"]= temporal_windows_df["last_365_instalments_is_delinquency_mean"]
    temporal_windows_df_to_save["last_365_instalments_completion_ratio"]= temporal_windows_df["last_365_instalments_completion_ratio"] 
    temporal_windows_df_to_save["last_90_instalments_completion_ratio"]= temporal_windows_df["last_90_instalments_completion_ratio"] 
    temporal_windows_df_to_save["last_90_instalments_amt_instalment_min"]= temporal_windows_df["last_90_instalments_amt_instalment_min"]
    temporal_windows_df_to_save["payment_trend"]= temporal_windows_df["payment_trend"]


    temporal_windows_df_to_save["id_curr"]= temporal_windows_df["id_curr"]

    output_dir = output_filepath.parent
    base_name = output_filepath.stem

    agg_metrics_df.to_parquet(output_filepath)
    temporal_windows_df_to_save.to_parquet(output_dir/ f"{base_name}-temporal_window.parquet")



def process_previous_application(input_filepath: Path, installments_filepath: Path, credit_card_filepath: Path, POS_CASH_balance_filepath: Path, installments_time_window_filepath: Path, credit_card_time_window_filepath: Path, cash_balance_time_window_filepath: Path, output_filepath: Path):
    previous_application_df = pd.read_parquet(cfg.CLEANS_DIR / input_filepath)
    #creating columns before aggregation
    previous_application_df["diff_application_credit"] = previous_application_df["amt_application"] - previous_application_df["amt_credit"]
    previous_application_df["ratio_credit_to_goods"] = previous_application_df["amt_credit"] / (previous_application_df["amt_goods_price"].replace(0,np.nan))
    previous_application_df["total_interest_charged"] = (previous_application_df["amt_annuity"] * previous_application_df["cnt_payment"]) - previous_application_df["amt_credit"]
    previous_application_df["implied_interest_rate"] = (previous_application_df["amt_annuity"] * previous_application_df["cnt_payment"]) / previous_application_df["amt_credit"].replace(0, np.nan)
    previous_application_df["ratio_credit_to_annuity"]= previous_application_df["amt_credit"] / (previous_application_df["amt_annuity"].replace(0,np.nan))

    print(f"path de installments = {installments_filepath}")
    instalament_df = pd.read_parquet(installments_filepath)
    print(f"path de credit_card = {credit_card_filepath}")
    credit_card_df = pd.read_parquet(credit_card_filepath)
    print(f"path de cash_balance = {POS_CASH_balance_filepath}")
    cash_balance_df = pd.read_parquet(POS_CASH_balance_filepath)

    previous_application_df= previous_application_df.merge(instalament_df,how="left",on= "id_prev")
    previous_application_df= previous_application_df.merge(credit_card_df,how="left",on= "id_prev")
    previous_application_df= previous_application_df.merge(cash_balance_df,how="left",on= "id_prev")


    previous_application_to_pivot_df= previous_application_df.drop(columns="id_prev")
    mask_no_final= previous_application_df["flag_last_application_for_the_contract"] == "N"
    previous_application_to_pivot_df= previous_application_to_pivot_df.loc[~mask_no_final]
    previous_application_to_pivot_df= previous_application_to_pivot_df.drop(columns=["flag_last_application_for_the_contract"])

    previous_application_to_pivot_df.sort_values(["id_curr","days_decision"],inplace=True,ascending=False)
    last_three= previous_application_to_pivot_df.groupby("id_curr").head(1)
    last_three = last_three.copy()
    last_three["loan_order"] = last_three.groupby("id_curr").cumcount() + 1
    df_wide = last_three.pivot(index="id_curr", columns="loan_order")
    df_wide.columns =[f"{col}_prev_{rank}" for col, rank in df_wide.columns]
    df_wide= df_wide.reset_index()
    previous_application_df["code_reject_reason"]= previous_application_df["code_reject_reason"].str.lower()
    previous_application_df = pd.get_dummies(previous_application_df, columns=[ "code_reject_reason"])
    previous_application_df["name_contract_status"]= previous_application_df["name_contract_status"].str.lower()
    previous_application_df = pd.get_dummies(previous_application_df, columns=[ "name_contract_status"])

    #we will calculate aggregations per client for differents tables so we will separate dictionaries per table
    agg_from_prev_app_dict= {

        #saving the ammount of contract
        "id_prev" : ["count"],
        
        #for log transformated we want to catch the mean and the std (avoiding the impact of the heavy tail from this columns)
        "log_amt_credit": ["mean","std"],   
        "log_amt_application": ["mean","std"],
        "log_amt_down_payment": ["mean","std"],
        "log_amt_goods_price": ["mean","std"],
        "log_amt_annuity": ["mean","std"],
        "log_total_interest_charged" : ["mean","std"],
        "instalments_completion_ratio" : ["mean","std"],
    
        

        #for non transformated columns we want to catch the representative values and the acumulated
        "amt_credit": ["max", "min","median","sum"],
        "amt_application": ["max", "min","median","sum"],
        "amt_down_payment": ["max", "min","median","sum"],
        "amt_goods_price": ["max", "min","median","sum"],
        "amt_annuity": ["max", "min","median"],
        "total_interest_charged": ["max", "min","median"],
        "implied_interest_rate" : ["max", "min","mean","std"],
        

        #others_monetary
        "diff_application_credit": ["max","mean","min","sum","median"], #
        "log_diff_application_credit": ["max","mean","min"],
        "rate_down_payment": ["max","mean","min","median","std"], #
        "ratio_credit_to_goods" : ["max","mean","median","min","std"], # 
        "ratio_credit_to_annuity" : ["max","mean","min","median","std"], #
        #categoricals
        #"name_contract_status_approved": ["mean"],#"sum"
        "name_contract_status_canceled": ["mean"],#,"sum"
        "name_contract_status_refused": ["mean"],#,"sum"
        #"name_contract_type_cash loans": ["mean", "sum"], 
        #"name_contract_type_consumer loans": ["mean", "sum"],
        #"name_contract_type_revolving loans": ["mean", "sum"], 
        "amt_annuity_and_cnt_payment_are_missing" :["mean","sum"],
        "amt_down_payment_is_missing" : ["mean","sum"],
        "nflag_insured_on_approval" : ["mean","sum"],
        "days_and_insurance_information_are_missing": ["mean","sum"],
        "amt_goods_price_is_missing" : ["mean","sum"],
        "rate_down_payment_is_missing" : ["mean","sum"],


        #counters
        "days_decision":["mean","min","max"],
        "cnt_payment":["mean","max","sum"]
    }



    agg_from_instalment_payment_dict= {
        "instalments_amount_of_versions_in_sequence" : ["mean","max","sum"],
        "instalments_potentially_on_going" : ["sum"],
        "instalments_dead_tail_length" : ["mean","max"],
        "instalments_amt_instalment_sum" : ["mean","sum","max"],
        "instalments_amt_payment_sum" : ["mean", "sum", "max"],
        "instalments_days_of_delinquency_max": ["max"],
        "instalments_days_of_delinquency_mean": ["mean", "max"],
        "instalments_extra_instalament_sum":["sum"],
        "instalments_extra_instalament_mean":["mean"],
        "instalments_days_in_advance_max":["max"],
        "instalments_days_of_underpayment_max":["max"],
        "instalments_days_in_advance_mean": ["mean", "max"],
        "instalments_is_delinquency_sum" : ["sum"] ,
        "instalments_is_delinquency_mean" : ["mean"],
        "instalments_diff_expected_received_sum" : ["sum"]
    }


    agg_from_credit_card_dict= {
        "credit_card_desesperation_ratio_max" : ["max"],
        "credit_card_desesperation_ratio_min" : ["min"],
        "credit_card_desesperation_ratio_mean"  : ["mean"],
        "credit_card_balance_limit_ratio_min" : ["min"],
        "credit_card_payment_ratio_min" : ["min"],
        "credit_card_payment_ratio_mean" : ["mean"],
        "credit_card_potential_on_going_loan" : ["sum"],
        "credit_card_is_over_the_limit_sum" : ["sum"],
        "credit_card_is_over_the_limit_mean" : ["mean"],
        "credit_card_months_balance_min" : ["min"],
        "credit_card_sk_dpd_max" : ["max"],
        "credit_card_sk_dpd_def_max" : ["max"],
        "credit_card_inconsistency_gap_max" : ["max"],
        "credit_card_cnt_instalment_mature_cum_max" : ["max"],
        "credit_card_cnt_drawings_atm_current_mean": ["mean"],
        "credit_card_amt_balance_mean" : ["mean"],
        "credit_card_amt_balance_max" : ["max"],
    }

    agg_from_cash_balance_dict= {
    #"cash_balance_potentaily_on_going" : ["sum"],
    "cash_balance_diff_expected_real_duration" : ["mean"],
    "cash_balance_sk_dpd_mean"  : ["mean"],
    "cash_balance_original_expected_duration" : ["max"],
    "cash_balance_sk_dpd_def_mean" : ["mean"],
    "cash_balance_factical_duration" : ["mean"],
    "cash_balance_sk_dpd_tecnical_mean": ["mean"],
    "cash_balance_sk_dpd_tecnical_sum": ["sum"],  
    "cash_balance_sk_dpd_severe_mean": ["mean"], 
    "cash_balance_sk_dpd_severe_sum": ["mean"], 
    "cash_balance_dpd_def_tecnical_mean": ["mean"], 
    "cash_balance_dpd_def_tecnical_sum": ["sum"], 
    "cash_balance_dpd_def_severe_mean": ["mean"], 
    "cash_balance_dpd_def_severe_sum": ["sum"], 
    }

    final_dict_for_agg= agg_from_prev_app_dict | agg_from_instalment_payment_dict | agg_from_credit_card_dict | agg_from_cash_balance_dict


    previous_application_df["log_amt_credit"] = np.log1p(previous_application_df["amt_credit"])
    previous_application_df["log_amt_application"] = np.log1p(previous_application_df["amt_application"])
    previous_application_df["log_amt_annuity"] = np.log1p(previous_application_df["amt_annuity"])
    previous_application_df["log_amt_down_payment"] = np.log1p(previous_application_df["amt_down_payment"])
    previous_application_df["log_amt_goods_price"] = np.log1p(previous_application_df["amt_goods_price"])
    previous_application_df["log_diff_application_credit"] = previous_application_df["log_amt_application"] - previous_application_df["log_amt_credit"]
    previous_application_df["total_interest_charged"] = previous_application_df["total_interest_charged"].clip(lower=0)
    previous_application_df["log_total_interest_charged"] = np.log1p(previous_application_df["total_interest_charged"])



    agg_metrics_df= previous_application_df.groupby("id_curr").agg(final_dict_for_agg)

    agg_metrics_df.columns= [f"{col[0]}_{col[1]}" for col in agg_metrics_df.columns]

    agg_metrics_df.rename(columns={"id_prev_count": "applications_count"},inplace=True)
    agg_metrics_df= agg_metrics_df.reset_index()

    agg_metrics_df["global_approval_ratio"] = agg_metrics_df["amt_credit_sum"] / agg_metrics_df["amt_application_sum"].replace(0, np.nan)
    agg_metrics_df["days_decision_spread"] = agg_metrics_df["days_decision_max"] - agg_metrics_df["days_decision_min"]

    previous_application_ready_to_merge= df_wide.merge(agg_metrics_df,on="id_curr",how="left")

    cols_to_fix = [
        "instalments_potentially_on_going_sum",
        "instalments_is_potentially_incomplete_sequence_sum",
        "instalments_is_potentially_incomplete_sequence_mean",
        "instalments_is_delinquency_sum",
        "instalments_is_delinquency_mean"
        "credit_card_potential_on_going_loan_sum"
    ]

    time_window_df= pd.read_parquet(installments_time_window_filepath)
    time_window_credit_card= pd.read_parquet(credit_card_time_window_filepath)
    time_window_cash_balance= pd.read_parquet(cash_balance_time_window_filepath)

    previous_application_ready_to_merge= previous_application_ready_to_merge.merge(time_window_df,how="left",on="id_curr")
    previous_application_ready_to_merge= previous_application_ready_to_merge.merge(time_window_credit_card,how="left",on="id_curr")
    previous_application_ready_to_merge= previous_application_ready_to_merge.merge(time_window_cash_balance,how="left",on="id_curr")

    for col in cols_to_fix:
        if col in previous_application_ready_to_merge.columns:
            previous_application_ready_to_merge[col] = pd.to_numeric(previous_application_ready_to_merge[col], errors='coerce')


    previous_application_ready_to_merge.to_parquet(cfg.PROCESSED_DIR / output_filepath, index=False)

def process_pos_cash_balance(input_filepath: Path, output_filepath: Path):
    cash_balance_df = pd.read_parquet(input_filepath)

    column_order_reference="months_balance"
    cash_balance_df.sort_values(["id_prev",column_order_reference],inplace=True)

    cash_balance_df["raw_size"]= cash_balance_df.groupby("id_prev").transform("size")

    #now, like in all the previous tables, we proceed tu use definition from the EDA.
    #For more details consult the markdown in the second celd of eda/eda_cash_balance.ipynb
    non_nan_cnt_instalment=  cash_balance_df[cash_balance_df["count_instalment"].notna()]
    first_non_nan_cnt_instalment= non_nan_cnt_instalment.groupby("id_prev")["count_instalment"].transform("first")
    cash_balance_df["original_expected_duration"] = cash_balance_df["id_prev"].map(first_non_nan_cnt_instalment)

    completed_status= cash_balance_df[cash_balance_df["name_contract_status"] == "Completed"]
    first_completed_status= completed_status.groupby("id_prev")["count_instalment"].transform("first")
    cash_balance_df["factical_duration"] = cash_balance_df["id_prev"].map(first_completed_status)

    cash_balance_df["nunique_instalment_future"] = cash_balance_df.where((cash_balance_df["potentaily_on_going"] == 0) & (cash_balance_df["incomplete_sequence"]==0)).groupby("id_prev")["count_instalment_future"].transform("nunique")

    cash_balance_df["diff_expected_real_duration"]= cash_balance_df["original_expected_duration"]  - cash_balance_df["factical_duration"]

    cash_balance_df["name_contract_status"]= cash_balance_df["name_contract_status"].str.lower()

    cash_balance_df= pd.get_dummies(cash_balance_df,columns=["name_contract_status"])

    cash_balance_agg= cash_balance_df.groupby("id_prev").agg({

    "potentaily_on_going" : ["first"],
    "incomplete_sequence" : ["first"],
    "raw_size" : ["first"],
    "original_expected_duration": ["first"],
    "factical_duration": ["first"],
    "diff_expected_real_duration" : ["first"],
    "nunique_instalment_future" : ["first"],

    "amount_advanced_payment": ["max", "sum"],
    "count_instalment" : ["max","min"],
    "count_instalment_future" : ["max","min"],
    "months_balance" : ["max","min"],

    #day_past_due
    "sk_dpd": ["mean","sum","max"],
    "sk_dpd_def": ["mean","sum","max"],

    #categoricals
    "sk_dpd_tecnical": ["mean", "sum"], 
    "sk_dpd_severe": ["mean", "sum"], 
    "dpd_def_tecnical": ["mean", "sum"], 
    "dpd_def_severe": ["mean", "sum"], 
    "name_contract_status_active": ["mean", "sum"],
    "name_contract_status_completed": ["mean", "sum"],
    "flag_is_dead_tail" : ["mean", "sum"],
    "flag_delay_tail" : ["mean", "sum"],
})

    cash_balance_agg.columns = [
        f"cash_balance_{col[0]}" if col[1] == "first" else f"cash_balance_{col[0]}_{col[1]}"
        for col in cash_balance_agg.columns
    ]

    def get_time_window(df: pd.DataFrame,time_reference_months) :
        recent_df =df[df["months_balance"] > time_reference_months]
        recent_agg_df=recent_df.groupby("id_curr").agg({
        #"diff_expected_real_duration" : ["mean"],
        "amount_advanced_payment": ["max", "sum"],
        "count_instalment" : ["max","min"],
        "count_instalment_future" : ["max","min"],

        #day_past_due
        "sk_dpd": ["mean","sum","max"],
        "sk_dpd_def": ["mean","sum","max"],

        #categoricals
        "sk_dpd_tecnical": ["mean", "sum"],
        "sk_dpd_severe": ["mean", "sum"],
        "dpd_def_tecnical": ["mean", "sum"],
        "dpd_def_severe": ["mean", "sum"],
        "flag_is_dead_tail" : ["mean", "sum"],
        "flag_delay_tail" : ["mean", "sum"],
    })
        time_reference_months= time_reference_months * -1
        recent_agg_df.columns = [
        f"last_{time_reference_months}_cash_balance_{col[0]}_{col[1]}"
        for col in recent_agg_df.columns
        ]

        return recent_agg_df

    last_6= get_time_window(cash_balance_df,-6)
    last_18= get_time_window(cash_balance_df,-18)

    output_dir = output_filepath.parent
    base_name = output_filepath.stem

    cash_balance_agg.to_parquet(output_filepath)
    last_6.to_parquet(output_dir/ f"{base_name}_time_window.parquet")
    last_18.to_parquet(output_dir / f"{base_name}_last_18.parquet")

def process_credit_card_balance(input_filepath: Path, output_filepath: Path):
    credit_card_df= pd.read_parquet(input_filepath)
    credit_card_df["raw_lenght"]= credit_card_df.groupby("id_prev").transform("size") 
    credit_card_df["next_mature_cum"] = credit_card_df.groupby("id_prev")["cnt_instalment_mature_cum"].shift(-1)
    credit_card_df["month_with_activity"] = (credit_card_df["next_mature_cum"] > 0) & (credit_card_df["amt_balance"] > 0)
    credit_card_df["is_over_the_limit"] = (credit_card_df["amt_credit_limit_actual"] >  credit_card_df["amt_balance"] ).astype(int)
    credit_card_df["balance_limit_ratio"] = np.where((credit_card_df["amt_credit_limit_actual"] != 0) & (credit_card_df["amt_balance"] > 0) ,credit_card_df["amt_balance"]  / credit_card_df["amt_credit_limit_actual"],np.nan)
    credit_card_df["payment_ratio"] = np.where((credit_card_df["amt_payment_total_current"] != 0) & (credit_card_df["amt_inst_min_regularity"] > 0) ,credit_card_df["amt_payment_total_current"]  / credit_card_df["amt_inst_min_regularity"],np.nan)
    credit_card_df["desesperation_ratio"] = np.where((credit_card_df["amt_drawings_current"] != 0) & (credit_card_df["amt_drawings_atm_current"] > 0),credit_card_df["amt_drawings_atm_current"]  / credit_card_df["amt_drawings_current"],np.nan)                                                        
    credit_card_df["name_contract_status"]= credit_card_df["name_contract_status"].str.lower()
    credit_card_df= pd.get_dummies(credit_card_df,columns= ["name_contract_status"])
    credit_card_df["potential_on_going_loan"]= credit_card_df["potential_on_going_loan"].astype(int)

    with_dpd= credit_card_df[(credit_card_df["sk_dpd"] != 0)  & (credit_card_df["sk_dpd_def"])] 
    months_since_dpd= with_dpd.groupby("id_prev")["months_balance"].transform("max")
    credit_card_df["months_since_dpd"]= credit_card_df["id_prev"].map(months_since_dpd)
    credit_card_agg_df= credit_card_df.groupby("id_prev").agg({
        "potential_on_going_loan" : ["first"],
        "incomplete_sequence" : ["first"],
        "closing_month": ["first"],
        "raw_lenght" : ["first"],
        "first_expected_payment_month": ["first"],
        "months_since_dpd": ["first"],

        "amt_balance": ["min","max","mean","std"],
        "balance_limit_ratio": ["min","max","mean","std"],
        "payment_ratio": ["min","max","mean","std"],
        "desesperation_ratio" : ["min","max","mean","std"],
        "amt_negative_balance": ["min","mean","std"],
        "diff_total_receivable_balance": ["min","mean","max","std"],
        "amt_credit_limit_actual": ["min","mean","max","std"],
        "amt_drawings_atm_current": ["mean","max","std","sum"],
        "amt_drawings_other_current": ["mean","max","std","sum"],
        "amt_drawings_pos_current": ["mean","max","std","sum"],
        "amt_inst_min_regularity": ["min","mean","max","std","sum"],
        "inconsistency_gap": ["mean","max","sum"],
        "diff_payment_current_total": ["mean","max","sum"],
        "amt_payment_total_current": ["min","mean","max","std","sum"],
        "amt_recivable_principal": ["mean","max","std","sum"],
        "amt_recivable": ["mean","max","std","sum"],
        "cnt_drawings_current": ["mean","max","sum"],
        "cnt_drawings_atm_current": ["mean","max","sum"],
        "cnt_drawings_other_current": ["mean","max","sum"],
        "cnt_drawings_pos_current": ["mean","max","sum"],
        "cnt_instalment_mature_cum": ["max"],

        "months_balance": ["max", "min"],
    

        #day_past_due
        "sk_dpd": ["mean","sum","max"],
        "sk_dpd_def": ["mean","sum","max"],

        #categoricals
        "sk_dpd_tecnical": ["mean", "sum"],
        "sk_dpd_severe": ["mean", "sum"],
        "sk_dpd_def_tecnical": ["mean", "sum"],
        "sk_dpd_def_severe": ["mean", "sum"],
        "name_contract_status_active": ["mean", "sum"],
        "name_contract_status_completed": ["mean", "sum"],
        "have_negative_balance": ["mean", "sum"],
        "cnt_drawings_are_present": ["mean", "sum"],
        "month_with_activity" : ["mean", "sum"], 
        "name_contract_status_demand" : ["mean", "sum"], 
        "is_over_the_limit" : ["mean", "sum"], 
    })

    credit_card_agg_df.columns = [
        f"credit_card_{col[0]}" if col[1] == "first" else f"credit_card_{col[0]}_{col[1]}"
        for col in credit_card_agg_df.columns
    ]

    def time_window_credit_card(period_in_months, df_credit_card: pd.DataFrame) :
        last_period= df_credit_card [df_credit_card["months_balance"] > period_in_months]
        agg_last_period= last_period.groupby("id_curr").agg({
            "sk_dpd": ["mean","max"],
            "sk_dpd_def": ["mean","max"],
            "is_over_the_limit" : ["mean","sum"], 
            "inconsistency_gap": ["mean","max"],
            "desesperation_ratio" : ["max","mean"],
            "balance_limit_ratio" : ["min","max","mean"],
            "payment_ratio": ["min","mean"],
            "amt_payment_total_current": ["sum"],
            "amt_balance": ["sum","mean"],
        })
        period_in_months= period_in_months * -1

        agg_last_period.columns = [
            f"last_{period_in_months}_credit_card_{col[0]}_{col[1]}"
            for col in agg_last_period.columns
        ]
        agg_last_period= agg_last_period.reset_index()

        agg_last_period[f"last_{period_in_months}_credit_card_completion_ratio"] = np.where( agg_last_period[f"last_{period_in_months}_credit_card_amt_balance_sum"] > 0, agg_last_period[f"last_{period_in_months}_credit_card_amt_payment_total_current_sum"] / agg_last_period[f"last_{period_in_months}_credit_card_amt_balance_sum"], 1.0 ) 
        return agg_last_period

    last_six_months= time_window_credit_card(-6,credit_card_df)

    output_dir = output_filepath.parent
    base_name = output_filepath.stem

    last_six_months.to_parquet(output_dir/ f"{base_name}-last_six_months_agg.parquet")
    credit_card_agg_df.to_parquet(output_filepath)

def process_application_train(input_filepath: Path, output_filepath: Path):
    cleaned_application_train= pd.read_parquet(input_filepath)

    cleaned_application_train["ratio_debt_income"] = (cleaned_application_train["amt_credit"] /  (cleaned_application_train["amt_income_total"]))

    cleaned_application_train["ratio_debt_age"]= cleaned_application_train["amt_credit"] /(cleaned_application_train["days_birth"] * -1) 
    cleaned_application_train["ratio_days_employed_days_lived"]= cleaned_application_train["days_employed"] /(cleaned_application_train["days_birth"] * -1) 
    cleaned_application_train["kui_ratio"] =  np.where(cleaned_application_train["days_employed"] != 0, cleaned_application_train["amt_credit"] / ((cleaned_application_train["days_employed"] * -1) * cleaned_application_train["amt_income_total"]), 0)
    cleaned_application_train["ratio_good_credit"]= cleaned_application_train["amt_goods_price"] / cleaned_application_train["amt_credit"]
    cleaned_application_train["ratio_annuity_income"] = cleaned_application_train["amt_annuity"] / cleaned_application_train["amt_income_total"]

    cleaned_application_train["credit_duration"]= cleaned_application_train["amt_credit"] / cleaned_application_train["amt_annuity"]
    cleaned_application_train["ext_1_x_2"] = cleaned_application_train["ext_source_1"] * cleaned_application_train["ext_source_2"]
    cleaned_application_train["ext_2_x_3"] = cleaned_application_train["ext_source_2"] * cleaned_application_train["ext_source_3"]
    cleaned_application_train["ext_1_x_3"] = cleaned_application_train["ext_source_1"] * cleaned_application_train["ext_source_3"]

    cleaned_application_train = pd.get_dummies(cleaned_application_train,columns=["education_type"])

    ext_cols = ['ext_source_1', 'ext_source_2', 'ext_source_3']

    building_features_names = [
        "APARTMENTS_AVG","BASEMENTAREA_AVG","YEARS_BEGINEXPLUATATION_AVG","YEARS_BUILD_AVG",
        "COMMONAREA_AVG","ELEVATORS_AVG","ENTRANCES_AVG","FLOORSMAX_AVG","FLOORSMIN_AVG",
        "LANDAREA_AVG","LIVINGAPARTMENTS_AVG","LIVINGAREA_AVG","NONLIVINGAPARTMENTS_AVG",
        "NONLIVINGAREA_AVG","APARTMENTS_MODE","BASEMENTAREA_MODE","YEARS_BEGINEXPLUATATION_MODE",
        "YEARS_BUILD_MODE","COMMONAREA_MODE","ELEVATORS_MODE","ENTRANCES_MODE","FLOORSMAX_MODE",
        "FLOORSMIN_MODE","LANDAREA_MODE","LIVINGAPARTMENTS_MODE","LIVINGAREA_MODE",
        "NONLIVINGAPARTMENTS_MODE","NONLIVINGAREA_MODE","APARTMENTS_MEDI","BASEMENTAREA_MEDI",
        "YEARS_BEGINEXPLUATATION_MEDI","YEARS_BUILD_MEDI","COMMONAREA_MEDI","ELEVATORS_MEDI",
        "ENTRANCES_MEDI","FLOORSMAX_MEDI","FLOORSMIN_MEDI","LANDAREA_MEDI","LIVINGAPARTMENTS_MEDI",
        "LIVINGAREA_MEDI","NONLIVINGAPARTMENTS_MEDI","NONLIVINGAREA_MEDI","FONDKAPREMONT_MODE",
        "HOUSETYPE_MODE","TOTALAREA_MODE","WALLSMATERIAL_MODE","EMERGENCYSTATE_MODE"
        ]

    building_features_names = [col.lower() for col in building_features_names]

    categorical_bldg = ['fondkapremont_mode', 'housetype_mode', 'wallsmaterial_mode', 'emergencystate_mode']

    numeric_bldg = [col for col in building_features_names if col not in categorical_bldg]



    # Agregaciones horizontales (axis=1)
    cleaned_application_train["ext_source_mean"] = cleaned_application_train[ext_cols].mean(axis=1)
    cleaned_application_train["ext_source_std"] = cleaned_application_train[ext_cols].std(axis=1)


    # Agregaciones horizontales (axis=1)
    cleaned_application_train["building_score_mean"] = cleaned_application_train[numeric_bldg].mean(axis=1)
    cleaned_application_train["building_score_max"] = cleaned_application_train[numeric_bldg].max(axis=1)
    cleaned_application_train["building_score_min"] = cleaned_application_train[numeric_bldg].min(axis=1)
    cleaned_application_train["building_score_std"] = cleaned_application_train[numeric_bldg].std(axis=1)
    cleaned_application_train["building_score_sum"] = cleaned_application_train[numeric_bldg].sum(axis=1)

    cleaned_application_train["building_features_nan_count"] = cleaned_application_train[building_features_names].isnull().sum(axis=1)

    cleaned_application_train= cleaned_application_train.drop(columns=numeric_bldg)
    cleaned_application_train.to_parquet(output_filepath)
