
from pathlib import Path

import pandas as pd
import numpy as np
import default_risk.config as cfg
from default_risk.config import CLEANS_DIR, PROCESSED_DIR
from default_risk.scripts.auxiliar_eda_function import recreate_and_sort_series_given_rows, recreate_and_sort_the_serie_given_ids


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

def process_bureau(input_filepath: Path, output_filepath: Path):
    bureau_df = pd.read_parquet(input_filepath)

    bureau_balance_agg= pd.read_parquet(cfg.PROCESSED_DIR / "bureu_balance_agg.parquet")
    bureau_df=bureau_df.merge(bureau_balance_agg,how="left",on="id_bureau")
    bureau_df['has_bureau_balance_data'] = bureau_df['balance_months_balance_min'].notna().astype(int)
    bureau_balance_agg.head()

    bureau_df["ratio_credit_annuity"]= np.where(bureau_df["amt_annuity"] !=0, bureau_df["amt_credit_sum"] / bureau_df["amt_annuity"] , np.nan )  
    bureau_df["completetitud_ratio"] = np.where(bureau_df["amt_credit_sum_debt"]!=0,bureau_df["amt_credit_sum"] / bureau_df["amt_credit_sum_debt"],np.nan)
    bureau_df["ratio_debt_limit"] =  np.where(bureau_df["amt_credit_sum"] !=0, bureau_df["amt_credit_sum_limit"] / bureau_df["amt_credit_sum"], np.nan )  

    bureau_df["credit_active"]=bureau_df["credit_active"].str.lower()
    bureau_df= pd.get_dummies(bureau_df, columns=["credit_active"], dtype=int)

    bureau_df.sort_values(["id_curr", "days_credit"],inplace=True,ascending=False)
    last_two = bureau_df.groupby("id_curr").head(2)
    last_two = last_two.copy()
    last_two["loan_order"] = last_two.groupby("id_curr").cumcount() + 1

    last_two_columns = last_two.drop(columns=["id_bureau"]).pivot(index="id_curr", columns="loan_order")
    last_two_columns.columns = [f"bureau_{col[0]}_loan_{col[1]}" for col in last_two_columns.columns]
    last_two_columns= last_two_columns.reset_index()

    bureau_df["log_amt_credit_sum"] = np.log1p(bureau_df["amt_credit_sum"])

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
        "days_credit_update": ["min","max","mean"], 
        "days_credit": ["min","max","mean"], 
        "days_enddate_fact": ["max"], 

        "ratio_credit_annuity" : ["max","mean","min"],
        "completetitud_ratio" : ["mean","min"],
        "credit_active_sold" : ["mean","sum"],
        "amt_annuity_is_missing" : ["mean","sum"],
        "amt_credit_max_overdue" :["max","mean","sum"],
        "amt_credit_max_overdue_is_missing" :["mean","sum"],
    }

    agg_from_bureau_balance_dict = {
        "balance_status_score_max": ["max"],
        "balance_months_balance_max": ["max"],
        "balance_months_balance_min": ["min"],
        "balance_months_since_delincuency" : ["max"],
        "balance_is_delincuency_sum" : ["max"],
        "balance_is_delincuency_mean" : ["mean"],
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
    bureau_final_df.to_parquet(output_filepath)

def process_installments_payments(input_filepath: Path, output_filepath: Path):
    installments_payment_df = pd.read_parquet(input_filepath)
    column_order_reference="days_instalment"
    installments_payment_df.sort_values(["id_prev",column_order_reference,"days_instalment"],inplace=True)

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

    installments_payment_df["diff_expected_received"]= installments_payment_df["amt_instalment"] -  installments_payment_df["amt_payment"]
    installments_payment_df["diff_deadline_factical_payment"]= installments_payment_df["days_instalment"] -  installments_payment_df["days_entry_payment"]
    installments_payment_df["days_of_delinquency"]= installments_payment_df["diff_deadline_factical_payment"].clip(lower=0)
    installments_payment_df["is_delinquency"] = installments_payment_df["days_of_delinquency"] > 0 

    next_installment_number = installments_payment_df.groupby("id_prev")["num_instalment_number"].shift(-1)
    next_version_number = installments_payment_df.groupby("id_prev")["num_instalment_version"].shift(-1)
    repeated_installment_mask= (installments_payment_df ["num_instalment_number"] == next_installment_number)
    underpayment_mask= (installments_payment_df[ "amt_payment" ] < installments_payment_df ["amt_instalment"]) & (installments_payment_df[ "amt_payment" ] == 0)
    full_payment_mask= installments_payment_df[ "amt_payment" ] == installments_payment_df ["amt_instalment"] 
    installments_payment_df[ "repeated_for_underpayment" ] = (repeated_installment_mask) & (underpayment_mask)
    installments_payment_df[ "repeated_for_reschedule" ] = (repeated_installment_mask) & (full_payment_mask)
    installments_payment_df["repeated_for_payment_in_advance"] = (repeated_installment_mask) & (installments_payment_df[ "amt_payment" ] == 0) & (installments_payment_df["days_of_delinquency"] == 0)
    installments_payment_df["log_amt_instalment"]= np.log1p(installments_payment_df ["amt_instalment"] )
    installments_payment_df["log_amt_payment"]= np.log1p(installments_payment_df ["amt_payment"] )

    agg_metrics_df= installments_payment_df.groupby("id_prev").agg({

        #static values calculated for the entire squenece
        "raw_size_serie" : ["first"],
        "dead_tail_length" : ["first"],
        "amount_of_versions_in_sequence" : ["first"],
        "is_potentially_incomplete_sequence" : ["first"],
        "potentially_on_going" : ["first"],

        #for log transformated we want to catch the mean and the std (avoiding the impact of the heavy tail from this columns)
        "log_amt_instalment": ["mean","std"],   
        "log_amt_payment": ["mean","std"],

        #natural scale
        "amt_instalment": ["max", "min","median","sum"],
        "amt_payment": ["max", "min","median","sum"],

        #computed_differences
        "diff_expected_received": ["max", "min","median","sum"],
        "diff_deadline_factical_payment": ["max", "min","median","sum"],


        #categoricals
        "repeated_for_underpayment": ["mean","sum"],
        "repeated_for_reschedule": ["mean","sum"],
        "repeated_for_payment_in_advance": ["mean","sum"],
        "is_delinquency" : ["mean","sum"],


        #counters
        "days_of_delinquency":["mean","max","sum"],
    })

    agg_metrics_df.columns = [
        f"instalments_{col[0]}" if col[1] == "first" else f"instalments_{col[0]}_{col[1]}"
        for col in agg_metrics_df.columns
    ]

    agg_metrics_df = agg_metrics_df.reset_index()

    agg_metrics_df["instalments_completion_ratio"] = np.where( agg_metrics_df["instalments_amt_instalment_sum"] > 0, agg_metrics_df["instalments_amt_payment_sum"] / agg_metrics_df["instalments_amt_instalment_sum"], 1.0 )  #if the debt is 0 or negative we assume competitud (1)                                                               )
    agg_metrics_df.to_parquet(output_filepath)

def process_previous_application(input_filepath: Path, output_filepath: Path):
    previous_application_df = pd.read_parquet(input_filepath)
    
    #creating columns before aggregation
    previous_application_df["diff_application_credit"] = previous_application_df["amt_application"] - previous_application_df["amt_credit"]
    previous_application_df["ratio_credit_to_goods"] = previous_application_df["amt_credit"] / (previous_application_df["amt_goods_price"].replace(0,np.nan))
    previous_application_df["total_interest_charged"] = (previous_application_df["amt_annuity"] * previous_application_df["cnt_payment"]) - previous_application_df["amt_credit"]
    previous_application_df["implied_interest_rate"] = (previous_application_df["amt_annuity"] * previous_application_df["cnt_payment"]) / previous_application_df["amt_credit"].replace(0, np.nan)
    previous_application_df["ratio_credit_to_annuity"]= previous_application_df["amt_credit"] / (previous_application_df["amt_annuity"].replace(0,np.nan))

    instalament_df = pd.read_parquet(cfg.PROCESSED_DIR / "installments_payments.train-processed-2.parquet")

    previous_application_df= previous_application_df.merge(instalament_df,how="left",on= "id_prev")
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
    #previous_application_df["name_contract_status"]= previous_application_df["name_contract_status"].str.lower()
    previous_application_df["code_reject_reason"]= previous_application_df["code_reject_reason"].str.lower()
    #previous_application_df["name_contract_type"] = previous_application_df["name_contract_type"].str.lower()
    previous_application_df = pd.get_dummies(previous_application_df, columns=[ "code_reject_reason"])

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

    final_dict_for_agg= agg_from_prev_app_dict | agg_from_instalment_payment_dict


    previous_application_df["log_amt_credit"] = np.log1p(previous_application_df["amt_credit"])
    previous_application_df["log_amt_application"] = np.log1p(previous_application_df["amt_application"])
    previous_application_df["log_amt_annuity"] = np.log1p(previous_application_df["amt_annuity"])
    previous_application_df["log_amt_down_payment"] = np.log1p(previous_application_df["amt_down_payment"])
    previous_application_df["log_amt_goods_price"] = np.log1p(previous_application_df["amt_goods_price"])
    previous_application_df["log_diff_application_credit"] = previous_application_df["log_amt_application"] - previous_application_df["log_amt_credit"]
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
    ]

    time_window_df= pd.read_parquet(cfg.PROCESSED_DIR / "time_window_instalments.parquet")


    previous_application_ready_to_merge= previous_application_ready_to_merge.merge(time_window_df,how="left",on="id_curr")

    for col in cols_to_fix:
        if col in previous_application_ready_to_merge.columns:
            previous_application_ready_to_merge[col] = pd.to_numeric(previous_application_ready_to_merge[col], errors='coerce')

    previous_application_ready_to_merge.to_parquet(output_filepath, index=False)