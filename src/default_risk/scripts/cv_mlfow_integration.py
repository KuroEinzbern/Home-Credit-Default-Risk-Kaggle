import numpy as np
import pandas as pd
from typing import Any
import sklearn as skt
from sklearn.metrics import roc_auc_score
import mlflow
from sklearn.base import BaseEstimator
import mlflow.sklearn
from sklearn.model_selection import BaseCrossValidator
from mlflow.models import infer_signature
from sklearn.base import clone
from default_risk.config import ARTIFACTS_DIR
from sklearn.pipeline import Pipeline
from functools import reduce
from sklearn.inspection import permutation_importance
import default_risk.config as cfg
import logging
from functools import singledispatch
import lightgbm as lgb
import xgboost as xgb


logging.getLogger("mlflow").setLevel(logging.ERROR)

def run_cv_tracked_mlflow(model: BaseEstimator | Pipeline, model_params: dict[str, Any], cv: BaseCrossValidator, X: pd.DataFrame, Y: pd.Series, experiment_name: str, run_name: str,persist_feature_importance: bool =True,save_final_model: bool =True,enable_models_autlog: bool=False, enable_feature_permutation: bool=False,silent: bool=False) -> None:
    mlflow.set_experiment(experiment_name)
    mlflow.xgboost.autolog(log_models=enable_models_autlog, silent=silent)
    mlflow.lightgbm.autolog(disable=True)
    mlflow.sklearn.autolog(disable=True)
    parent_name= "Parent_" + run_name
    oof_auc_prob=np.zeros(len(Y)) #for pre-alocated memory 
    list_scores=[]
    pfi_folds_list = []
    
    fold=0
    with mlflow.start_run(run_name= parent_name) as parent_run:
        mlflow.log_params(model_params)
        mlflow.log_param("n_folds",cv.get_n_splits())
        original_model = model
        for train_index, val_index in cv.split(X,Y) : 
            fold = fold +1
            x_train = X.iloc[train_index]
            y_train = Y.iloc[train_index]
            x_val = X.iloc[val_index]
            y_val = Y.iloc[val_index]
            model= clone(original_model)
            #model= model_class(**model_params)
            name_of_nested_run= f"{run_name}_child_{fold:d}"

            with mlflow.start_run(run_name= name_of_nested_run, nested=True) :
                model= train_the_model(model,x_train,y_train,x_val,y_val)
                #model.fit(x_train,y_train,eval_set =[(x_val,y_val)], verbose=False)
                predictions= model.predict_proba(x_val)
                default_prob= predictions[:,1]
                oof_auc_prob[val_index] = default_prob
                score= roc_auc_score(y_val,default_prob)
                list_scores.append(score)
                

                #feature_importance per fold
                if(persist_feature_importance):
                    save_feature_importance(model,x_train,run_name)

                #feature_permutation per fold
                if enable_feature_permutation:
                    print(f"Calculando PFI para el Fold {fold}...")
                    res_pfi = permutation_importance(model, x_val, y_val, scoring='roc_auc', n_repeats=5, n_jobs=-1, random_state=42)
                    column_name_importance= f'importance_fold_{fold}'
                    pfi_df = pd.DataFrame({'feature': X.columns, column_name_importance: res_pfi.importances_mean})
                    pfi_folds_list.append(pfi_df)

        mean_auc= np.mean(list_scores)
        standar_deviation= np.std(list_scores)
        mlflow.log_metric("cv_mean_auc",(mean_auc))
        mlflow.log_metric("cv_mean_std",(standar_deviation))
        oof_auc_score = roc_auc_score(Y,oof_auc_prob)
        mlflow.log_metric("oof_auc",oof_auc_score)

        #create the csv and artifact for ml_flow.
        if enable_feature_permutation :
            persist_and_export_feature_permutation(pfi_folds_list,run_name)


        print(f"AUC per fold= {mean_auc:.3f} ± {standar_deviation:.3f}(std), auc_score_OOF= {oof_auc_score:.3f} result of CV with {fold:d} folds. ")
        if(save_final_model) :
            final_model= clone(original_model)
            final_model.fit(X,Y)
            input_sample = X.head(5)
            predict_input_sample= final_model.predict(input_sample)
            signature= infer_signature(input_sample,predict_input_sample)
            mlflow.sklearn.log_model(final_model, "final_model",signature=signature)
        
            save_feature_importance(final_model,X,run_name)
                

    
    return oof_auc_score, standar_deviation

def persist_and_export_feature_permutation(pfi_folds_list: list, run_name : str)-> None:

    #in this point we have a DF per folder so we consume each at time to merge it into one.
    pfi_global = reduce(lambda left, right: pd.merge(left, right, on=['feature'], how='outer'), pfi_folds_list)
    pfi_global['mean_importance_cv'] = pfi_global.filter(like='importance_fold_').mean(axis=1)
    pfi_global = pfi_global.sort_values(by='mean_importance_cv', ascending=True)
    pfi_global.to_csv(cfg.ARTIFACTS_DIR / f"pfi_cv_results_{run_name}.csv", index=False)
    mlflow.log_artifact(cfg.ARTIFACTS_DIR / f"pfi_cv_results_{run_name}.csv")
    return

def persist_and_export_feature_importance(feature_importance_df: pd.DataFrame, run_name: str)-> None:
    file_path= str(ARTIFACTS_DIR / f"{run_name}_feature_importance.csv")
    with open(file_path, "w", encoding="utf-8") as f:
        feature_importance_df.to_csv(f,index=False)
    mlflow.log_artifact(file_path)
    return


def save_feature_importance(trained_model, X: pd.DataFrame, run_name: str) -> None:
    df_feature_importance=pd.DataFrame()
    #features_names= X.columns
    trained_model= extract_final_model(trained_model)
    features_names_raw = get_the_features_name(X,train_the_model)
    features_names_clean = [col.replace('remainder__', '').replace('target_encode_cat__', '') for col in features_names_raw]
    importances= trained_model.feature_importances_
    df_feature_importance["feature_name"] = features_names_clean
    df_feature_importance["importances"] = importances
    df_feature_importance.sort_values(by="importances",ascending=True,inplace=True)
    persist_and_export_feature_importance(df_feature_importance,run_name)
    return



@singledispatch
def extract_final_model(model) -> BaseEstimator: 
    print("unknow type in the model")
    return model


@extract_final_model.register
def _(model: Pipeline) :
    return model[-1]


@extract_final_model.register
def _(model: BaseEstimator) :
    return model




@singledispatch
def get_the_features_name(X, model) -> BaseEstimator: 
    print("unknow type in the model")
    return X.columns


@get_the_features_name.register
def _(X, model: lgb.LGBMClassifier) :
    return model.booster_.feature_name()


@get_the_features_name.register
def _(X, model: xgb.XGBClassifier) :
    return model.feature_names_in_




@singledispatch
def train_the_model(model,x_train,y_train,x_val,y_val) : 
    print("unknow type in the model")
    return model


@train_the_model.register
def _(model: Pipeline,x_train,y_train,x_val,y_val) :
    final_step_name = model.steps[-1][0]
    preprocessor = model[:-1] 
    
    preprocessor.fit(x_train, y_train)
    x_val_transformed = preprocessor.transform(x_val)
    
    fit_kwargs = {
        f"{final_step_name}__eval_set": [(x_val_transformed, y_val)],
       # f"{final_step_name}__verbose": False
    }
    model.fit(x_train, y_train, **fit_kwargs)
    return model


@train_the_model.register
def _(model: BaseEstimator,x_train,y_train,x_val,y_val) :
    model.fit(x_train,y_train,eval_set =[(x_val,y_val)], verbose=False)
    return model


@train_the_model.register(lgb.LGBMClassifier)
def _(model: lgb.LGBMClassifier, x_train, y_train, x_val, y_val):
    model.fit(
        x_train, 
        y_train,
        eval_set=[(x_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
    return model