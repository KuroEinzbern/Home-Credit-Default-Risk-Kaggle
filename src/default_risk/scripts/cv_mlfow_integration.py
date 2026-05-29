import numpy as np
import pandas as pd
from typing import Any
import sklearn as skt
from sklearn.metrics import roc_auc_score
import mlflow
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator


def run_cv_tracked_mlflow(model_class: type[BaseEstimator], model_params: dict[str, Any], cv: BaseCrossValidator, X: pd.DataFrame, Y: pd.Series, experiment_name: str, run_name: str):

    mlflow.set_experiment(experiment_name)
    mlflow.xgboost.autolog(log_models=False, silent=True)
    parent_name= "Parent_" + run_name
    oof_auc_prob=np.zeros(len(Y)) #for pre-alocated memory 
    list_scores=[]
    
    fold=0
    with mlflow.start_run(run_name= parent_name) as parent_run:
        mlflow.log_params("model_hiperparms",model_params)
        mlflow.log_param("n_folds",cv.get_n_splits())
        for train_index, val_index in cv.split(X,Y) : 
            fold = fold +1
            x_train= X.iloc[train_index]
            y_train= Y.iloc[train_index]
            x_val= X.iloc[val_index]
            y_val= Y.iloc[val_index]
            model= model_class(**model_params)
            name_of_nested_run= f"{run_name}_child_{fold:d}"

            with mlflow.start_run(run_name= name_of_nested_run, nested=True) :
                model.fit(x_train,y_train,eval_set =[(x_val,y_val)])
                predictions= model.predict_proba(x_val)
                default_prob= predictions[:,1]
                oof_auc_prob[val_index] = default_prob
                score= roc_auc_score(y_val,default_prob)
                list_scores.append(score)

        mean_auc= np.mean(list_scores)
        standar_deviation= np.std(list_scores)
        mlflow.log_metric("cv_mean_auc",(mean_auc))
        mlflow.log_metric("cv_mean_std",(standar_deviation))
        oof_auc_score = roc_auc_score(Y,oof_auc_prob)
        mlflow.log_metric("oof_auc",oof_auc_score)
        print("=================================================================================")
        print(f"result of CV with {fold:d} folds")
        print(f"mean AUC per fold= {mean_auc:.3f} ± {standar_deviation:.3f}(std)")
        print(f"auc_score_OOF= {oof_auc_score:.3f}")
        print("=================================================================================")
    return