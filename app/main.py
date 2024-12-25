import warnings

import optuna
import mlflow

import pandas as pd
from sklearn.metrics import accuracy_score

from catboost import CatBoostClassifier

from model.train import train_model
from preprocessing.data_preprocessing import prepare_data
from utils.utils import plot_feature_importance, validate_model, plot_roc_curve

warnings.filterwarnings('ignore')

import os
import sys

RANDOM_STATE = 42
TEST_SIZE = 0.3
DATA_PATH = "data/data_wo_processing.csv"

EXPERIMENT_NAME = 'first_experiment'
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)

def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 50, 500),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'random_state': RANDOM_STATE,
        'verbose': 0
    }

    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=20, verbose=0)

    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    mlflow.log_params(params)
    mlflow.log_metric("accuracy", accuracy)

    return accuracy

if __name__ == '__main__':
    data = pd.read_csv(DATA_PATH)
    X_train, X_test, y_train, y_test = prepare_data(data, TEST_SIZE, RANDOM_STATE)
    print(TEST_SIZE)

    study = optuna.create_study(direction='maximize')

    with mlflow.start_run():

        study.optimize(objective, n_trials=10)

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_accuracy", study.best_value)

        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("iterations", 100)
        mlflow.log_param("depth", 6)

        dataset = mlflow.data.from_pandas(
            X_train
        ) 
        mlflow.log_input(dataset, context='heart-disease')

        model = train_model(X_train, y_train)
        print(model.best_score_)

        input_example = X_test.sample(5)
        mlflow.catboost.log_model(model, "model", input_example=input_example)

        accuracy = validate_model(model, X_test, y_test)
        print(accuracy)
        mlflow.log_metric("accuracy_score", accuracy)

        if 'MultiClass' in model.best_score_['learn']:
            mlflow.log_metric("best_score", model.best_score_['learn']['MultiClass'])
        elif 'Logloss' in model.best_score_['learn']:
            mlflow.log_metric("best_score", model.best_score_['learn']['Logloss'])
        elif 'AUC' in model.best_score_['learn']:
            mlflow.log_metric("best_score", model.best_score_['learn']['AUC'])
        else:
            print("Unknown metric in model.best_score_['learn']")

        fig = plot_feature_importance(model)
        print(fig)
        mlflow.log_figure(fig, "feature_importance.png")

        fig2 = plot_roc_curve(model, X_test, y_test)
        print(fig2)
        mlflow.log_figure(fig2, "roc_curve.png")
    
    optuna.visualization.plot_parallel_coordinate(study).show()
    optuna.visualization.plot_optimization_history(study).show()