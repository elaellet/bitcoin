import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb

from .models.arima import ARIMAForecaster
from .models.ensemble import StackedEnsembleForecaster
from .models.gru import GRUForecaster
from .models.lstm import LSTMForecaster
from .models.naive import NaiveForecaster
from .models.xxgb import XGBForecaster
from .preprocessing import *
from .training import *
from .utils import print_header

def train_and_forecast_with_naive_model(series_ds, target_col, forecast_horizon,
                                        data_type, time_unit):
    '''
    Trains a naive model on all data, evaluates it on the test set, 
    and generates a final forecast output from NaiveForecaster.

    Args:
        series_ds (pd.DataFrame): The dataset containing the time series data.
        forecast_horizon (int): The number of steps to forecast.
        target_col (str): The name of the column to forecast.
        data_type (str): The type of dataset being used.        
        time_unit (str): The time unit for the forecast horizon.

    Returns:
        tuple[dict, dict]: A tuple containing:
            - A dictionary of evaluation metrics ('mape', 'da') from the
              evaluation on the test set.
            - A dictionary containing the final future forecast ('pred', 'sig').
    '''
    print_header(f'Naive Model Training, Evaluation, and Forecasting on {data_type} Set (Horizon: {forecast_horizon} {time_unit.capitalize()})')

    model = NaiveForecaster(series_ds, target_col)

    model.fit()
    metrics = model.evaluate(forecast_horizon)
    pred_out = model.predict(forecast_horizon)

    return (metrics, pred_out)

def find_best_arima_order(train_ds, valid_ds, target_col, window_size, forecast_horizon,
                          time_unit, p_values, d_values, q_values, naive_metrics):
    '''
    Finds the best ARIMA (p,d,q) order by performing a grid search using walk-forward valiadation
    using a rolling window, fixed parameters, and direct forecasting.

    This function orchestrates the ARIMA hyperparameter tuning pipeline 
    by first defining a search space for ARIMA orders. 
    It then performs a grid search using walk-forward validation on the validation set. 
    Finally, it selects the optimal model order by comparing their performance against a naive baseline.

    Args:
        train_ds (pd.DataFrame): The training dataset.
        valid_ds (pd.DataFrame): The validation dataset.
        target_col (str): The name of the column to forecast.
        window_size (int): The number of past time steps to use as input.
        forecast_horizon (int): The number of steps to forecast.
        time_unit (str): The time unit for the forecast horizon.
        p_values (list): A list of p values to test.
        d_values (list): A list of d values to test.
        q_values (list): A list of q values to test.
        naive_metrics (dict): The metrics from the naive model for comparison.

    Returns:
        tuple[dict, dict, bool]: A tuple containing:
            - The best hyperparameter configuration.
            - The corresponding evaluation metrics.
            - A boolean flag that is True if the model's performance
            outperforms the naive baseline, and False otherwise.
    '''
    print_header(f'ARIMA Model Tuning on Validation Set (Horizon: {forecast_horizon} {time_unit})')

    orders = generate_arima_orders(p_values, d_values, q_values)
    search_results = run_arima_grid_search(
        train_ds, valid_ds, target_col,
        orders, window_size, forecast_horizon
    )

    best_model, beats_baseline = select_best_model(search_results, naive_metrics, 'ARIMA')
    best_metrics = search_results[0]

    return (best_metrics, beats_baseline)

def train_and_forecast_with_best_arima_model(train_ds, valid_ds, test_ds, 
                                             best_order, target_col, window_size, 
                                             forecast_horizon, time_unit):
    '''
    Trains the best ARIMA model on all data, evaluates it on the test set, 
    and generates a final forecast output from ARIMAForecaster.

    Args:
        train_ds (pd.DataFrame): The training dataset.
        valid_ds (pd.DataFrame): The validation dataset.
        test_ds (pd.DataFrame): The test dataset.
        best_order (tuple): The best (p,d,q) order to use for the model.
        target_col (str): The name of the column to forecast.
        window_size (int): The number of past time steps to use as input.
        forecast_horizon (int): The number of steps to forecast.
        time_unit (str): The time unit for the forecast horizon.

    Returns:
        tuple[dict, dict]: A tuple containing:
            - A dictionary of evaluation metrics ('mape', 'da') from the
              evaluation on the test set.
            - A dictionary containing the final future forecast ('pred', 'sig').
    '''
    print_header(f'ARIMA Model Training, Evaluation, and Forecasting on Test Set (Horizon: {forecast_horizon} {time_unit.capitalize()})')

    train_full_ds = pd.concat([train_ds, valid_ds])
    model = ARIMAForecaster(train_full_ds, test_ds, 
                            target_col, window_size,
                            best_order, False)


    model.fit()
    metrics = model.evaluate(forecast_horizon)
    pred_out = model.predict(forecast_horizon)
    
    return (metrics, pred_out)

def find_best_rnn_hps(rnn_type, train_ds, valid_ds,
                      full_cols, target_col, forecast_horizon,
                      input_window, target_window,
                      time_unit, naive_metrics, max_trials=15):
    '''
    Finds the best hyperparameters for an RNN model using a validation set.

    This function orchestrates the RNN hyperparameter tuning pipeline:
    1. Prepares the feature datasets.
    2. Runs Keras Tuner to find the optimal hyperparameters.
    3. Trains a model with the best hyperparameters on the training set.
    4. Evaluates the model on the validation set.
    5. Selects the hyperparameters only if the model outperforms a naive baseline.

    Args:
        rnn_type (str): The type of RNN ('lstm' or 'gru').
        train_ds (pd.DataFrame): The training dataset.
        valid_ds (pd.DataFrame): The validation dataset.
        full_cols (list): All columns to be used as features.
        target_col (str): The name of the column to forecast.
        forecast_horizon (int): The number of steps to forecast.
        input_window (int): The number of past time steps to use as input.
        target_window (int): The number of future time steps to predict.
        time_unit (str): The time unit for the forecast horizon.
        naive_metrics (dict): The metrics from the naive model for comparison.
        max_trials: The number of hyperparameter combinations to test.

    Returns:
        tuple[dict, dict, bool]: A tuple containing:
            - The best hyperparameter configuration.
            - The corresponding evaluation metrics.
            - A boolean flag that is True if the model's performance
            outperforms the naive baseline, and False otherwise.         
    '''
    rnn_type_upper = rnn_type.upper()
    rnn_type_lower = rnn_type.lower()

    print_header(f'{rnn_type_upper} Model Tuning on Validation Set (Horizon: {forecast_horizon} {time_unit})')
        
    if rnn_type_lower == 'lstm':
        tuner_model = LSTMForecaster(train_ds, valid_ds,
                                     full_cols, target_col, 
                                     input_window, target_window)
    else:
        tuner_model = GRUForecaster(train_ds, valid_ds,
                                    full_cols, target_col, 
                                    input_window, target_window)

    tuner_model._prepare_data()
    n_model_cols = tuner_model.n_model_cols

    train_steps = (len(tuner_model.train_ds) - input_window - target_window + 1) // tuner_model.batch_size
    valid_steps = (len(tuner_model.valid_ds) - input_window - target_window + 1) // tuner_model.batch_size

    best_hps = tune_rnn_forecaster(tuner_model.train_set, tuner_model.valid_set, 
                                   rnn_type, n_model_cols, target_window,
                                   train_steps, valid_steps, max_trials)
    
    best_hypermodel = RNNHyperModel(n_model_cols, target_window, rnn_type).build(best_hps)

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(f'{rnn_type_lower}_validation.weights.h5', save_weights_only=True)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_mae', patience=15, restore_best_weights=True)
    lr_scheduler_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_mae', factor=0.5, patience=7)
    
    tuner_model.fit(
        best_hypermodel,
        epochs=100,
        callbacks=[checkpoint_cb, early_stopping_cb, lr_scheduler_cb]
    )
    
    best_metrics = tuner_model.evaluate(forecast_horizon)
    best_model, beats_baseline = select_best_model([best_metrics], naive_metrics, rnn_type)

    return (best_hps, best_metrics, beats_baseline)

def train_and_forecast_with_best_rnn_model(rnn_type, train_ds, valid_ds, test_ds, best_hps,
                                           full_cols, target_col, forecast_horizon,
                                           input_window, target_window, time_unit):
    '''
    Trains the best RNN model on all data, evaluates it on the test set, 
    and generates a final forecast output from LSTMForecaster/GRUForecaster.

    Args:
        rnn_type (str): The type of RNN ('lstm' or 'gru').
        train_ds (pd.DataFrame): The training dataset.
        valid_ds (pd.DataFrame): The validation dataset.
        test_ds (pd.DataFrame): The test dataset.
        best_hps (keras_tuner.HyperParameters): Best hyperparameters from tuning.
        full_cols (list): All columns to be used as features.
        target_col (str): The name of the column to forecast.
        forecast_horizon (int): The number of steps to forecast.
        input_window (int): The number of past time steps to use as input.
        target_window (int): The number of future time steps to predict.
        time_unit (str): The time unit for the forecast horizon.

    Returns:
        tuple[dict, dict]: A tuple containing:
            - A dictionary of evaluation metrics ('mape', 'da') from the
              evaluation on the test set.
            - A dictionary containing the final future forecast ('pred', 'sig').
    '''
    rnn_type_upper = rnn_type.upper()
    rnn_type_lower = rnn_type.lower()
    
    print_header(f'{rnn_type_upper} Model Training, Evaluation, and Forecasting on Test Set (Horizon: {forecast_horizon} {time_unit.capitalize()})')

    test_full_ds = pd.concat([train_ds, valid_ds])

    if rnn_type_lower == 'lstm':
        model = LSTMForecaster(test_full_ds, test_ds,
                               full_cols, target_col,
                               input_window, target_window)
    else:
        model = GRUForecaster(test_full_ds, test_ds,
                              full_cols, target_col,
                              input_window, target_window)
    
    model._prepare_data()
    n_model_cols = model.n_model_cols
    
    hypermodel = RNNHyperModel(n_model_cols, target_window, rnn_type).build(best_hps)
    hypermodel.summary()

    # There's no validation set for callbacks on the full dataset.
    # Train for a fixed number of epochs and save the final model weights.
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(f'{rnn_type_lower}_test.weights.h5', save_weights_only=True)
    epochs = 100

    model.fit(hypermodel, epochs, callbacks=[checkpoint_cb])

    metrics = model.evaluate(forecast_horizon)
    pred_out = model.predict(hypermodel, forecast_horizon)
    
    return (metrics, pred_out)

def find_best_xgb_params(train_ds, valid_ds,
                         full_cols, target_col, forecast_horizon,
                         input_window, target_window,
                         time_unit, naive_metrics, 
                         n_trials=15, early_stopping_rounds=20):
    '''
    Orchestrates the XGBoost tuning and evaluation process.

    1. Runs hyperparameter tuning for a specific forecast horizon.
    2. Trains all forecast models using the best-found parameters.
    3. Evaluates the model at the same horizon that was tuned.
    4. Compares performance against the naive baseline.
    5. Returns the best HPs and metrics if the model is an improvement.
    
    Args:
        forecast_horizon (int): The specific horizon to tune and evaluate.
        train_ds (pd.DataFrame): The training dataset.
        valid_ds (pd.DataFrame): The validation dataset.
        full_cols (list): All columns to be used as features.
        target_col (str): The name of the column to forecast.
        forecast_horizon (int): The number of steps to forecast.
        input_window (int): The number of past time steps to use as input.
        target_window (int): The number of future time steps to predict.        
        time_unit (str): The time unit for display (e.g., 'Day').
        naive_metrics (dict): The metrics dict from the naive model, for comparison.
        n_trials (int): Number of Optuna trials to run.
        early_stopping_rounds (int): Patience for early stopping.

    Returns:
        tuple[dict, dict, bool]: A tuple containing:
            - The best hyperparameter configuration (dict).
            - The corresponding evaluation metrics (dict).
            - A boolean flag that is True if the model's performance
            outperforms the naive baseline, and False otherwise.
    '''
    print_header(f'XGBoost Model Tuning on Validation Set (Horizon: {forecast_horizon} {time_unit})')

    model = XGBForecaster(train_ds, valid_ds, 
                          full_cols, target_col, 
                          input_window, target_window)
    
    best_params = tune_xgb_forecaster(model, forecast_horizon, 
                                      n_trials, early_stopping_rounds)

    model.fit(best_params)

    best_metrics = model.evaluate(forecast_horizon)
    best_model, beats_baseline = select_best_model([best_metrics], naive_metrics, 'xgb')

    return (best_params, best_metrics, beats_baseline)

def train_and_forecast_with_best_xgb_model(train_ds, valid_ds, test_ds,
                                           best_params, full_cols,
                                           target_col, input_window, target_window,
                                           forecast_horizon, time_unit):
    '''
    Trains the best XGBoost model on all data, evaluates it on the test set, 
    and generates a final forecast output from XGBForecaster.

    Args:
        train_ds (pd.DataFrame): The initial training dataset.
        valid_ds (pd.DataFrame): The validation dataset.
        test_ds (pd.DataFrame): The test dataset (used for final evaluation).
        best_params (dict): The dictionary of best hyperparameters
            found from the tuning process.
        full_cols (list): The list of all column names in the data.
        target_col (str): The name of the column to forecast.
        input_window (int): The number of past time steps to use as input.
        target_window (int): The total number of horizons the models were
            trained for (e.g., 9, for 1-9 days).
        forecast_horizon (int): The specific horizon to evaluate and
            predict (e.g., 8, for the 8-day-ahead forecast).
        time_unit (str): The time unit for the forecast horizon (e.g., 'Day').

    Returns:
        tuple[dict, dict]: A tuple containing:
            - A dictionary of evaluation metrics ('mape', 'da') from the
              evaluation on the test set.
            - A dictionary containing the final future forecast ('pred', 'sig').
    '''
    print_header(f'XGBoost Model Training, Evaluation, and Forecasting on Test Set (Horizon: {forecast_horizon} {time_unit.capitalize()})')

    test_full_ds = pd.concat([train_ds, valid_ds])

    model = XGBForecaster(test_full_ds, test_ds, 
                          full_cols, target_col, 
                          input_window, target_window)
    
    model.fit(best_params)

    metrics = model.evaluate(forecast_horizon)
    pred_out = model.predict(forecast_horizon)

    return (metrics, pred_out)

def train_and_forecast_with_stacked_ensemble_model(base_models, meta_model,
                                                   test_ds, base_model_params_fit,
                                                   base_model_params_eval_and_pred,
                                                   forecast_horizon, time_unit):
    '''
    Trains the stacked ensemble on all data, evaluates it on the test set,
    and generates a final forecast output from StackedEnsembleForecaster.

    This function coordinates the entire stacking workflow:
    1.  Trains the meta-model using the base forecasters' validation predictions.
    2.  Re-trains base models on (train + valid) data to evaluate the
        stack on the test set.
    3.  Re-trains base models on (train + valid + test) data to make
        a final future prediction.

    Args:
        base_models (list): A list of pre-initialized base forecaster
            instances (e.g., [gru_model, xgb_model]). These instances must
            already contain their respective train_ds and valid_ds.
        meta_model (sklearn.base.RegressorMixin): An un-fitted scikit-learn
            regressor (e.g., LinearRegression()) to use as the meta-model.
        test_ds (pd.DataFrame): The test dataset (used for final evaluation).
        base_model_params_fit (dict): A nested dictionary containing the arguments
            for each base forecaster's .fit() method.
            Example:
            {
                'forecaster_gru': {'model': gru_hypermodel, 'epochs': 500,
                                   'callbacks': [early_stopping_cb], ...},
                'forecaster_xgb': {'params': xgb_params}
            } 
        base_model_params_eval_and_pred (dict): A nested dictionary containing the
            arguments for the base forecasters' .evaluate() and .predict()
            methods during the evaluation and prediction phases. This dictionary
            should not contain validation-based callbacks.
            Example:
            {
                'forecaster_gru': {'model': gru_hypermodel, 'epochs': 100,
                                   'callbacks': [checkpoint_cb], ...},
                'forecaster_xgb': {'params': xgb_params}
            }            
        forecast_horizon (int): The specific horizon to evaluate and
            predict (e.g., 8, for the 8-day-ahead forecast).
        time_unit (str): The time unit for the forecast horizon.

    Returns:
        tuple[dict, dict]: A tuple containing:
            - A dictionary of evaluation metrics ('mape', 'da') from the
              evaluation on the test set.
            - A dictionary containing the final future forecast ('pred', 'sig').
    '''
    print_header(f'Stacked Ensemble Model Training, Evaluation, and Forecasting (Horizon: {forecast_horizon} {time_unit.capitalize()})')

    stacked_model = StackedEnsembleForecaster(base_models, meta_model)
    stacked_model.fit(forecast_horizon, base_model_params_fit)
    
    metrics = stacked_model.evaluate(test_ds, forecast_horizon, 
                                     base_model_params_eval_and_pred, time_unit)

    pred_out = stacked_model.predict(forecast_horizon, base_model_params_eval_and_pred)
    
    return (metrics, pred_out)