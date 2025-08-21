import numpy as np
import pandas as pd
import tensorflow as tf

from .models.arima import ARIMAForecaster
from .models.lstm import LSTMForecaster
from .models.naive import NaiveForecaster
from .training import *
from .utils import print_header

def run_naive_model(X_series, target_col, forecast_horizon,
                    time_unit, data_type):
    '''
    Initializes, evaluates, and gets a final prediction from a NaiveForecaster.

    This function first evaluates the naive model's historical performance on the
    provided series (calculating MAPE and DA). It then generates a single final
    forecast for the future based on the last available value.

    Args:
        X_series (pd.DataFrame): The DataFrame containing the time series data.
        forecast_horizon (int): The number of periods for historical evaluation.
        target_col (str): The name of the column to forecast.
        time_unit (str): The time unit for the forecast horizon (e.g., 'Days').
        data_type (str): The type of dataset being used (e.g., 'Validation').

    Returns:
        tuple[dict, float]: A tuple containing:
            - A dictionary of evaluation metrics ('mape', 'da').
            - A dictionary of float values for future predictions ('one_month', 'one_year').
    '''
    print_header(f'Naive Model Evaluation on {data_type} Set (Horizon: {forecast_horizon} {time_unit.capitalize()})')

    model = NaiveForecaster(X_series, target_col)
    # The 'fit' method does nothing but is called for interface consistency.
    model.fit()
    metrics = model.evaluate(forecast_horizon)
    y_pred = model.predict(forecast_horizon)

    return metrics, y_pred

def run_arima_model(X_train, X_valid, X_test, 
                    target_col, forecast_horizon,
                    time_unit, data_type,
                    p_values, d_values, q_values, 
                    refit_interval, naive_metrics):
    '''
    High-level function to find, evaluate, and get predictions from the best ARIMA model.

    This function orchestrates the entire ARIMA pipeline:
    1. Defines a search space for ARIMA orders.
    2. Runs a grid search using walk-forward validation on the validation set.
    3. Selects the best model based on performance against a naive baseline.
    4. Fits the best model on all available data and makes final future predictions.

    Args:
        X_train: The training dataset.
        X_valid: The validation dataset.
        X_test: The test dataset.
        forecast_horizon (int): The number of steps to forecast ahead.
        target_col (str): The name of the column to forecast.
        time_unit (str): The time unit for the forecast horizon (e.g., 'days', 'weeks').
        data_type (str): The type of data set being used (e.g., 'Validation', 'Test').
        p_values (int): A list of p values to test.
        d_values (int): A list of d values to test.
        q_values (int): A list of q values to test.
        refit_interval (int): How often to re-fit the ARIMA model during evaluation.
        naive_metrics (dict): The results from the naive model for comparison.

    Returns:
        tuple[dict, dict]: A tuple containing:
            - The results dictionary of the best performing ARIMA model.
            - A dictionary of final predictions ('one_month', 'one_year').
    '''
    print_header(f'ARIMA Model Evaluation on {data_type} Set (Horizon: {forecast_horizon} {time_unit})')

    orders = generate_arima_orders(p_values, d_values, q_values)
    search_results = run_arima_grid_search(
        X_train, X_valid, target_col,
        orders, forecast_horizon, refit_interval
    )
    best_model = select_best_model(search_results, naive_metrics, 'arima')

    metrics = None
    y_pred = None

    if best_model:
        best_order = best_model['order']

        X_full_train = pd.concat([X_train, X_valid])
        model = ARIMAForecaster(X_full_train, X_test, target_col, best_order, False)

        print('\nFitting, evaluating, and predicting ARIMA model with the best order...')

        model.fit()
        # Setting `refit_interval` to a value greater than 1 significantly speeds up the full grid search process.
        # However, this can lead to a 'stale' model, resulting in less accurate performance.
        # A `refit_interval` of 1 ensures the model is evaluated with the best order and produces a more reliable performance metric.
        metrics = model.evaluate(forecast_horizon, 1)
        y_pred = model.predict(forecast_horizon)
    
    return metrics, y_pred

def run_optimized_arima_model(X_train, X_valid, X_test,
                              target_col, best_order, 
                              forecast_horizon, time_unit):
    '''Runs the pre-tuned, optimal ARIMA model.'''
    if best_order == None:
        print_header(f'No Optimized ARIMA Model (Horizon: {forecast_horizon} {time_unit})')
        return None

    print_header(f'Running Optimized ARIMA{best_order} Model (Horizon: {forecast_horizon} {time_unit})')
    
    X_full_train = pd.concat([X_train, X_valid])
    model = ARIMAForecaster(X_full_train, X_test, target_col, best_order, False)

    model.fit()
    metrics = model.evaluate(forecast_horizon, 1)
    y_pred = model.predict(forecast_horizon)

    return metrics, y_pred

def _run_rnn_pipeline(rnn_type, X_train, X_valid, X_test,
                      feature_cols, target_col, forecast_horizon,
                      input_window, target_window,
                      time_unit, data_type,
                      naive_metrics, strategy):
    
    rnn_type_upper = rnn_type.upper()
    print_header(f'{rnn_type_upper} Model Evaluation on {data_type} Set (Horizon: {forecast_horizon} {time_unit})')

    tuner_model = LSTMForecaster(X_train, X_valid,
                                 feature_cols, target_col, 
                                 input_window, target_window, strategy=strategy)
    tuner_model._prepare_data()

    n_features = len(feature_cols)

    if strategy == 'direct':
        train_steps = (len(tuner_model.X_train_processed) - input_window - target_window + 1) // tuner_model.batch_size
        valid_steps = (len(tuner_model.X_valid_processed) - input_window - target_window + 1) // tuner_model.batch_size
    else:
        train_steps = (len(tuner_model.X_train_processed) - input_window) // tuner_model.batch_size
        # The original code used integer division (//), which resulted in valid_steps = 0 for a small validation set, causing the training to hang.
        # This fix uses ceiling division to ensure valid_steps is at least 1 if there are any validation samples.
        valid_samples = len(tuner_model.X_valid_processed) - input_window
        if valid_samples > 0:
            valid_steps = int(np.ceil(valid_samples / tuner_model.batch_size))
        else:
            valid_steps = 0

    best_hps = tune_rnn_forecaster(tuner_model.train_set, tuner_model.valid_set, 
                                   'lstm', n_features, target_window,
                                   train_steps, valid_steps)
    
    tuner_hypermodel = RNNHyperModel(n_features, target_window, rnn_type, strategy).build(best_hps)

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('lstm_checkpoints.weights.h5', save_weights_only=True)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_mae', patience=15, restore_best_weights=True
    )
    lr_scheduler_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_mae', factor=0.5, patience=7)
    callbacks = [checkpoint_cb, early_stopping_cb, lr_scheduler_cb]
    epochs = 100

    tuner_model.fit(
        tuner_hypermodel,
        epochs,
        callbacks
    )
    tuning_results = tuner_model.evaluate(forecast_horizon)

    best_model = select_best_model([tuning_results], naive_metrics, rnn_type)

    metrics = None
    y_pred = None
    
    if best_model:
        X_full_train = pd.concat([X_train, X_valid])
        model = LSTMForecaster(X_full_train, X_test,
                               feature_cols, target_col,
                               input_window, target_window, strategy=strategy)
        
        print(f'\nFitting, evaluating, and predicting {rnn_type_upper} model with the best hyperparameters...')

        hypermodel = RNNHyperModel(n_features, target_window, rnn_type, strategy).build(best_hps)
        hypermodel.summary()

        model.fit(hypermodel, epochs, callbacks)

        metrics = model.evaluate(forecast_horizon)
        y_pred = model.predict(model.model, forecast_horizon)
    
    return metrics, y_pred

def run_lstm_model(X_train, X_valid, X_test,
                   feature_cols, target_col, forecast_horizon,
                   input_window, target_window,
                   time_unit, data_type, 
                   naive_metrics, strategy):
    '''High-level function to run the complete LSTM forecasting pipeline.'''
    return _run_rnn_pipeline('lstm', X_train, X_valid, X_test, 
                             feature_cols, target_col, forecast_horizon,
                             input_window, target_window, 
                             time_unit, data_type, 
                             naive_metrics, strategy)


def run_optimized_lstm_model(X_train, X_valid, X_test,
                             input_window, target_window, forecast_horizon,
                             time_unit, best_hps, strategy):
    '''Runs the pre-tuned, optimal LSTM model.'''
    print_header(f'Running Optimized LSTM Model (Horizon: {forecast_horizon} {time_unit})')

    FEATURE_COLS = ['open', 'high', 'low', 'close', 'volume']
    TARGET_COL = 'close'

    X_full_train = pd.concat([X_train, X_valid])

    forecaster = LSTMForecaster(
        X_full_train,
        X_test,
        FEATURE_COLS,
        TARGET_COL,
        input_window,
        target_window,
        strategy=strategy
    )

    n_features = len(FEATURE_COLS)
    hypermodel = RNNHyperModel(n_features, target_window, 'lstm', strategy)
    model = hypermodel.build(kt.HyperParameters.from_config({'values': best_hps}))

    print('\n--- Optimal Model Architecture ---')
    model.summary()

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_mae', patience=15, restore_best_weights=True
    )
    lr_scheduler_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_mae', factor=0.5, patience=7)
    callbacks = [early_stopping_cb, lr_scheduler_cb]
    epochs = 100

    forecaster.fit(model, epochs, callbacks)

    metrics = forecaster.evaluate(forecast_horizon)
    y_pred = forecaster.predict(model, forecast_horizon)
    
    return metrics, y_pred

def run_gru_model(X_train, X_valid, X_test,
                   feature_cols, target_col, forecast_horizon,
                   input_window, target_window,
                   time_unit, data_type, naive_metrics):
    '''High-level function to run the complete GRU forecasting pipeline.'''
    return _run_rnn_pipeline('gru', X_train, X_valid, X_test, 
                             feature_cols, target_col, forecast_horizon,
                             input_window, target_window, 
                             time_unit, data_type, naive_metrics)
