import pandas as pd

from .models.arima import ARIMAForecaster
from .models.lstm import LSTMForecaster
from .models.naive import NaiveForecaster
from .training import *
from .utils import print_header

def run_naive_model(df_series, target_col, forecast_horizon,
                    time_unit, data_type):
    '''
    Initializes, evaluates, and gets a final prediction from a NaiveForecaster.

    This function first evaluates the naive model's historical performance on the
    provided series (calculating MAPE and DA). It then generates a single final
    forecast for the future based on the last available value.

    Args:
        df_series (pd.DataFrame): The DataFrame containing the time series data.
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

    model = NaiveForecaster(df_series, target_col)
    # The 'fit' method does nothing but is called for interface consistency.
    model.fit()
    metrics = model.evaluate(forecast_horizon)
    pred = model.predict(forecast_horizon, time_unit)

    return metrics, pred

def run_arima_model(df_train, df_valid, df_test, target_col, forecast_horizon,
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
        df_train: The training dataset.
        df_valid: The validation dataset.
        df_test: The test dataset.
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
        df_train, df_valid, target_col,
        orders, forecast_horizon, refit_interval
    )
    best_metrics = select_best_model(search_results, naive_metrics, 'arima')

    metrics = None
    pred = None 

    if best_metrics:
        best_order = best_metrics['order']

        df_full_train = pd.concat([df_train, df_valid])
        model = ARIMAForecaster(df_full_train, df_test, target_col, best_order, False)

        print('\nFitting, evaluating, and predicting ARIMA model with the best order...')

        model.fit()
        # Setting `refit_interval` to a value greater than 1 significantly speeds up the full grid search process.
        # However, this can lead to a 'stale' model, resulting in less accurate performance.
        # A `refit_interval` of 1 ensures the model is evaluated with the best order and produces a more reliable performance metric.
        metrics = model.evaluate(forecast_horizon, 1)
        pred = model.predict(forecast_horizon)
    else:
        print('\nNo best ARIMA model')
    
    return metrics, pred

# def run_optimized_arima_model(df_train, df_valid, target_col, 
#                               best_order, forecast_horizon, refit_interval, unit):
#     '''Runs the pre-tuned, optimal ARIMA model.'''

#     print(f'\nRunning optimized ARIMA{best_order} model (Horizon: {forecast_horizon} {unit})...')
#     forecaster = ARIMAForecaster(df_train, df_valid, target_col, order=best_order)
    
#     result = forecaster.fit_and_evaluate(forecast_horizon, refit_interval)
    
#     print(f'MAPE: {result["mape"]:.4f}%, DA: {result["da"]:.4f}%')
#     return result

def run_lstm_model(df_train, df_valid, input_window, target_window, unit):
    '''
    High-level function to run the LSTM forecasting pipeline.

    This function orchestrates the entire process:
    1. Defines the model's configuration (features, windows).
    2. Optionally runs the hyperparameter tuner to find the best model architecture.
    3. Instantiates and trains the final LSTM forecaster.
    4. Evaluates the model and returns the performance metrics.

    Args:
        df_train: The training dataset.
        df_valid: The validation dataset.
        forecast_horizon (int): The number of steps to forecast ahead.
        use_tuner (bool): If True, runs the Keras Tuner. If False, uses pre-defined
                          optimal hyperparameters. Defaults to False.

    Returns:
        A dictionary containing the model's performance metrics.
    '''
    print(f'\nLSTM model pipeline (Horizon: {target_window} {unit})')

    FEATURE_COLS = ['open', 'high', 'low', 'close', 'volume']
    TARGET_COL = 'close'

    lstm_forecaster = LSTMForecaster(
        df_train,
        df_valid,
        input_window,
        target_window,
        FEATURE_COLS,
        TARGET_COL
    )

    train_set, valid_set = lstm_forecaster._prepare_data()
    best_hps = tune_rnn_forecaster(
        train_set, valid_set, 'lstm', len(FEATURE_COLS), target_window
    )
    model = RNNHyperModel(len(FEATURE_COLS), target_window, 'lstm').build(best_hps)

    model.summary()
    
    results = lstm_forecaster.train_and_evaluate(model, 100)
    
    return results

def run_optimized_lstm_model(df_train, df_valid, input_window, target_window, unit):
    '''Runs the pre-tuned, optimal LSTM model.'''
    print(f'\nUsing pre-defined optimal hyperparameters for the LSTM model (Horizon: {target_window} {unit})...')

    FEATURE_COLS = ['open', 'high', 'low', 'close', 'volume']
    TARGET_COL = 'close'

    lstm_forecaster = LSTMForecaster(
        df_train,
        df_valid,
        input_window,
        target_window,
        FEATURE_COLS,
        TARGET_COL
    )

    hps = {
        'n_conv_layers': 1, 'filters_0': 128, 'kernel_size_0': 5,
        'n_rnn_layers': 2, 'units_0': 128, 'units_1': 64,
        'dropout_rate': 0.2, 'learning_rate': 0.001
    }

    hypermodel = RNNHyperModel(len(FEATURE_COLS), target_window, 'lstm')
    model = hypermodel.build(kt.HyperParameters.from_config({'values': hps}))

    model.summary()
    
    results = lstm_forecaster.train_and_evaluate(model, 100)
    
    return results