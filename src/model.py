from .models.arima import ARIMAForecaster
from .models.lstm import LSTMForecaster
from .models.naive import NaiveForecaster
from .training import *
from .utils import print_header

def run_naive_model(df_series, forecast_horizon,
                    target_col, unit, type):
    '''
    Runs a naive forecast model, evaluates its performance, and gets a prediction.

    This function first evaluates the naive model's historical performance on the
    provided series (calculating MAPE and DA). It then generates a single
    forecast for the future based on the last value in the series.

    Args:
        df_series (pd.DataFrame): The DataFrame containing the time series data.
        forecast_horizon (int): The number of periods to forecast into the future (e.g., 30 for 30 days).
        target_col (str): The name of the column to forecast.
        unit (str): The time unit for the forecast horizon (e.g., 'days', 'weeks').
        data_type (str): The type of data set being used (e.g., 'Validation', 'Test').

    Returns:
        tuple[dict, float]: A tuple containing:
            - A dictionary of evaluation metrics ('mape', 'da').
            - A single float value representing the future prediction.
    '''
    print_header(f'Naive Model Evaluation on {type} Set (Horizon: {forecast_horizon} {unit})')

    naive_model = NaiveForecaster(df_series, target_col)

    naive_model.fit()

    metrics = naive_model.evaluate(forecast_horizon)

    preds = naive_model.predict()
    
    return metrics, preds

def run_arima_model(df_train, df_valid, target_col, 
                    p_values, d_values, q_values, 
                    forecast_horizon, refit_interval, naive_result,
                    type, unit):
    '''
    High-level function to find and evaluate the best ARIMA model.

    This function orchestrates the entire ARIMA pipeline:
    1. Defines a search space for ARIMA orders.
    2. Runs a grid search using walk-forward validation.
    3. Selects the best model based on performance against a baseline.

    Args:
        df_train: The training dataset.
        df_valid: The validation dataset.
        forecast_horizon (int): The number of steps to forecast ahead.
        naive_result (dict): The results from the naive model for comparison.
        target_col (str): The name of the target column.

    Returns:
        The results dictionary of the best performing ARIMA model.
    '''
    print(f'\nARIMA Model Evaluation on {type} Set (Horizon: {forecast_horizon} {unit})...')

    # 1. Define the hyperparameter search space.
    orders = generate_arima_orders(p_values, d_values, q_values)
    
    # 2. Run the grid search.
    # A smaller refit interval is more accurate but much slower. 30 is a reasonable trade-off.
    arima_results = run_arima_grid_search(
        df_train, df_valid, target_col,
        orders, forecast_horizon, refit_interval
    )
    
    # 3. Select the best model.
    best_arima_model = select_best_arima_model(arima_results, naive_result)

    # 4. Predict a price a month/year ahead.
    if preds:
        preds = best_arima_model.predict()
    
    return best_arima_model

def run_optimized_arima_model(df_train, df_valid, target_col, 
                              best_order, forecast_horizon, refit_interval, unit):
    '''Runs the pre-tuned, optimal ARIMA model.'''

    print(f'\nRunning optimized ARIMA{best_order} model (Horizon: {forecast_horizon} {unit})...')
    forecaster = ARIMAForecaster(df_train, df_valid, target_col, order=best_order)
    
    result = forecaster.fit_and_evaluate(forecast_horizon, refit_interval)
    
    print(f'MAPE: {result["mape"]:.4f}%, DA: {result["da"]:.4f}%')
    return result

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