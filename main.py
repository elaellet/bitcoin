import os
import sys

import keras_tuner as kt
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LinearRegression

from src import analysis, data_cleaner, data_loader, data_splitter
from src import model, preprocessing, training, utils, visualization
from src.models import XGBForecaster, LSTMForecaster, GRUForecaster

def get_btc_ds_path():
    '''Gets the absolute path to the bundled raw dataset file.'''
    if getattr(sys, 'frozen', False):
        # Path for the PyInstaller bundled executable.
        base_path = sys._MEIPASS
    else:
        # Path for running as a standard Python script.
        base_path = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(base_path, 'data', 'raw', 'btc_274.csv')

def get_data_path(rel_path):
    '''
    Gets the absolute path to a data file.

    Args:
        relative_path (str): The path to the file relative to the 
                             project root (e.g., 'config/models.yaml' 
                             or 'data/raw/btc_274.csv').

    Returns:
        str: The absolute, platform-specific path to the requested data file.        
    '''
    if getattr(sys, 'frozen', False):
        # Path for the PyInstaller bundled executable.
        base_path = sys._MEIPASS
    else:
        # Path for running as a standard Python script.
        # os.path.dirname(os.path.abspath(__file__)) gets the script's directory.
        base_path = os.path.dirname(os.path.abspath(__file__))

    # Join the base path, 'data', the specified subdirectory, and the filename.
    return os.path.join(base_path, rel_path)

def main():
    '''Main function to run the entire BTC price prediction pipeline.'''
    utils.print_header('Bitcoin Price Forecasting: A Comparative Analysis of Statistical, ML, and DL Models: START')

    BTC_DS_PATH = 'data/raw/ohlcv_274.csv'
    FEDFUNDRATE_PATH = 'data/external/fedfunds.csv'
    M2SL_PATH = 'data/external/m2sl.csv'
    CONFIG_PATH = 'config.yaml'

    # 1. Load the Bitcoin dataset.
    btc_ds_path = get_data_path(BTC_DS_PATH)
    fedfundrate_path = get_data_path(FEDFUNDRATE_PATH)
    m2sl_path = get_data_path(M2SL_PATH)
    CONFIG_PATH = get_data_path(CONFIG_PATH)

    raw_ds = data_loader.load_btc_ds(btc_ds_path)

    # 2. Clean and Preprocess Data
    cleaned_ds = data_cleaner.clean_btc_data(raw_ds)
    imputed_ds = data_cleaner.fill_time_series_gaps(cleaned_ds, 'min')    
    resampled_ds = preprocessing.resample_btc_data(imputed_ds)
    train_ds, valid_ds, test_ds = data_splitter.split_btc_ds(resampled_ds['daily'], 'Day')

    # 3. Perform EDA (Analysis and Visualization)
    # Generate visualizations.
    visualization.plot_btc_price_distribution(train_ds, 'price')
    visualization.plot_btc_volume_distribution(train_ds, 'volume')

    visualization.plot_btc_price_boxplot(train_ds)
    visualization.plot_btc_volume_boxplot(train_ds)

    visualization.plot_btc_price_trend(train_ds)
    # Calculate daily log returns. Log calculates percentage change to correctly measure statistical volatility.
    train_ds.loc[:, 'log_return'] = np.log(train_ds['close']).diff()
    # Multiply the 30-day rolling standard deviation of these returns by sqrt(365) to annualize it.
    train_ds.loc[:, 'rolling_volatility'] = train_ds['log_return'].rolling(window=30).std() * np.sqrt(365)
    visualization.plot_btc_price_rollig_volatility(train_ds)
    visualization.plot_btc_volume_trend(train_ds)

    visualization.plot_btc_price_and_volume_corr(train_ds)
    visualization.plot_btc_price_and_volume(train_ds)
    visualization.plot_autocorrelation(train_ds, 'close', 40, 'BTC Daily Closing Price', 'price')
    visualization.plot_autocorrelation(train_ds, 'log_return', 40, 'BTC Daily Closing Log Returns', 'log_return')
    visualization.plot_time_series_decomposition(train_ds, 'close', 'multiplicative', 365, 'BTC Daily Closing Price')
    visualization.plot_stationarity_analysis(train_ds, 'close')
    print()

    # Run statistical tests.
    analysis.run_adf_test(train_ds, 'close', 'Daily Closing Price')
    analysis.run_adf_test(train_ds, 'log_return', 'Daily Closing Log Returns')

    # 4. Build, train and evaluate models
    train_prep_ds, valid_prep_ds, test_prep_ds = preprocessing.prepare_feature_ds(
        train_ds, valid_ds, test_ds,
        ['open', 'high', 'low', 'close', 'volume'],
        fedfundrate_path, m2sl_path
    )
    print()

    FULL_COLS = len(train_prep_ds)
    TARGET_COL = 'close'
    WINDOW_SIZE = 7
    INPUT_WINDOW = 7
    TARGET_WINDOW = 7
    FORECAST_HORIZON = 7

    model_hps = utils.load_config(CONFIG_PATH)

    # Naive Model
    naive_metrics, naive_pred_out = model.train_and_forecast_with_naive_model(train_prep_ds, TARGET_COL, 
                                                                              FORECAST_HORIZON, 'Test', 'Days')
    visualization.plot_residuals_analysis(naive_metrics, 'comp', 14, 'NAIVE', 'Week')
    analysis.run_ljung_box_test(naive_metrics, 'comp', 14, 'NAIVE')
    print(naive_metrics['comp'])    
    print()
    
    # ARIMA Model
    arima_order = tuple(model_hps['ARIMA']['weekly']['order'])
    arima_metrics, arima_pred_out = model.train_and_forecast_with_best_arima_model(train_prep_ds, valid_prep_ds, test_prep_ds, 
                                                                                   arima_order, 'close', TARGET_COL, WINDOW_SIZE,
                                                                                   FORECAST_HORIZON, 'Days')
    visualization.plot_residuals_analysis(arima_metrics, 'comp', 14, 'ARIMA', 'Week')
    analysis.run_ljung_box_test(arima_metrics, 'comp', 14, 'ARIMA')
    print(arima_metrics['comp'])
    print()
    
    lstm_hps = kt.HyperParameters()
    lstm_hps.values = model_hps['LSTM']['weekly']
    lstm_metrics, lstm_pred_out = model.train_and_forecast_with_best_rnn_model('lstm', train_prep_ds, valid_prep_ds, test_prep_ds, 
                                                                               lstm_hps, FULL_COLS, TARGET_COL, FORECAST_HORIZON, 
                                                                               INPUT_WINDOW, TARGET_WINDOW, 'Days')
    visualization.plot_residuals_analysis(lstm_metrics, 'comp', 14, 'LSTM', 'Week')
    analysis.run_ljung_box_test(lstm_metrics, 'comp', 14, 'LSTM')
    print(lstm_metrics['comp'])
    print()

    # GRU Model
    gru_hps = kt.HyperParameters()
    gru_hps.values = model_hps['GRU']['weekly']
    gru_metrics, gru_pred_out = model.train_and_forecast_with_best_rnn_model('gru', train_prep_ds, valid_prep_ds, test_prep_ds, 
                                                                              gru_hps, FULL_COLS, TARGET_COL, FORECAST_HORIZON, 
                                                                              INPUT_WINDOW, TARGET_WINDOW, 'Days')   
    visualization.plot_residuals_analysis(gru_metrics, 'comp', 14, 'GRU', 'Week')
    analysis.run_ljung_box_test(gru_metrics, 'comp', 14, 'GRU')
    print(gru_metrics['comp'])
    print()

    # XGBoost Model
    xgb_params = model_hps['XGB']['weekly']
    xgb_metrics, xgb_pred_out = model.train_and_forecast_with_best_xgb_model(train_prep_ds, valid_prep_ds, test_prep_ds,
                                                                             xgb_params, FULL_COLS, TARGET_COL, 
                                                                             INPUT_WINDOW, TARGET_WINDOW, 
                                                                             FORECAST_HORIZON, 'Days')
    visualization.plot_residuals_analysis(xgb_metrics, 'comp', 14, 'XGB', 'Week')
    analysis.run_ljung_box_test(xgb_metrics, 'comp', 14, 'XGB')
    print(gru_metrics['comp'])
    print()
    
    # Stacked Ensemble Model
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(f'lstm_validation.weights.h5', save_weights_only=True)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_mae', patience=15, restore_best_weights=True)
    lr_scheduler_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_mae', factor=0.5, patience=7)

    callbacks_fit = [checkpoint_cb, early_stopping_cb, lr_scheduler_cb]
    callbacks_eval_and_pred = [checkpoint_cb]

    xgb_model = XGBForecaster(train_prep_ds, valid_prep_ds,
                              FULL_COLS, TARGET_COL,
                              INPUT_WINDOW, TARGET_WINDOW)
    xgb_params = {
        'params': model_hps['XGB']['weekly']
    }

    lstm_model = LSTMForecaster(train_prep_ds, valid_prep_ds,
                                FULL_COLS, TARGET_COL,
                                INPUT_WINDOW, TARGET_WINDOW)
    lstm_model._prepare_data()
    lstm_hps = kt.HyperParameters()
    lstm_hps.values = model_hps['LSTM']['weekly']
    lstm_hypermodel = training.RNNHyperModel(lstm_model.n_model_cols, TARGET_WINDOW, 'lstm').build(lstm_hps)
    lstm_params_fit = {
        'model': lstm_hypermodel,
        'epochs': 100,
        'callbacks': callbacks_fit
    }
    lstm_params_eval_and_pred = {
        'model': lstm_hypermodel,
        'epochs': 100,
        'callbacks': callbacks_eval_and_pred
    }

    base_models = [lstm_model, xgb_model]

    base_model_params_fit = {
        'forecaster_lstm': lstm_params_fit,
        'forecaster_xgb': xgb_params
    }

    base_model_params_eval_and_pred = {
        'forecaster_lstm': lstm_params_eval_and_pred,
        'forecaster_xgb': xgb_params
    }

    stacked_metrics_xl, stacked_pred_out_xl = model.train_and_forecast_with_stacked_ensemble_model(base_models, LinearRegression(),
                                                                                                   test_prep_ds, base_model_params_fit, base_model_params_eval_and_pred,
                                                                                                   FORECAST_HORIZON, 'Days')
    
    visualization.plot_residuals_analysis(stacked_metrics_xl, 'comp', 14, 'Stacked Ensemble (LSTM+XGB)', 'Week')
    analysis.run_ljung_box_test(stacked_metrics_xl, 'comp', 14, 'Stacked Ensemble (LSTM+XGB)')
    print(stacked_metrics_xl['comp'])
    print()

    utils.print_header('Bitcoin Price Forecasting: A Comparative Analysis of Statistical, ML, and DL Models: COMPLETE')

    input('\nPress Enter to exit...')
    
if __name__ == '__main__':
    main()