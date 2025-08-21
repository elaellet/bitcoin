from warnings import catch_warnings
from warnings import filterwarnings

import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
import tqdm
from tqdm import tqdm

from .base import *

class ARIMAForecaster(BaseForecaster):
    '''A forecaster for the ARIMA model, optimized for walk-forward validation.'''
    def __init__(self, X_train, X_valid, target_col, order, use_tqdm):
        if not isinstance(order, tuple) or len(order) != 3:
            raise ValueError('Order must be a tuple of (p, d, q).')
        
        super().__init__(X_train, X_valid, target_col)

        self.order = order
        self.use_tqdm = use_tqdm

    def _inverse_transform(self, log_value):
        '''Helper function to reverse the log transformation.'''
        return np.exp(log_value)

    def _calculate_da(self, Y_results, history, forecast_horizon):
        '''Calculates directional accuracy from a results dataframe.'''
        # Make a copy to avoid modifying the original dataframe.
        Y = Y_results.copy()

        # Find the starting price for each forecast.
        # The price at the beginning of the forecast period is needed to determine direction.
        # Shifting the full log-transformed price series produces the price.
        # The start date for a prediction made for 't' is 't - forecast_horizon'.        
        start_values = history.shift(forecast_horizon)
        # Add the starting prices to the results dataframe, aligning by index.
        Y['start_value'] = start_values
        # Drop any rows where a start price cannot be obtained (at the beginning).
        Y.dropna(inplace=True)

        if Y.empty:
            return 0.0

        # Determine the true and predicted directions.
        # A direction is positive (1) if the price went up, and negative (-1) if it went down.
        Y['true_direction'] = np.sign(Y['true_value'] - Y['start_value'])
        Y['pred_direction'] = np.sign(Y['prediction'] - Y['start_value'])
        
        # Compare the directions.
        # The prediction is correct if the signs are the same (e.g., both positive or both negative).
        correct_preds = (Y['true_direction'] == Y['pred_direction']).sum()

        return correct_preds / len(Y) * 100
    
    def fit(self):
        '''
        Fits the ARIMA model on the entire training dataset.
        The fitted model is stored in the `self.model_fit` attribute.
        '''
        if self.use_tqdm == False:
            print(f'--- Fitting ARIMA{self.order} Model ---')

        x_train = self.X_train[self.target_col]
        x_train_transformed = np.log(x_train)

        with catch_warnings():
            filterwarnings('ignore')
            model = ARIMA(x_train_transformed, order=self.order)
            model_fit = model.fit()

    def evaluate(self, forecast_horizon, refit_interval):
        '''
        Performs walk-forward validation on the validation set to evaluate performance.

        This method iteratively trains on a growing history of data to predict 
        future points in the validation set.

        Args:
            forecast_horizon (int): How many steps ahead to forecast.
            refit_interval (int): How often to re-fit the model.

        Returns:
            dict: A dictionary containing performance metrics (MAPE, DA) and the model order.
        '''
        if self.use_tqdm == False:
            print(f'\n--- Evaluating ARIMA{self.order} Model ---')

        # Log transformation stabilizes the variance.
        # Differencing stabilizes the mean by removing or reducing the trend and seasonality.
        x_train = self.X_train[self.target_col]
        x_valid = self.X_valid[self.target_col]

        x_train_transformed = np.log(x_train)
        x_valid_transformed = np.log(x_valid)
        
        history = list(x_train_transformed)
        y_preds = list()
        y_trues = list()

        # The loop should only go up to the point where the last forecast has a corresponding true value.
        loop_end = len(x_valid_transformed) - forecast_horizon
        
        for t in range(loop_end):
            # Re-fit the model at the specified interval.
            if t % refit_interval == 0:
                # Suppress convergence warnings to keep the output clean.            
                with catch_warnings():
                    filterwarnings('ignore')
                    model = ARIMA(history, order=self.order)
                    model_fit = model.fit()
            
            # Forecast and inverse-transform.
            # The .forecast() method automatically handles un-differencing.
            # The validation set starts a day after the last day of the training set.
            # The forecast has to cover the period up to the first day of the validation set.
            # 1 has to be added to the forecast horizon variable.      
            y_pred_transformed = model_fit.forecast(steps=forecast_horizon + 1)
            y_pred = self._inverse_transform(y_pred_transformed[-1])
            y_preds.append(y_pred)
            
            # Get the corresponding true value (already in original scale).
            y_true = x_valid.iloc[t + forecast_horizon]
            y_trues.append(y_true)
            
            # Update history with the transformed value from the validation set.
            history.append(x_valid_transformed.iloc[t])

        Y_results = pd.DataFrame({
            'true_value': y_trues,
            'prediction': y_preds
        }, index=self.X_valid.index[forecast_horizon:]) # The index starts from the first date that has a true value.

        y_true = Y_results['true_value']
        y_pred = Y_results['prediction']

        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        # Combine train and validation data to get a full history for lookback.
        history = pd.concat([x_train, x_valid])
        da = self._calculate_da(Y_results, history, forecast_horizon)

        if self.use_tqdm == True:
            tqdm.write(f'Order: {self.order}, MAPE: {mape:.4f}%, DA: {da:.4f}%')
        else:
            print(f'- Order: {self.order}')
            print(f'- Mean Absolute Percentage Error (MAPE): {mape:.4f}%')
            print(f'- Directional Accuracy (DA): {da:.4f}%')
                
        return {
            'order': self.order,
            'mape': mape,
            'da': da,
            'forecast_comparison': Y_results
        }
    
    def predict(self, forecast_horizon):
        '''
        Trains a final model on all available data (train + valid) and 
        generates a forecast for the actual future.

        Args:
            forecast_horizon (int): The number of steps to forecast into the future.

        Returns:
            pd.DataFrame: A DataFrame containing the future forecast with dates.
        '''
        print(f'\n--- Generating Final ARIMA{self.order} Forecast ---')

        history = pd.concat([self.X_train[self.target_col], self.X_valid[self.target_col]])
        history_transformed = np.log(history)

        with catch_warnings():
            filterwarnings('ignore')
            model = ARIMA(history_transformed, order=self.order)
            model_fit = model.fit()

        y_pred_transformed = model_fit.forecast(steps=forecast_horizon)
        y_pred = self._inverse_transform(y_pred_transformed.iloc[-1])

        last_time = history.index[-1]
        freq = history.index.freqstr
        future_time = pd.date_range(start=last_time, periods=forecast_horizon + 1, freq=freq)[-1]
        formatted_future_time = future_time.date().strftime('%Y-%m-%d')

        print(f'- Forecast for {formatted_future_time}: ${y_pred:.2f}')

        return y_pred