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
    def __init__(self, df_train, df_valid, target_col, order, use_tqdm):
        if not isinstance(order, tuple) or len(order) != 3:
            raise ValueError('Order must be a tuple of (p, d, q).')
        
        super().__init__(df_train, df_valid, target_col)

        self.order = order
        self.use_tqdm = use_tqdm

    def _inverse_transform(self, log_value):
        '''Helper function to reverse the log transformation.'''
        return np.exp(log_value)

    def _calculate_da(self, df_results, s_history, forecast_horizon):
        '''Calculates directional accuracy from a results dataframe.'''
        # Make a copy to avoid modifying the original dataframe.
        df = df_results.copy()

        # Find the starting price for each forecast.
        # The price at the beginning of the forecast period is needed to determine direction.
        # Shifting the full log-transformed price series produces the price.
        # The start date for a prediction made for 't' is 't - forecast_horizon'.        
        start_values = s_history.shift(forecast_horizon)
        # Add the starting prices to the results dataframe, aligning by index.
        df['start_value'] = start_values
        # Drop any rows where a start price cannot be obtained (at the beginning).
        df.dropna(inplace=True)

        if df.empty:
            return 0.0

        # Determine the true and predicted directions.
        # A direction is positive (1) if the price went up, and negative (-1) if it went down.
        df['true_direction'] = np.sign(df['true_value'] - df['start_value'])
        df['pred_direction'] = np.sign(df['prediction'] - df['start_value'])
        
        # Compare the directions.
        # The prediction is correct if the signs are the same (e.g., both positive or both negative).
        correct_preds = (df['true_direction'] == df['pred_direction']).sum()

        return correct_preds / len(df) * 100
    
    def fit(self):
        '''
        Fits the ARIMA model on the entire training dataset.
        The fitted model is stored in the `self.model_fit` attribute.
        '''
        if self.use_tqdm == False:
            print(f'--- Fitting ARIMA{self.order} Model on Training Data ---')

        s_train = self.df_train[self.target_col]
        train_transformed = np.log(s_train)

        with catch_warnings():
            filterwarnings('ignore')
            model = ARIMA(train_transformed, order=self.order)
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
        s_train = self.df_train[self.target_col]
        s_valid = self.df_valid[self.target_col]

        train_transformed = np.log(s_train)
        valid_transformed = np.log(s_valid)
        
        history = list(train_transformed)
        y_preds = list()
        y_trues = list()

        # The loop should only go up to the point where the last forecast has a corresponding true value.
        loop_end = len(valid_transformed) - forecast_horizon
        
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
            y_true = s_valid.iloc[t + forecast_horizon]
            y_trues.append(y_true)
            
            # Update history with the transformed value from the validation set.
            history.append(valid_transformed.iloc[t])

        df_results = pd.DataFrame({
            'true_value': y_trues,
            'prediction': y_preds
        }, index=self.df_valid.index[forecast_horizon:]) # The index starts from the first date that has a true value.

        y_true = df_results['true_value']
        y_pred = df_results['prediction']

        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        # Combine train and validation data to get a full history for lookback.
        s_history = pd.concat([s_train, s_valid])
        da = self._calculate_da(df_results, 
                                s_history, 
                                forecast_horizon)

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
            'forecast_comparison': df_results
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

        s_history = pd.concat([self.df_train[self.target_col], self.df_valid[self.target_col]])
        history_transformed = np.log(s_history)

        with catch_warnings():
            filterwarnings('ignore')
            model = ARIMA(history_transformed, order=self.order)
            model_fit = model.fit()

        pred_transformed = model_fit.forecast(steps=forecast_horizon)
        pred = self._inverse_transform(pred_transformed.iloc[-1])

        last_time = s_history.index[-1]
        freq = s_history.index.freqstr
        future_time = pd.date_range(start=last_time, periods=forecast_horizon + 1, freq=freq)[-1]
        formatted_future_time = future_time.date().strftime('%Y-%m-%d')

        print(f'- Forecast for {formatted_future_time}: ${pred:.2f}')

        return pred