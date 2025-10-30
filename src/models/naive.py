import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_percentage_error

from .base import *

class NaiveForecaster(BaseForecaster):
    '''A forecaster for naive persistece models.'''
    def __init__(self, series_ds, target_col, hold_thld=0.005):
        super().__init__(series_ds, series_ds, target_col)

        self.series_ds = series_ds
        self.hold_thld = hold_thld

    def _calculate_mape(self, true, pred):
        '''
        Calculates the Mean Absolute Percentage Error (MAPE).
        
        Args:
            true (pd.Series): The Series of the truth values.
            pred (pd.Series): The Series of the predicted values.
        
        Returns:
            float: The MAPE value as a percentage.
        '''      
        return mean_absolute_percentage_error(true, pred) * 100

    def _calculate_da(self, true, pred, forecast_horizon):
        '''
        Calculates the Directional Accuracy (DA).

        For a comparison over the `forecast_horizon`, this method finds the 
        directional change for both the true and predicted values. It uses 
        pandas' `.diff(forecast_horizon)` to determine the direction of change 
        (up or down) and `.intersection()` to align the resulting true and 
        predicted direction Series.

        Args:
            true (pd.Series): The Series of truth values.
            pred (pd.Series): The Series of predicted values.
            forecast_horizon (int): The number of steps to forecast.

        Returns:
            float: A DA value, returned as a percentage.
                Returns 50.0 if the pred DataFrame becomes empty after handling NaNs.
        '''
        # Assumption: The direction over the next X days will be the same as the direction over the previous X days.

        # Get the sign of the price change over the specified horizon (+1 for up, -1 for down).
        true_dirs = np.sign(true.diff(forecast_horizon).dropna())
        pred_dirs = np.sign(pred.diff(forecast_horizon).dropna())
        
        # Align the series, as the first prediction might not match the first true value.
        common_idx = true_dirs.index.intersection(pred_dirs.index)
        true_dirs = true_dirs.loc[common_idx]
        pred_dirs = pred_dirs.loc[common_idx]
        
        # Correct prediction if the signs are the same.
        correct_preds = (true_dirs == pred_dirs).sum()
        
        # If the pred_direction variable ends up being empty, use 50% (a coin flip) as the baseline.
        if len(common_idx) == 0:
            return 50.0
            
        return (correct_preds / len(common_idx)) * 100
    
    def fit(self):
        '''
        Placeholder for API consistency. No fitting required.
        '''        
        print('--- Fitting Naive Model ---')
        pass
    
    def evaluate(self, forecast_horizon):
        '''
        Generates and evaluates a naive forecast using the shift method.

        Args:
            forecast_horizon (int): The number of steps to forecast.

        Returns:
            dict: A dictionary containing performance metrics ('mape', 'da') and a
                DataFrame ('comp') with the actual vs. predicted values.
        '''
        print(f'\n--- Evaluating Naive Model ---')

        trues = self.series_ds[self.target_col]
        preds = self.series_ds[self.target_col].shift(forecast_horizon)   

        comp = pd.DataFrame({
            'true': trues,
            'pred': preds
        }).dropna()
        
        true = comp['true']
        pred = comp['pred']

        mape = self._calculate_mape(true, pred)
        da = self._calculate_da(true, pred, forecast_horizon)

        print(f'- Mean Absolute Percentage Error (MAPE): {mape:.4f}%')
        print(f'- Directional Accuracy (DA): {da:.4f}%')
        
        return {
            'mape': mape,
            'da': da, 
            'comp': comp
        }
    
    def predict(self, forecast_horizon, hold_thld=None):
        '''
        Generates a single naive forecast and its directional signal 
        based on the last available observation.

        The directional signal is calculated based on the percentage change 
        between the last observation and the second-to-last observation.

        Args:
            forecast_horizon (int): The number of steps to forecast.
            hold_thld (float, optional): The percentage threshold for the hold signal.

        Returns:
            dict: A dictionary containing the following keys:
                - pred (float): The single, predicted value for the date
                    `forecast_horizon` steps after the last known data point.
                - sig (float): The sign of the directional signal 
                    (1.0 for up, -1.0 for down, 0.0 for hold).
        '''
        thld = self.hold_thld if hold_thld is None else hold_thld

        print(f'\n--- Generating Final Naive Forecast (Threshold: {thld * 100}%) ---')

        last_value = self.series_ds[self.target_col].iloc[-1]
        prev_value = self.series_ds[self.target_col].iloc[-2]

        pred = last_value

        if prev_value == 0:
            diff = last_value - prev_value
            sig = np.sign(diff)
        else:
            pct_change = (last_value - prev_value) / prev_value            
            if pct_change > thld:
                sig = 1.0
            elif pct_change < -thld:
                sig = -1.0
            else:
                sig = 0.0

        last_time = self.series_ds.index[-1]
        freq = pd.infer_freq(self.series_ds.index)
        future_time = pd.date_range(start=last_time, periods=forecast_horizon + 1, freq=freq)[-1]
        formatted_future_time = future_time.date().strftime('%Y-%m-%d')

        print(f'- Forecast for {formatted_future_time}: ${pred:.2f}')
        print(f'- Directional Signal for {formatted_future_time}: {sig:.1f}')
        
        return {'pred': pred, 'sig': sig}