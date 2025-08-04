import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_percentage_error

from .base import *

class NaiveForecaster(BaseForecaster):
    '''A forecaster that uses the naive method (last observation carried forward).'''
    def __init__(self, df_series, target_col):
        super().__init__(df_series, df_series, target_col)

        self.df_series = df_series

    def _calculate_mape(self, y_true, y_pred):
        '''Calculates the Mean Absolute Percentage Error (MAPE).'''
        return mean_absolute_percentage_error(y_true, y_pred) * 100

    def _calculate_da(self, y_true, y_pred, forecast_horizon):
        '''
        Calculates the Directional Accuracy (DA).

        DA measures the percentage of times the model correctly predicts the direction
        of the price change (up or down).
        '''
        # Assumption: The direction over the next X days will be the same as the direction over the previous X days.

        # Get the sign of the price change over the specified horizon (+1 for up, -1 for down).
        true_direction = np.sign(y_true.diff(forecast_horizon).dropna())
        pred_direction = np.sign(y_pred.diff(forecast_horizon).dropna())
        
        # Align the series, as the first prediction might not match the first true value.
        common_index = true_direction.index.intersection(pred_direction.index)
        true_direction = true_direction.loc[common_index]
        pred_direction = pred_direction.loc[common_index]
        
        # Correct prediction if signs are the same.
        correct_preds = (true_direction == pred_direction).sum()
        
        if len(common_index) == 0:
            return 0.0
            
        return (correct_preds / len(common_index)) * 100
    
    def fit(self):
        '''A naive model does not require training.'''
        print('--- Skipping Naive Model Fitting ---')
        pass
    
    def evaluate(self, forecast_horizon):
        '''
        Generates naive forecasts and evaluates them using MAPE and DA.

        Args:
            forecast_horizon (int): The number of steps to forecast ahead.

        Returns:
            dict: A dictionary containing performance metrics.
        '''
        print(f'\n--- Evaluating Naive Model ---')

        y_trues = self.df_series[self.target_col]
        y_preds = self.df_series[self.target_col].shift(forecast_horizon)   

        df_results = pd.DataFrame({
            'true_value': y_trues,
            'prediction': y_preds
        }).dropna()
        
        y_true = df_results['true_value']
        y_pred = df_results['prediction']

        mape = self._calculate_mape(y_true, y_pred)
        da = self._calculate_da(y_true, y_pred, forecast_horizon=forecast_horizon)

        print(f'- Mean Absolute Percentage Error (MAPE): {mape:.4f}%')
        print(f'- Directional Accuracy (DA): {da:.4f}%')
        
        return {'mape': mape, 'da': da}
    
    def predict(self, forecast_horizon, time_unit):
        '''
        Generates a single naive forecast based on the last available observation.

        Returns:
            float: The last value in the target column.

        Raises:
            ValueError: If the unsupported time unit is used.
        '''
        print(f'\n--- Generating Final Naive Forecast ---')

        time_unit_lower = time_unit.lower()
        pred = self.df_series[self.target_col].iloc[-1]
        last_time = self.df_series.index[-1]

        if time_unit_lower == 'days':
            offset = pd.Timedelta(days=forecast_horizon)
        elif time_unit_lower == 'weeks':
            offset = pd.Timedelta(weeks=forecast_horizon)
        else:
            raise ValueError('Error: Unsupported time unit')

        future_time = last_time + offset
        formatted_future_time = future_time.strftime('%Y-%m-%d')

        print(f'- Forecast for {formatted_future_time}: ${pred:.2f}')

        return pred