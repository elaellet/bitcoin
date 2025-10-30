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
    '''A forecaster for the ARIMA models.'''
    def __init__(self, train_ds, valid_ds,
                 target_col, window_size, order, 
                 use_tqdm, hold_thld=0.005):
        if not isinstance(order, tuple) or len(order) != 3:
            raise ValueError('Order must be a tuple of (p, d, q).')
        
        super().__init__(train_ds, valid_ds, target_col)

        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.window_size = window_size
        self.order = order
        self.use_tqdm = use_tqdm
        self.hold_thld = hold_thld
        self.window_size = window_size

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
    
    def _calculate_da(self, comp, history, forecast_horizon):
        '''
        Calculates the Directional Accuracy (DA).

        For a forecast made at time `t` for a future time `t + forecast_horizon`,
        this function defines:
        - The `baselines` as the actual values at time `t`.
        - The true direction as sign (value at `t + forecast_horizon` - baselines).
        - The predicted direction as sign (prediction for `t + forecast_horizon` - baselines).

        Args:
            comp (pd.DataFrame): The DataFrame containing the actual and predicted values for comparison.
            history (pd.Series): The Series containing the complete, true historical values of the time series.
            forecast_horizon (int): The number of steps to forecast.

        Returns:
            float: A DA value, returned as a percentage.            
                Returns 50.0 if the comp DataFrame becomes empty after handling NaNs.
        '''
        comp_copy = comp.copy()

        # Find a baseline price for each forecast.
        # The price at the beginning of the forecast period is needed to determine direction.
        baselines = history.shift(forecast_horizon)
        # Add the baseline prices to the comp DataFrame, aligning by index.
        comp_copy['baseline'] = baselines
        # Drop any rows where a baseline price cannot be obtained (at the beginning).
        comp_copy.dropna(inplace=True)

        if comp_copy.empty:
            return 0.0

        comp_copy['true_dir'] = np.sign(comp_copy['true'] - comp_copy['baseline'])
        comp_copy['pred_dir'] = np.sign(comp_copy['pred'] - comp_copy['baseline'])
        
        correct_preds = (comp_copy['true_dir'] == comp_copy['pred_dir']).sum()

        return correct_preds / len(comp_copy) * 100
    
    def fit(self):
        '''
        Placeholder for API consistency. No fitting required.
        '''
        if self.use_tqdm == False:
            print(f'--- Fitting ARIMA{self.order} Model ---')
        pass

    def evaluate(self, forecast_horizon):
        '''
        Performs walk-forward validation using a rolling window, fixed parameters,
        and direct forecasting.

        This method trains the model once on the initial training data to get
        a fixed set of parameters. It then iterates through the validation set,
        moving a fixed-size window of data one step at a time (rolling window).

        At each step, it applies the original, fixed parameters to the new window
        of data to directly forecast 'forecast_horizon' steps into the future.

        Args:
            forecast_horizon (int): The number of steps to forecast.

        Returns:
            dict: A dictionary containing performance metrics (`mape`, `da`), a model order, 
                and a comparison DataFrame (`comp`) with the actual vs. predicted values.
        '''
        if self.use_tqdm == False:
            print(f'\n--- Evaluating ARIMA{self.order} Model ---')

        train_ds = self.train_ds[self.target_col]
        valid_ds = self.valid_ds[self.target_col]

        train_ds_transformed = np.log(train_ds)
        valid_ds_transformed = np.log(valid_ds)
        
        full_history_transformed = list(train_ds_transformed)
        preds = list()
        trues = list()

        loop_end = len(valid_ds_transformed) - forecast_horizon
        history_transformed = list(train_ds_transformed[-self.window_size:])

        with catch_warnings():
            filterwarnings('ignore')
            model = ARIMA(full_history_transformed, order=self.order)
            model_fit = model.fit()
            params = model_fit.params
        
        for t in range(loop_end):
            # Apply the initial parameters to the current data window without refitting.           
            model = ARIMA(history_transformed, order=self.order)
            model_fit = model.filter(params)

            # The validation dataset starts a day after the last day of the training dataset.
            # The forecast has to cover the period up to the first day of the validation dataset.
            # 1 has to be added to the forecast horizon variable.
            pred_transformed = model_fit.forecast(steps=forecast_horizon + 1)[-1]
            pred = np.exp(pred_transformed)
            preds.append(pred)
            
            # Get the corresponding true value (already in original scale).
            true = valid_ds.iloc[t + forecast_horizon]
            trues.append(true)
            
            # Update the history using a rolling window.
            history_transformed.append(valid_ds_transformed.iloc[t])
            history_transformed.pop(0)

        comp = pd.DataFrame({
            'true': trues,
            'pred': preds
        }, index=self.valid_ds.index[forecast_horizon:]) # The index starts from the first date that has a true value.

        true = comp['true']
        pred = comp['pred']

        mape = self._calculate_mape(true, pred)
        history = pd.concat([train_ds, valid_ds])
        da = self._calculate_da(comp, history, forecast_horizon)

        if self.use_tqdm == True:
            tqdm.write(f' Order: {self.order}, MAPE: {mape:.4f}%, DA: {da:.4f}%')
        else:
            print(f'- Order: {self.order}')
            print(f'- Mean Absolute Percentage Error (MAPE): {mape:.4f}%')
            print(f'- Directional Accuracy (DA): {da:.4f}%')
                
        return {
            'order': self.order,
            'mape': mape,
            'da': da,
            'comp': comp
        }
    
    def predict(self, forecast_horizon, hold_thld=None):
        '''
        Trains a final model on all available history to forecast 
        a single future value and its corresponding directional signal.

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

        print(f'\n--- Generating Final ARIMA{self.order} (Threshold: {thld * 100}%) Forecast ---')

        history = pd.concat([self.train_ds[self.target_col], self.valid_ds[self.target_col]])
        history_transformed = np.log(history)

        last_value = history.iloc[-1]

        with catch_warnings():
            filterwarnings('ignore')
            model = ARIMA(history_transformed, order=self.order)
            model_fit = model.fit()

        pred_transformed = model_fit.forecast(steps=forecast_horizon).iloc[-1]
        pred = np.exp(pred_transformed)

        if last_value == 0:
            diff = pred - last_value
            sig = np.sign(diff)
        else:
            pct_change = (pred - last_value) / last_value
            
            if pct_change > thld:
                sig = 1.0
            elif pct_change < -thld:
                sig = -1.0
            else:
                sig = 0.0

        last_time = history.index[-1]
        freq = pd.infer_freq(history.index)
        future_time = pd.date_range(start=last_time, periods=forecast_horizon + 1, freq=freq)[-1]
        formatted_future_time = future_time.date().strftime('%Y-%m-%d')

        print(f'- Forecast for {formatted_future_time}: ${pred:.2f}')
        print(f'- Directional Signal for {formatted_future_time}: {sig:.1f}')
        
        return {'pred': pred, 'sig': sig}