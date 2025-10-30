import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm import tqdm

from .base import BaseForecaster

class XGBForecaster(BaseForecaster):
    '''A forecaster for XGBoost models.'''
    def __init__(self, train_ds, valid_ds,
                 full_cols, target_col,
                 input_window, target_window,
                 hold_thld=0.005):
        super().__init__(train_ds, valid_ds, target_col)

        self.input_window = input_window
        self.target_window = target_window
        self.full_cols = full_cols
        self.hold_thld = hold_thld
        self.preprocessor = None
        self.models = list() # Store one model for each forecast horizon.

        self.X_train, self.y_train = list(), list()
        self.X_valid, self.y_valid = list(), list()

    def __deepcopy__(self, memo):
        '''
        Custom deepcopy logic for XGBForecaster.
        
        This is called automatically by copy.deepcopy() in the
        StackedEnsembleForecaster.
        
        It creates a new, fresh instance of the forecaster with the
        same initial configuration. It explicitly avoids copying
        trained artifacts like 'self.models', 'self.preprocessor',
        'self.X_train', 'self.y_train', etc.
        
        This forces .fit() to call ._prepare_data() again on the
        new data (train_full_ds and test_ds).
        '''
        new_wrapper = self.__class__(
            train_ds=self.train_ds,
            valid_ds=self.valid_ds,
            full_cols=self.full_cols,
            target_col=self.target_col,
            input_window=self.input_window,
            target_window=self.target_window,
            hold_thld=self.hold_thld
        )
    
        memo[id(self)] = new_wrapper
        
        return new_wrapper

    def _create_tabular_ds(self, data, forecast_horizon):
        '''Converts time series data into a tabular format for XGBoost.'''
        '''
        Converts time series data into a tabular format.

        It works by creating lagged features. For each time step `t`,
        it takes a window of historical data (from `t-input_window` to `t-1`),
        flattens it into a single feature vector (X), and maps it to a
        single future target (y) at `t + forecast_horizon - 1`.

        Args:
            data (np.ndarray): The preprocessed time series data, with
                shape (n_samples, n_model_cols).
            forecast_horizon (int): The number of steps to use 
            as the target 'y' label for each window.

        Returns:
            (np.ndarray, np.ndarray): A tuple containing:
                - X (np.ndarray): The tabular data of shape
                  (n_samples, input_window * n_model_cols).
                - y (np.ndarray): The target vector of shape (n_samples,).
        '''
        X, y = list(), list()

        loop_end = len(data) - self.input_window - forecast_horizon + 1

        for i in range(loop_end):
            window = data[i:(i + self.input_window)]
            target = data[i + self.input_window + forecast_horizon - 1, self.model_target_idx]
            X.append(window)
            y.append(target)

        X = np.array(X).reshape(len(X), -1) # Flatten the window (n_samples, input_window, n_model_cols) -> (n_samples, input_window * n_model_cols)
        y = np.array(y)
        return X, y

    def _prepare_data(self):
        '''
        Prepares and transforms data into a tabular format for XGBoost.

        This method sets up the ColumnTransformer, trains it on the training data, 
        and then generates all required training and validation sets
        for all forecast horizons.

        It loops from 1 to `self.target_window`, calling
        `_create_tabular_dataset` for each step to create and store
        separate datasets (e.g., `self.X_trains[0]` for the 1-day forecast,
        `self.X_trains[1]` for the 2-day forecast, etc.).

        Returns:
            None: This method modifies the class attributes in-place, setting:
                - `self.preprocessor`
                - `self.train_ds_transformed`, `self.valid_ds_transformed`
                - `self.X_trains`, `self.y_trains` (lists of np.ndarray)
                - `self.X_valids`, `self.y_valids` (lists of np.ndarray)
        '''

        '''Prepares and transforms data into a tabular format.'''
        print('\n--- Preparing Data for XGBoost ---')

        base_cols = ['open', 'high', 'low', 'close', 'volume', 'log_open', 'log_high', 'log_low', 'log_close', 'log_volume']
        self.model_cols = [col for col in self.full_cols if col not in base_cols]
        self.num_cols = [col for col in self.model_cols if col != 'rsi_state']
        self.cat_cols = ['rsi_state']

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.num_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.cat_cols)
            ],
            remainder='passthrough'
        )

        self.target_log_returns_col = f'log_returns_{self.target_col}'

        self.model_target_idx = self.model_cols.index(self.target_log_returns_col)
        self.num_target_idx = self.num_cols.index(self.target_log_returns_col)

        self.train_ds_transformed = self.preprocessor.fit_transform(self.train_ds[self.model_cols])
        self.valid_ds_transformed = self.preprocessor.transform(self.valid_ds[self.model_cols])

        self.n_model_cols = self.train_ds_transformed[1]
        self.n_num_cols = len(self.num_cols)

        # Create a separate training dataset for each forecast horizon.
        for horizon in range(1, self.target_window + 1):
            X_train, y_train = self._create_tabular_ds(self.train_ds_transformed, horizon)
            # (target_window, n_samples - input_window, n_model_cols * n_samples)
            self.X_train.append(X_train)
            # (target_window, n_samples - input_window)
            self.y_train.append(y_train)

            X_valid, y_valid = self._create_tabular_ds(self.valid_ds_transformed, horizon)
            self.X_valid.append(X_valid)
            self.y_valid.append(y_valid)

    def fit(self, params):
        '''
        Trains the complete set of XGBoost models, one for each forecast horizon
        (the direct forecasting strategy).

        It iterates from 1 to `self.target_window`, training a separate
        `xgb.XGBRegressor` model on the corresponding dataset
        (e.g., `self.X_trains[0]`, `self.y_trains[0]`). It uses the
        corresponding validation set (`self.X_valids`, `self.y_valids`)
        for early stopping.

        Args:
            params (dict): A dictionary of hyperparameters to initialize
                `xgb.XGBRegressor`. For compatibility with older XGBoost
                versions, this dict *must* include 'early_stopping_rounds'.
        '''
        print(f'\n--- Fitting XGBoost Model ---')
        if not self.X_train or not self.X_valid:
            self._prepare_data()

        self.models = []
        for horizon in tqdm(range(1, self.target_window + 1), desc='Training'):
            model = xgb.XGBRegressor(**params)

            X_train = self.X_train[horizon - 1]
            y_train = self.y_train[horizon - 1]
            X_valid = self.X_valid[horizon - 1]
            y_valid = self.y_valid[horizon - 1]

            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], 
                      verbose=False)
            self.models.append(model)

    def evaluate(self, forecast_horizon):
        '''
        Performs walk-forward validation for a single, specific forecast horizon.

        This method implements a rolling window, no-refitting, and direct forecasting strategy. 
        It selects the model trained specifically for the given `forecast_horizon`
        and iterates through the validation set. In each step, it predicts the future value 
        using the current history window, stores the prediction, and then updates its history by 
        adding the true data from the current step and dropping the oldest data point (rolling window).

        Args:
            forecast_horizon (int): The specific future step to predict
                (e.g., `8` to test the 8-day-ahead model). Must be
                between 1 and `self.target_window`.
                                
        Returns:
            dict: A dictionary containing the performance metrics (`mape`, `da`) and a
                comparison DataFrame (`comp`) with the actual vs. predicted values.

        Raises:
            ValueError: If `forecast_horizon` is not between 1 and
                `self.target_window`.
        '''
        if forecast_horizon < 1 or forecast_horizon > self.target_window:
            raise ValueError(
                f'Forecast horizon ({forecast_horizon}) must be between 1 and '
                f'the model\'s trained target window ({self.target_window}).'
        )

        print('\n--- Evaluating XGBoost Model ---')
        model = self.models[forecast_horizon - 1]

        history = self.train_ds_transformed[-self.input_window:].copy()
        loop_end = len(self.valid_ds_transformed) - forecast_horizon
        scaler = self.preprocessor.named_transformers_['num']

        preds_scaled = list()
        true_values = list()
        true_log_returns =list()
        last_values =list()
        comp_indices = list()

        for t in tqdm(range(loop_end), desc='Walk-Forward Validation'):
            input_vector = history.flatten().reshape(1, -1)
            pred_scaled = model.predict(input_vector)[0]
            preds_scaled.append(pred_scaled)

            true_idx_loc = t + forecast_horizon
            comp_indices.append(self.valid_ds.index[true_idx_loc])
            true_values.append(self.valid_ds[self.target_col].iloc[true_idx_loc])
            true_log_returns.append(self.valid_ds[self.target_log_returns_col].iloc[true_idx_loc])

            last_idx_loc = true_idx_loc - forecast_horizon

            if last_idx_loc < 0:
                last_values.append(self.train_ds[self.target_col].iloc[-1])
            else:
                last_values.append(self.valid_ds[self.target_col].iloc[last_idx_loc])

            history = np.vstack([history, self.valid_ds_transformed[t]])
            history = history[1:]

        buf = np.zeros((len(preds_scaled), self.n_num_cols))
        buf[:, self.num_target_idx] = preds_scaled
        pred_log_returns = scaler.inverse_transform(buf)[:, self.num_target_idx]
        pred_values = np.array(last_values) * np.exp(pred_log_returns)

        mape = mean_absolute_percentage_error(true_values, pred_values) * 100
        pred_dirs = np.sign(pred_log_returns)
        true_dirs = np.sign(true_log_returns)
        da = np.mean(pred_dirs == true_dirs) * 100

        print(f'- Mean Absolute Percentage Error (MAPE): {mape:.4f}%')
        print(f'- Directional Accuracy (DA): {da:.4f}%')

        comp = pd.DataFrame({
            'true': true_values,
            'pred': pred_values
        }, index=comp_indices)

        return {
            'mape': mape,
            'da': da,
            'comp': comp
        }

    def predict(self, forecast_horizon, hold_thld=None):
        '''
        Generates a single, final forecast and its corresponding directional signal
        using the appropriate trained model.

        It first combines all available data (train + valid) to create the
        most up-to-date input window. It then selects the appropriate model
        for the specified `forecast_horizon`, makes a prediction,
        and converts it from a scaled log return to a final price and
        directional signal.

        Args:
            forecast_horizon (int): The specific future step to predict
                (e.g., `8` to get the 8-day-ahead forecast). Must be
                between 1 and `self.target_window`.
            hold_thld (float, optional): The percentage threshold for the hold signal.

        Returns:
            dict: A dictionary containing the following keys:
                - pred (float): The single, predicted value for the date
                                 `forecast_horizon` steps after the last known data point.
                - sig (float): The sign of the directional signal 
                                 (1.0 for up, -1.0 for down, 0.0 for hold).

        Raises:
            ValueError: If `forecast_horizon` is not between 1 and
                `self.target_window`.
        '''

        if forecast_horizon < 1 or forecast_horizon > self.target_window:
            raise ValueError(
                f'Forecast horizon ({forecast_horizon}) must be between 1 and '
                f'the model\'s trained target window ({self.target_window}).'
        )

        thld = self.hold_thld if hold_thld is None else hold_thld

        print(f'\n--- Generating Final XGBoost (Threshold: {thld * 100}%) Forecast ---')

        scaler = self.preprocessor.named_transformers_['num']
        model = self.models[forecast_horizon - 1]

        history = pd.concat([self.train_ds, self.valid_ds])
        last_window_unscaled = history[self.model_cols].tail(self.input_window)
        last_window_scaled = self.preprocessor.transform(last_window_unscaled)
        input_vector = last_window_scaled.flatten().reshape(1, -1)

        pred_scaled_log_return = model.predict(input_vector)[0]

        buf = np.zeros((1, self.n_num_cols))
        buf[0, self.num_target_idx] = pred_scaled_log_return
        pred_log_return = scaler.inverse_transform(buf)[0, self.num_target_idx]

        last_value = history[self.target_col].iloc[-1]
        pred = last_value * np.exp(pred_log_return)

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