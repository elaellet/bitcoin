import pandas as pd
import numpy as np
import sklearn
import tensorflow as tf
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import tqdm
from tqdm import tqdm

from .base import BaseForecaster
from ..preprocessing import calculate_returns

class GRUForecaster(BaseForecaster):
    '''A forecaster for GRU-based models, handling data preparation, training, and evaluation.''' 
    def __init__(self, X_train, X_valid,
                 feature_cols, target_col,
                 input_window, target_window, 
                 batch_size=32, strategy='direct'):
        super().__init__(X_train, X_valid, target_col)

        self.input_window = input_window
        self.target_window = target_window
        self.feature_cols = feature_cols
        self.target_col_index = self.feature_cols.index(self.target_col)
        self.n_features = len(self.feature_cols)
        self.batch_size = batch_size
        self.strategy = strategy

        self.scaler = StandardScaler()
        self.train_set, self.valid_set = None, None
        self.valid_set_unbatched = None
        self.X_train_processed, self.X_valid_processed = None, None
        self.n_valid_samples = 0
        
    def _prepare_data(self):
        '''Prepares the data by calculating returns, scaling, and creating windowed TensorFlow datasets.'''        
        print('--- Preparing Data for GRU ---')
        self.X_train_processed = self.X_train.copy()
        self.X_valid_processed = self.X_valid.copy()

        # -----------------------------------------------------------------
        # self.X_train_processed = add_technical_indicators(self.X_train_processed)
        # self.X_valid_processed = add_technical_indicators(self.X_valid_processed)
        # indicator_cols = ['RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']
        # self.feature_cols.extend(indicator_cols)
        # self.n_features = len(self.feature_cols)
        # -----------------------------------------------------------------

        self.feature_cols_returns = list()
        for col in self.feature_cols:
            return_col = f'returns_{col}'
            self.X_train_processed = calculate_returns(self.X_train_processed, col)
            self.X_valid_processed = calculate_returns(self.X_valid_processed, col)
            self.feature_cols_returns.append(return_col)

        self.X_train_processed.dropna(inplace=True)
        self.X_valid_processed.dropna(inplace=True)

        self.scaler.fit(self.X_train_processed[self.feature_cols_returns])
        X_train_scaled = self.scaler.transform(self.X_train_processed[self.feature_cols_returns])
        X_valid_scaled = self.scaler.transform(self.X_valid_processed[self.feature_cols_returns])

        self.n_valid_samples = len(X_valid_scaled) - self.input_window - self.target_window + 1
        
        self.train_set = self._to_seq2seq_dataset(X_train_scaled, self.input_window, 
                                                  self.target_window, True, 42, True)
        self.valid_set = self._to_seq2seq_dataset(X_valid_scaled, self.input_window,
                                                  self.target_window)
        self.valid_set_unbatched = self._to_seq2seq_dataset(X_valid_scaled, self.input_window, 
                                                            self.target_window, is_training=False)

    def _to_windows(self, dataset, length):
        '''Helper function to create rolling windows.'''
        dataset = dataset.window(length, shift=1, drop_remainder=True)
        return dataset.flat_map(lambda window_ds: window_ds.batch(length))
    
    def _to_seq2seq_dataset(self, series,
                            input_window, target_window, 
                            shuffle=False, seed=None, is_training=True):
        '''
            Creates a sequence-to-sequence dataset from a time series.
            Trains the model to make a target_window-step forecast at every single point in input_window.
        '''
        if self.strategy == 'direct':
            ds = self._to_windows(tf.data.Dataset.from_tensor_slices(series), target_window + 1)
            ds = self._to_windows(ds, input_window).map(lambda S: (S[:, 0], S[:, 1:, self.target_col_index]))
        else:
            ds = self._to_windows(tf.data.Dataset.from_tensor_slices(series), input_window + 1)
            ds = ds.map(lambda window: (window[:-1], window[-1, self.target_col_index]))            

        ds = ds.cache()
        
        if shuffle:
            ds = ds.shuffle(8 * self.batch_size, seed=seed)

        if is_training:
            return ds.repeat().batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        else:
            return ds

    def fit(self, model, epochs, callbacks=None):
        '''
        Takes a compiled Keras model and trains it on the prepared datasets.

        Args:
            model (tf.keras.Model): A compiled Keras model.
            epochs (int): The number of epochs to train the model.
            callbacks (list): A list of Keras callbacks to use during training.
        '''       
        print(f'--- Fitting GRU Model ---')
        
        self.model = model

        if self.train_set is None or self.valid_set is None:
            self._prepare_data()

        if self.strategy == 'direct':
            steps_per_epoch = (len(self.X_train_processed) - self.input_window - self.target_window + 1) // self.batch_size
            validation_steps = (len(self.X_valid_processed) - self.input_window - self.target_window + 1) // self.batch_size
        else:
            steps_per_epoch = (len(self.X_train_processed) - self.input_window) // self.batch_size
            validation_samples = len(self.X_valid_processed) - self.input_window
            if validation_samples > 0:
                validation_steps = int(np.ceil(validation_samples / self.batch_size))
            else:
                validation_steps = 0    

        self.history = self.model.fit(
            self.train_set,
            steps_per_epoch=steps_per_epoch,
            validation_data=self.valid_set,
            validation_steps=validation_steps,
            epochs=epochs,
            callbacks=callbacks,
            verbose=2
        )
    
    def evaluate(self, forecast_horizon=None):
        '''
        Evaluates the trained model on the validation set. This method calculates
        MAPE and DA for the final day of the target window across all predictions.
        
        Returns:
            dict: A dictionary containing the final performance metrics.
        '''
        print(f'\n--- Evaluating GRU Model ---')

        if self.strategy == 'direct':
            return self._evaluate_direct()
        else:
            if forecast_horizon is None:
                raise ValueError('forecast_horizon must be provided for iterative evaluation.')            
            return self._evaluate_iterative(forecast_horizon)

    def _evaluate_direct(self):    
        X_eval_unscaled = self.X_valid_processed

        eval_set = self.valid_set_unbatched.take(self.n_valid_samples).batch(self.batch_size)
        Y_preds = self.model.predict(eval_set)

        mape_scores = list()
        da_scores = list()

        for ahead in range(self.target_window):
            y_pred_scaled_returns = Y_preds[:, -1, ahead]

            dummy_array = np.zeros((len(y_pred_scaled_returns), self.n_features))
            dummy_array[:, self.target_col_index] = y_pred_scaled_returns
            y_pred_returns = self.scaler.inverse_transform(dummy_array)[:, self.target_col_index]            

            n_preds = len(y_pred_returns)
            indices = np.arange(n_preds)
            
            last_known_prices = X_eval_unscaled[self.target_col].iloc[indices + self.input_window - 1].values
            y_true_prices = X_eval_unscaled[self.target_col].iloc[indices + self.input_window + ahead].values
            y_true_returns = X_eval_unscaled[f'returns_{self.target_col}'].iloc[indices + self.input_window + ahead].values
            y_pred_prices = last_known_prices * (1 + y_pred_returns)

            mape = mean_absolute_percentage_error(y_true_prices, y_pred_prices) * 100
            mape_scores.append(mape)

            pred_direction = np.sign(y_pred_returns)
            true_direction = np.sign(y_true_returns)

            da = np.mean(pred_direction == true_direction) * 100
            da_scores.append(da)

            print(f'Day {ahead + 1} | MAPE: {mape:6.4f}% | DA: {da:6.4f}%')

        mape = mape_scores[-1]
        da = da_scores[-1]

        print(f'- Mean Absolute Percentage Error (MAPE): {mape:.4f}%')
        print(f'- Directional Accuracy (DA): {da:.4f}%')

        return {'mape': mape, 'da': da}

    def _evaluate_iterative(self, forecast_horizon):
        '''
        Evaluates the one-step-ahead model using iterative walk-forward validation.

        Args:
            forecast_horizon (int): The number of steps to forecast iteratively (e.g., 52).
        '''        
        history = self.X_train_processed.copy()
        y_trues = list()
        y_preds = list()

        loop_end = len(self.X_valid_processed) - forecast_horizon

        for i in tqdm(range(loop_end), desc='Walk-Forward Validation'):
            current_history = pd.concat([history, self.X_valid_processed.iloc[:i]])
            current_window_unscaled = current_history[self.feature_cols_returns].tail(self.input_window)
            current_window_scaled = self.scaler.transform(current_window_unscaled)
            
            iterative_preds_scaled = list()
            for _ in range(forecast_horizon):
                model_input = np.expand_dims(current_window_scaled, axis=0)
                y_pred_scaled = self.model.predict(model_input, verbose=0)[0, 0]
                iterative_preds_scaled.append(y_pred_scaled)
                
                dummy_row = np.zeros((1, self.n_features))
                dummy_row[0, self.target_col_index] = y_pred_scaled
                
                current_window_scaled = np.vstack([current_window_scaled[1:], dummy_row])
            
            y_pred_scaled = iterative_preds_scaled[-1]
            
            dummy_array = np.zeros((1, self.n_features))
            dummy_array[0, self.target_col_index] = y_pred_scaled
            y_pred_return = self.scaler.inverse_transform(dummy_array)[0, self.target_col_index]
            
            last_known_price = current_history[self.target_col].iloc[-1]
            y_pred = last_known_price * (1 + y_pred_return)
            y_preds.append(y_pred)

            y_true = self.X_valid_processed[self.target_col].iloc[i + forecast_horizon - 1]
            y_trues.append(y_true)

        common_index = self.X_valid_processed.index[forecast_horizon - 1: forecast_horizon - 1 + len(y_trues)]
        y_true = pd.Series(y_trues, index=common_index)
        y_pred = pd.Series(y_preds, index=common_index)

        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        da = self._calculate_da(y_true, y_pred)

        print(f'\n- Mean Absolute Percentage Error (MAPE): {mape:.4f}%')
        print(f'- Directional Accuracy (DA): {da:.4f}%')

        return {'mape': mape, 'da': da}
    
    def _calculate_da(self, y_true, y_pred):
        '''Calculates Directional Accuracy.'''
        true_diff = np.sign(y_true.diff().dropna())
        pred_diff = np.sign(y_pred.diff().dropna())

        common_index = true_diff.index.intersection(pred_diff.index)
        true_diff = true_diff.loc[common_index]
        pred_diff = pred_diff.loc[common_index]
        
        return np.mean(true_diff == pred_diff) * 100 
    
    def predict(self, model, forecast_horizon):
        '''
        Generates a single future prediction using the final trained model.
        It uses the last `input_window` of available data as input.

        Args:
            model (tf.keras.Model): The final, trained Keras model.
            forecast_horizon (int): The number of steps ahead to predict (e.g., 30 or 365).

        Returns:
            float: The single predicted value for the target horizon.
        '''
        print(f'\n--- Generating Final GRU Forecast ---')

        if self.strategy == 'direct':
            return self._predict_direct(model, forecast_horizon)
        else:
            return self._predict_iterative(model, forecast_horizon)

    def _predict_direct(self, model, forecast_horizon):            
        history_unscaled = pd.concat([self.X_train_processed, self.X_valid_processed])

        for col in self.feature_cols:
            if f'{col}_returns' not in history_unscaled.columns:
                history_unscaled = calculate_returns(history_unscaled, col)
        
        history_unscaled.dropna(inplace=True)

        last_window_unscaled = history_unscaled[self.feature_cols_returns].tail(self.input_window)
        last_window_scaled = self.scaler.transform(last_window_unscaled)
        model_input = np.expand_dims(last_window_scaled, axis=0)

        y_pred_scaled_seq = model.predict(model_input)
        y_pred_scaled_vector = y_pred_scaled_seq[0, -1, :]
        y_pred_scaled = y_pred_scaled_vector[forecast_horizon - 1]

        dummy_array = np.zeros((1, self.n_features))
        dummy_array[0, self.target_col_index] = y_pred_scaled
        y_pred_return = self.scaler.inverse_transform(dummy_array)[0, self.target_col_index]

        last_known_price = history_unscaled[self.target_col].iloc[-1]
        y_pred = last_known_price * (1 + y_pred_return)

        last_time = history_unscaled.index[-1]
        freq = history_unscaled.index.freqstr
        future_time = pd.date_range(start=last_time, periods=forecast_horizon + 1, freq=freq)[-1]
        formatted_future_time = future_time.date().strftime('%Y-%m-%d')        

        print(f'- Forecast for {formatted_future_time}: ${y_pred:.2f}')

        return y_pred
    
    def _predict_iterative(self, model, forecast_horizon):
        '''Generates a single future forecast using the iterative method.'''        
        self.model = model
        history_unscaled = pd.concat([self.X_train_processed, self.X_valid_processed])
        
        last_window_unscaled = history_unscaled[self.feature_cols_returns].tail(self.input_window)
        current_window_scaled = self.scaler.transform(last_window_unscaled)

        iterative_preds_scaled = []
        for _ in range(forecast_horizon):
            model_input = np.expand_dims(current_window_scaled, axis=0)
            y_pred_scaled = self.model.predict(model_input, verbose=0)[0, 0]
            iterative_preds_scaled.append(y_pred_scaled)
            
            dummy_row = np.zeros((1, self.n_features))
            dummy_row[0, self.target_col_index] = y_pred_scaled
            current_window_scaled = np.vstack([current_window_scaled[1:], dummy_row])
        
        y_pred_scaled = iterative_preds_scaled[-1]
        dummy_array = np.zeros((1, self.n_features))
        dummy_array[0, self.target_col_index] = y_pred_scaled
        y_pred_return = self.scaler.inverse_transform(dummy_array)[0, self.target_col_index]

        last_known_price = history_unscaled[self.target_col].iloc[-1]
        y_pred = last_known_price * (1 + y_pred_return)

        last_time = history_unscaled.index[-1]
        freq = history_unscaled.index.freqstr
        future_time = pd.date_range(start=last_time, periods=forecast_horizon + 1, freq=freq)[-1]
        formatted_future_time = future_time.date().strftime('%Y-%m-%d')        

        print(f'- Forecast for {formatted_future_time}: ${y_pred:.2f}')
        
        return y_pred            