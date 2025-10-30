import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
import tqdm
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm import tqdm

from .base import BaseForecaster

class LSTMForecaster(BaseForecaster):
    '''A forecaster for LSTM models.''' 
    def __init__(self, train_ds, valid_ds,
                 full_cols, target_col,
                 input_window, target_window,
                 batch_size=32, hold_thld=0.005):
        super().__init__(train_ds, valid_ds, target_col)

        self.input_window = input_window
        self.target_window = target_window
        self.full_cols = full_cols
        self.target_col_idx = self.full_cols.index(self.target_col)
        self.n_full_cols = len(self.full_cols)
        self.batch_size = batch_size
        self.hold_thld = hold_thld

        self.preprocessor = None
        self.train_set, self.valid_set = None, None

    def __deepcopy__(self, memo):
        '''
        Custom deepcopy logic for LSTMForecaster.

        This is called automatically by copy.deepcopy() in the
        StackedEnsembleForecaster.

        It creates a new, fresh instance of the forecaster with the
        same initial configuration. It explicitly avoids copying
        trained artifacts like 'self.model', 'self.history', or
        'self.train_set', which are unpickleable.
        
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
            batch_size=self.batch_size,
            hold_thld=self.hold_thld
        )

        memo[id(self)] = new_wrapper

        return new_wrapper

    def _prepare_data(self):
        '''
        Prepares and transforms the data for model training and evaluation.
        
        1.  Feature Selection: Defines the final list of features to be used by
            the model (`self.model_cols`). It then further splits this list into
            numerical (`self.num_cols`) and categorical (`self.cat_cols`) components.
        
        2.  Preprocessing: Initializes a `ColumnTransformer` (`self.preprocessor`)
            that applies `StandardScaler` to numerical columns and `OneHotEncoder` to
            categorical columns. This pipeline is fit only on the training data
            (using `fit_transform`) and then applied to the validation data
            (using `transform`) to prevent data leakage.
        
        3.  Dataset Creation: Converts the preprocessed NumPy arrays into windowed
            `tf.data.Dataset` objects using the `_to_seq2seq_ds` helper.
            This creates the final (`input_window`, `n_model_cols`) and (`input_window`, `target`) 
            structure required by the sequence model.
        
        4.  Attribute Caching: Stores important calculated values as class attributes. 
            This includes the `tf.data.Dataset` objects (`self.train_set`, `self.valid_set`),
            feature counts (`self.n_model_cols`), and the indices of the target column in different contexts
            (`self.model_target_idx`, `self.num_target_idx`).
        '''
        print('\n--- Preparing Data for LSTM ---')
        
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

        # ColumnTransformer reorders the columns. It processes the transformers in the specified order.
        self.target_log_returns_col = f'log_returns_{self.target_col}'
        self.model_target_idx = self.model_cols.index(self.target_log_returns_col)
        self.num_target_idx = self.num_cols.index(self.target_log_returns_col)

        self.train_ds_transformed = self.preprocessor.fit_transform(self.train_ds[self.model_cols])
        self.valid_ds_transformed = self.preprocessor.transform(self.valid_ds[self.model_cols])

        # The rsi_state column has three unique categories.
        # OneHotEncoder creates a new binary column for each of these categories.
        self.n_model_cols = self.train_ds_transformed.shape[1]
        self.n_num_cols = len(self.num_cols)

        # Create the TensorFlow datasets.
        self.train_set = self._to_seq2seq_ds(self.train_ds_transformed, self.input_window, 
                                             self.target_window, True, 42)
        self.valid_set = self._to_seq2seq_ds(self.valid_ds_transformed, self.input_window,
                                             self.target_window)
        
    def _to_windows(self, ds, len):
        '''
        Converts a time series into a dataset of sliding windows.

        This method takes a flat dataset of individual time steps and
        transforms it into a new dataset where each element is a complete
        sequence (or `window`) of a specified length.

        It operates in two main steps:
        1.  ds.window(): Creates a nested dataset, where each element is
            itself a smaller dataset representing one window.
        2.  flat_map(): Flattens this structure by converting each small
            window dataset into a single tensor, resulting in a clean dataset
            of windowed tensors.

        Args:
            ds (tf.data.Dataset): The input time series data, where each
                element represents a single time step.
            len (int): The desired length of each window (i.e., the
                sequence length for the model).

        Returns:
            tf.data.Dataset: A new dataset where each element is a tensor
                representing one window of shape (`len`, `n_model_cols`).
        '''
        ds = ds.window(len, shift=1, drop_remainder=True)

        return ds.flat_map(lambda window_ds: window_ds.batch(len))
    
    def _to_seq2seq_ds(self, series,
                       input_window, target_window, 
                       shuffle=False, seed=None):
        '''
        Creates a sequence-to-sequence tf.data.Dataset from a time series.

        This function prepares the data for training a sequence model by
        converting a flat time series into pairs of (input, target) windows.

        It trains the model to forecast an entire future sequence based on a
        single, fixed historical point. Each sample consists of an input
        of shape (input_window, n_model_cols) and a target of shape
        (input_window, target_window). This is data-intensive as it requires
        input_window + target_window steps to create one sample.

        Args:
            series (np.array or similar): The input time series data.
            input_window (int): The number of past time steps to use as input.
            target_window (int): The number of future time steps to forecast.
            shuffle (bool): Whether to shuffle the dataset. Defaults to False.
            seed (int, optional): Random seed for shuffling. Defaults to None.

        Returns:
            tf.data.Dataset: A prepared dataset of (input, target) tuples ready
                for model training.
        '''
        ds = self._to_windows(tf.data.Dataset.from_tensor_slices(series), target_window + 1)
        
        # ds = self._to_windows(ds, input_window)
        # - The result is a dataset where each element is a 3D tensor of shape (input_window, target_window + 1, n_features).
        # ds = self._to_windows(ds, input_window).map(lambda S: ...)
        # - S[:, 0] takes the element at index 0 from each of the input_window windows, corresponding to the historical data point in each window.
        # - The resulting input is a 2D tensor of shape (input_window, n_model_cols).
        # - S[:, 1:, self.target_col_idx] creates the corresponding target sequence that the model must predict.
        # - The resulting target is a 2D tensor of shape (input_window, target_window).
        ds = self._to_windows(ds, input_window).map(lambda S: (S[:, 0], S[:, 1:, self.model_target_idx]))

        # Caching should be done before shuffling and batching.
        # It saves the expensive I/O and preprocessing work.
        ds = ds.cache()

        if shuffle:
            # The .repeat() method converts a finite dataset into an infinite one by looping back to the beginning when it reaches the end.
            # Shuffling helps the model become more robust and prevents it from overfitting.
            # It ensures that the model learns the fundamental patterns in a generalized way instead of patterns related to specific ordering.
            ds = ds.shuffle(8 * self.batch_size, seed=seed)

        # Prefetching works by creating a background thread that prepares future batches (CPU work) 
        # while the GPU is busy processing the current batch (GPU work). 
        # This overlapping of data preparation and model execution is the key to eliminating I/O bottlenecks.
        return ds.repeat().batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

    def fit(self, model, epochs, callbacks=None):
        '''
        Takes a compiled Keras model and trains it on the prepared datasets.

        Because the datasets are configured to repeat indefinitely, this method
        relies on the pre-calculated `steps_per_epoch` and `validation_steps`
        attributes to define the length of each epoch. This ensures that the
        model trains on one full pass of the unique windows from the original
        data during each epoch.

        Args:
            model (tf.keras.Model): A compiled Keras model ready for training.
            epochs (int): The number of full training epochs to perform.
            callbacks (list, optional): A list of Keras callbacks to use during
                training. Defaults to None.
        ''' 
        print(f'\n--- Fitting LSTM Model ---')
        
        self.model = model

        if self.train_set is None or self.valid_set is None:
            self._prepare_data()
        
        # Keras needs to know how many steps constitute one epoch for an infinite dataset.
        # If a dataset has N rows, and each sample requires a total of input_window + target_window time steps
        # The total number of complete windows (samples) is N - input_window - target_window + 1.
        steps_per_epoch = (len(self.train_ds) - self.input_window - self.target_window + 1) // self.batch_size
        validation_steps = (len(self.valid_ds) - self.input_window - self.target_window + 1) // self.batch_size

        self.history = self.model.fit(
            self.train_set,
            steps_per_epoch=steps_per_epoch,
            validation_data=self.valid_set,
            validation_steps=validation_steps,
            epochs=epochs,
            callbacks=callbacks,
            verbose=2
        )
    
    def evaluate(self, forecast_horizon):
        '''
        Performs walk-forward validation using a rolling window, no refitting,
        and direct forecasting.

        This method uses a fixed-size rolling window of historical data to 
        make predictions without refitting the model at each step 
        (The model weights are frozen).

        It uses direct forecasting, where the model is trained to
        directly predict the value at the end of a specified `target_window`.
        
        Returns:
            dict: A dictionary containing the performance metrics (`mape`, `da`) and a
                comparison DataFrame (`comp`) with the actual vs. predicted values.

        Raises:
            ValueError: If `forecast_horizon` does not match the model's
                trained `self.target_window`.                
        '''
        if forecast_horizon != self.target_window:
            raise ValueError(
                f'{self.__class__.__name__} is trained to predict '
                f'{self.target_window} steps ahead, but was asked to '
                f'evaluate {forecast_horizon} steps.'
            )
                
        print(f'\n--- Evaluating LSTM Model ---')

        history = self.train_ds_transformed[-self.input_window:].copy()
        loop_end = len(self.valid_ds_transformed) - forecast_horizon
        scaler = self.preprocessor.named_transformers_['num']

        preds_scaled = list()
        true_values = list()
        true_log_returns = list()
        last_values = list()
        comp_indices = list()        

        for t in tqdm(range(loop_end), desc='Walk-Forward Validation'):
            # (1, input_window, n_model_cols)
            input_batch = np.expand_dims(history, axis=0)
            # (1, input_window, target_window)
            pred_scaled = self.model.predict(input_batch, verbose=0)
            preds_scaled.append(pred_scaled[0, -1, -1])

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

    def predict(self, model, forecast_horizon, hold_thld=None):        
        '''
        Generates a final forecast and its corresponding directional signal.

        This method uses the most recent data (`input_window`) from all available history 
        to generate a forecast

        It uses direct multi-step forecasting, where the model is fed the scaled input window
        and outputs a set of future predictions, one of which is selected, inverse-transformed, 
        and converted to the final price for the desired `forecast_horizon`.

        Args:
            model (tf.keras.Model): A compiled Keras model ready for forecasting.
            forecast_horizon (int): The number of steps to forecast.
            hold_thld (float, optional): The percentage threshold for the hold signal.

        Returns:
            dict: A dictionary containing the following keys:
                - pred (float): The single, predicted value for the date
                                 `forecast_horizon` steps after the last known data point.
                - sig (float): The sign of the directional signal 
                                 (1.0 for up, -1.0 for down, 0.0 for hold).

        Raises:
            ValueError: If the requested `forecast_horizon` is greater than the
                model's trained `target_window`.
        '''
        if forecast_horizon < 1 or forecast_horizon > self.target_window:
            raise ValueError(
                f'Forecast horizon ({forecast_horizon}) must be between 1 and '
                f'the model\'s trained target window ({self.target_window}).'
        )

        thld = self.hold_thld if hold_thld is None else hold_thld

        print(f'\n--- Generating Final LSTM (Threshold: {thld * 100}%) Forecast ---')

        scaler = self.preprocessor.named_transformers_['num']

        history = pd.concat([self.train_ds, self.valid_ds])
        last_window_unscaled = history[self.model_cols].tail(self.input_window)
        last_window_scaled = self.preprocessor.transform(last_window_unscaled)   
        input_batch = np.expand_dims(last_window_scaled, axis=0)

        pred_scaled = model.predict(input_batch)
        pred_scaled_log_return = pred_scaled[0, -1, forecast_horizon - 1]

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