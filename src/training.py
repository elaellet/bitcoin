import keras_tuner as kt
import tensorflow as tf
import tqdm
from tqdm import tqdm

from .models.arima import ARIMAForecaster

def select_best_model(results, naive_metrics, model_type):
    '''
    Selects the best model of a specific type based on performance against a naive model.

    The selection criteria are:
    1. Must have a lower MAPE than the naive model.
    2. Must have a higher DA than the naive model.
    3. Of the candidates, choose the one with the highest DA.
    4. If there's a tie in DA, choose the one with the lowest MAPE.

    Args:
        results (list): A list of metrics from the grid search/hyperparameter tuning.
        naive_metrics (dict): The performance dictionary of the naive model.
        model_type (str): The type of model being selected (e.g., 'ARIMA', 'LSTM').

    Returns:
        The dictionary of the best performing model, or None if none outperform the naive model.
    '''
    model_type_upper = model_type.upper()

    print(f'\n--- Selecting Best {model_type_upper} Model ---')
    if not results:
        print('No successful model results to evaluate.')
        return None

    naive_mape = naive_metrics['mape']
    naive_da = naive_metrics['da']

    print(f'- Naive Model Benchmark: MAPE: {naive_mape:.4f}%, DA: {naive_da:.4f}%')
    
    candidates = [
        result for result in results 
        if result['mape'] < naive_mape and result['da'] > naive_da
    ]

    if not candidates:
        print(f'\nConclusion: No {model_type_upper} model outperformed the naive model.')
        return None

    print(f'\nFound {len(candidates)} candidate model(s) that beat the naive model:')
    for model in candidates:
         identifier = model.get('order', 'hyperparameters')
         print(f'- Model: {identifier}, MAPE: {model["mape"]:.4f}%, DA: {model["da"]:.4f}%')

    # Sort by DA (decreasing), then by MAPE (ascending) to break ties.
    candidates.sort(key=lambda x: (-x['da'], x['mape']))
    best_model = candidates[0]

    print(f'\n--- Best {model_type_upper} Model Chosen ---')
    if model_type_upper == 'ARIMA':
        print(f'- Order: {best_model["order"]}')
    print(f'- MAPE: {best_model["mape"]:.4f}%')
    print(f'- DA: {best_model["da"]:.4f}%')

    return best_model

def generate_arima_orders(p_values, d_values, q_values):
    '''Generates a list of all possible (p, d, q) order combinations for ARIMA.'''    
    orders = list()
    for p in p_values:
        for d in d_values:
            for q in q_values:
                orders.append((p, d, q))

    return orders

def run_arima_grid_search(X_train, X_valid, 
                          target_col, orders, 
                          forecast_horizon, refit_interval):
    '''
    Performs a grid search over ARIMA orders using walk-forward validation.

    Args:
        X_train: The training dataset.
        X_valid: The validation dataset.
        target_col: The name of the target column to forecast.
        orders: A list of (p, d, q) tuples to test.
        forecast_horizon: The number of steps to forecast ahead.
        refit_interval: How often to re-fit the ARIMA model.

    Returns:
        list: A list of dictionaries, where each dictionary contains the results for one order.
    ''' 
    print(f'Starting ARIMA grid search for {len(orders)} combinations...\n')
    all_metrics = list()

    for order in tqdm(orders, desc='ARIMA Grid Search Progress'):
        try:
            model = ARIMAForecaster(X_train, X_valid, target_col, order, True)
            model.fit()
            metrics = model.evaluate(forecast_horizon, refit_interval)
            
            if metrics is not None:
                all_metrics.append(metrics)

        except Exception as e:
            # Catch any unexpected errors during model fitting.
            print(f'ARIMA{order} with error: {e}. Skipping.')
            continue
    
    all_metrics.sort(key=lambda x: x['mape'])
    print('\nARIMA grid search complete.')
    
    return all_metrics

class RNNHyperModel(kt.HyperModel):
    '''
    A Keras Tuner HyperModel for building and tuning sequence-to-sequence
    time series forecasters. It is initialized with a specific RNN type.
    '''
    def __init__(self, n_features, target_window,
                 rnn_type, strategy='direct'):
        self.n_features = n_features
        self.target_window = target_window
        if rnn_type.lower() not in ['lstm', 'gru']:
            raise ValueError('rnn_type must be "lstm" or "gru"')
        self.rnn_type = rnn_type.lower()
        self.strategy = strategy

    def build(self, hp):
        '''Builds a compiled Keras model with tunable hyperparameters.'''
        # Architectural Hyperparameters.
        n_conv_layers = hp.Int('n_conv_layers', min_value=1, max_value=3, step=1)
        kernel_size = hp.Choice('kernel_size', values=[3, 5, 7])
        n_rnn_layers = hp.Int('n_rnn_layers', min_value=1, max_value=3, step=1)

        # Regularization Hyperparameters.
        use_l2 = hp.Boolean('use_l2')
        # It tells the tuner to search the exponents evenly, so it spends just as much effort exploring values between 0.0001 and 0.001 as it does between 0.001 and 0.01.
        l2_rate = hp.Float('l2_rate', min_value=1e-4, max_value=1e-2, sampling='log', parent_name='use_l2', parent_values=[True]) if use_l2 else 0.0
        dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)

        # Compile Hyperparameters.
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
        # Adam and Nadam maintain a separate, adaptive learning rate for every single parameter in a model.
        # They learn which parameters need big steps and which need small steps, allowing them to navigate the complex landscape much more efficiently. 
        # They combine the best ideas from other optimizers (like momentum and RMSprop) into one robust package.
        # TODO: AdamW?
        optimizer = hp.Choice('optimizer', values=['adam', 'nadam'])
        # The clipnorm prevents the 'exploding gradient' problem without corrupting the valuable directional information, making it the standard and superior choice for RNNs.
        clipnorm = hp.Float('clipnorm', min_value=0.5, max_value=1.5, step=0.1)

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(None, self.n_features)))

        for i in range(n_conv_layers):
            # 1D convolutional layer acts as a pre-processor.
            # It takes a long sequence, shortens it by extracting key features, and then passes this shorter, feature-rich sequence to the LSTM/GRU layers.
            model.add(tf.keras.layers.Conv1D(
                # Number of different patterns the layer will learn to detect in a sequence.
                filters=hp.Int(f'filters_{i}', min_value=32, max_value=256, step=32),
                # Length of the pattern to detect.
                # A larger kernel looks for longer-term patterns, while a smaller kernel looks for shorter-term patterns.
                kernel_size=kernel_size,
                # Jump size. It controls the downsampling factor of a sequence.
                strides=1,
                # Causal padding is important for time series as it ensures a convolution at time step t only uses data from t and earlier, preventing any lookahead bias.
                padding='causal',
                # The combination of ReLU + He initialization is a robust default.
                activation='relu', kernel_initializer='he_normal',
                kernel_regularizer=tf.keras.regularizers.l2(l2_rate)
            ))

        RNNCell = tf.keras.layers.LSTM if self.rnn_type == 'lstm' else tf.keras.layers.GRU
        for i in range(n_rnn_layers):
            model.add(RNNCell(
                units=hp.Int(f'units_{i}', min_value=32, max_value=256, step=32),
                # Set return_sequences=True for all recurrent layers.
                # False for one recurrent layer will output a 2D array containing only the output of the last time step.
                # Instead of a 3D array containing outputs for all time steps.
                return_sequences=True, 
                activation='tanh', kernel_initializer='glorot_uniform',
                kernel_regularizer=tf.keras.regularizers.l2(l2_rate),
                # Recurrent dropout randomly drops connections within the recurrent cell. It uses the same dropout mask for every time step in a given sequence. 
                # This prevents overfitting on the time-dependent connections without causing the model to forget what it just saw.
                recurrent_dropout=dropout_rate
            ))
            model.add(tf.keras.layers.LayerNormalization())

         # Add dropout before the final dense layer.
        if dropout_rate > 0:
            model.add(tf.keras.layers.Dropout(dropout_rate))

        if self.strategy == 'direct':
            # For monthly forecast, predict the full target window.
            model.add(tf.keras.layers.Dense(self.target_window))
        else:
            # For yearly forecast, predict only one step ahead.
            model.add(tf.keras.layers.Dense(1))

        if optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=clipnorm)
        else:
            optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate, clipnorm=clipnorm)

        model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics=['mae'])

        return model

    # def fit(self, hp, model, *args, **kwargs):
    #     '''Overrides the fit method to tune the batch size'''
    #     # Define a tunable batch size.
    #     batch_size = hp.Choice('batch_size', values=[32, 64, 128])
    #     # Pass the selected batch size to the original model.fit().
    #     return model.fit(*args, batch_size=batch_size, **kwargs)

def tune_rnn_forecaster(train_set, valid_set, 
                        rnn_type, n_features, 
                        target_window, 
                        train_steps, valid_steps,
                        max_trials=15):
    '''
    Runs Keras Tuner to find the best hyperparameters for an RNN-based forecaster.

    Args:
        train_set: The windowed training dataset.
        valid_set: The windowed validation dataset.
        rnn_type: The type of RNN cell to use ('lstm' or 'gru').
        n_features: The number of input features.
        target_window: The number of steps to forecast.
        train_steps (int): The number of batches per training epoch.
        valid_steps (int): The number of batches per validation epoch.        
        max_trials: The number of hyperparameter combinations to test.

    Returns:
        The best set of hyperparameters found by the tuner.
    '''
    hypermodel = RNNHyperModel(
        n_features,
        target_window,
        rnn_type
    )
    
    # Bayesian optimization is the most efficient with number of trials. 
    # It gradually learns which regions of the hyperparameter space are most promising by fitting a probabilistic model (Gaussian process).
    # This allows it to focus on the most promising areas of the search space, making it ideal for finding the best model when each trial is expensive.
    # It has its own hyperparameters (alpha and beta).
    project_name = f'{rnn_type}_forecaster_tuning'
    tuner = kt.BayesianOptimization(
        hypermodel=hypermodel,
        objective='val_mae',
        max_trials=max_trials,
        directory='keras_tuner',
        project_name=project_name,
        seed=42,
        # Directory is deleted before training starts.
        overwrite=True
    )

    # TODO: Custom metric function (MAPE DA)?
    # TODO: val_loss vs. val_mae?
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_mae', patience=10, verbose=1)
    # A dynamic scheduler can significantly improve performance.
    # ReduceLROnPlateau reduces the learning rate automatically when the validation loss stops improving.
    # It lets the model learn as much as it can with the current learning rate and only steps in when it gets stuck.
    lr_scheduler_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_mae', factor=0.5, patience=5)

    rnn_type_upper = rnn_type.upper()
    print(f'\nStarting hyperparameter search for {rnn_type_upper} model...')
    tuner.search(
        train_set,
        epochs=30,
        validation_data=valid_set,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
        callbacks=[early_stopping_cb, lr_scheduler_cb]
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f'\n{rnn_type_upper} hyperparameter tuning complete:')
    for hp, value in best_hps.values.items():
        print(f'- {hp}: {value}')
    print()
    
    return best_hps