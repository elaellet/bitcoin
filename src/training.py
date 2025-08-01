import keras_tuner as kt
import tensorflow as tf
import tqdm
from tqdm import tqdm

from .models.arima import ARIMAForecaster

def generate_arima_orders(p_values, d_values, q_values):
    '''Generates a list of all possible (p, d, q) order combinations for ARIMA.'''    
    orders = list()
    for p in p_values:
        for d in d_values:
            for q in q_values:
                orders.append((p, d, q))

    return orders

def run_arima_grid_search(df_train, df_valid, target_col, orders, forecast_horizon, refit_interval):
    '''
    Performs a grid search over ARIMA orders using walk-forward validation.

    Args:
        df_train: The training dataset.
        df_valid: The validation dataset.
        target_col: The name of the target column to forecast.
        orders: A list of (p, d, q) tuples to test.
        forecast_horizon: The number of steps to forecast ahead.
        refit_interval: How often to re-fit the ARIMA model.

    Returns:
        A list of dictionaries, where each dictionary contains the results for one order.
    ''' 
    print(f'\nStarting ARIMA grid search for {len(orders)} combinations...')

    results = list()

    for order in tqdm(orders, desc='ARIMA Grid Search Progress'):
        try:
            model = ARIMAForecaster(df_train, df_valid, target_col, order)
            result = model.fit_and_evaluate(forecast_horizon, refit_interval)
            
            if result is not None:
                print(f'ARIMA{order} MAPE={result["mape"]:.4f}% DA={result["da"]:.4f}%')
                results.append(result)

        except Exception as e:
            # Catch any unexpected errors during model fitting.
            print(f'ARIMA{order} with error: {e}. Skipping.')
            continue
    
    results.sort(key=lambda x: x['mape'])
    print('\nARIMA grid search complete.')
    
    return results

def select_best_model(results, naive_result, type):
    '''
    Selects the best model of a specific type based on performance against a naive model.

    The selection criteria are:
    1. Must have a lower MAPE than the naive model.
    2. Must have a higher Horizon DA than the naive model.
    3. Of the candidates, choose the one with the highest Horizon DA.
    4. If there's a tie in DA, choose the one with the lowest MAPE.

    Args:
        results: A list of results from the grid search or hyperparameters search.
        naive_result: The performance dictionary of the naive model.

    Returns:
        The dictionary of the best performing model, or None if none outperform the naive model.
    '''
    type = type.capitalize()

    print(f'\n--- Selecting Best {type} Model ---')
    if not results:
        print('No successful results to evaluate.')
        return

    naive_mape = naive_result['mape']
    naive_da = naive_result['da']

    print(f'Naive Model: MAPE={naive_mape:.4f}%, Horizon DA={naive_da:.4f}%')
    
    candidates = [
        result for result in results 
        if result['mape'] < naive_mape and result['da'] > naive_da
    ]

    if not candidates:
        print('\nConclusion: No {type} model outperformed the naive model.')
        return None

    print(f'\nFound {len(candidates)} candidate model(s) that beat the naive model:')
    for model in candidates:
         print(f'  - Order: {model["order"]}, MAPE: {model["mape"]:.4f}%, DA: {model["da"]:.4f}%')

    # Sort by DA (desc), then by MAPE (asc).
    candidates.sort(key=lambda x: (-x['da'], x['mape']))
    best_model = candidates[0]

    print('\n--- Best {type} Model Chosen ---')
    print(f'Order: {best_model["order"]}')
    print(f'MAPE: {best_model["mape"]:.4f}%')
    print(f'DA: {best_model["da"]:.4f}%')

    return best_model

class RNNHyperModel(kt.HyperModel):
    '''
    A Keras Tuner HyperModel for building and tuning time series forecasters.
    It is initialized with a specific RNN type ("lstm" or "gru").
    '''
    def __init__(self, n_features, target_window, rnn_type):
        self.n_features = n_features
        self.target_window = target_window
        if rnn_type.lower() not in ['lstm', 'gru']:
            raise ValueError('rnn_type must be "lstm" or "gru"')
        self.rnn_type = rnn_type.lower()

    def build(self, hp):
        '''Builds a compiled Keras model with tunable hyperparameters.'''
        # Architectural Hyperparameters.
        n_conv_layers = hp.Int('n_conv_layers', min_value=1, max_value=3, step=1)
        kernel_size = hp.Choice('kernel_size', values=[3, 5, 7])
        n_rnn_layers = hp.Int('n_rnn_layers', min_value=1, max_value=8, step=1)

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
                kernel_regularizer=tf.keras.regularizers.l2(l2_rate)# if use_l2 else None
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
                kernel_regularizer=tf.keras.regularizers.l2(l2_rate), # if use_l2 else None,
                # Recurrent dropout randomly drops connections within the recurrent cell, but—and this is the key—it uses the same dropout mask for every time step in a given sequence. 
                # This prevents overfitting on the time-dependent connections without causing the model to forget what it just saw.
                recurrent_dropout=dropout_rate
            ))
            model.add(tf.keras.layers.LayerNormalization())

         # Add Dropout before the final Dense layer.       
        if dropout_rate > 0:
            model.add(tf.keras.layers.Dropout(dropout_rate))

        model.add(tf.keras.layers.Dense(self.target_window))

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

def tune_rnn_forecaster(train_set, valid_set, rnn_type, n_features, target_window, max_trials=15):
    '''
    Runs Keras Tuner to find the best hyperparameters for an RNN-based forecaster.

    Args:
        train_set: The windowed training dataset.
        valid_set: The windowed validation dataset.
        rnn_type: The type of RNN cell to use ('lstm' or 'gru').
        n_features: The number of input features.
        target_window: The number of steps to forecast.
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
        objective='val_loss',
        max_trials=max_trials,
        directory='keras_tuner',
        project_name=project_name,
        # Directory is deleted before training starts.
        overwrite=True
    )

    # TODO: Custom metric function (MAPE DA)?
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    # A dynamic scheduler can significantly improve performance.
    # ReduceLROnPlateau reduces the learning rate automatically when the validation loss stops improving.
    # It lets the model learn as much as it can with the current learning rate and only steps in when it gets stuck.
    lr_scheduler_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

    print(f'\nStarting hyperparameter search for {rnn_type.upper()} Model...')
    tuner.search(
        train_set,
        epochs=30,
        validation_data=valid_set,
        callbacks=[early_stopping_cb, lr_scheduler_cb]
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f'\nSearch Complete for {rnn_type.upper()}. Best Hyperparameters: ---')
    for hp, value in best_hps.values.items():
        print(f'{hp}: {value}')
    
    return best_hps