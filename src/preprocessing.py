import finta
import numpy as np
import pandas as pd

from finta import TA

from .utils import print_header

def resample_btc_data(df):
    '''
    Resamples the 1-minute BTC data to various timeframes (hourly, daily, weekly, monthly).

    This function aggregates the OHLCV (Open, High, Low, Close, Volume) data for
    each new timeframe. The aggregation rules are:
    - 'open': First price in the period.
    - 'high': Maximum price in the period.
    - 'low': Minimum price in the period.
    - 'close': Last price in the period.
    - 'volume': Sum of all volumes in the period.

    Args:
        df (pd.DataFrame): The cleaned, 1-minute frequency DataFrame with a DatetimeIndex.

    Returns:
        dict[str, pd.DataFrame]: A dictionary where keys are the resampling frequencies
                                 (e.g., 'hourly', 'daily') and values are the
                                 corresponding resampled DataFrames.
    '''
    print('\n--- Resampling BTC Data to Multiple Timeframes ---')

    resampling_rules  = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }

    timeframes = {
        'hourly': 'h', 
        'daily': 'D', 
        'weekly': 'W', 
        'monthly': 'ME'
    }

    resampled = dict()

    for name, freq in timeframes.items():
      print(f'- Resampling to {name} frequency...')
      df_resampled = df.resample(freq).agg(resampling_rules)
      # Drop rows with no data, which can occur in empty time periods.
      resampled[name] = df_resampled.dropna()

    print('--- BTC Data Resampling Complete ---')

    return resampled

def calculate_log_and_diff(df, col):
    '''
    Calculates the log and log returns for a specific column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        col (str): The col for which to calculate log and log returns.

    Returns:
        pd.DataFrame: The DataFrame with a new 'log_{col}' and 'log_returns_{col}' column.
    '''
    df.loc[:, f'log_{col}'] = np.log(df[col])
    df.loc[:, f'log_returns_{col}'] = df[f'log_{col}'].diff()

    return df

def calculate_price_range_pct(df):
    '''
    Calculates the price range as a percentage of the opening price.

    This feature is a measure of volatility. It is calculated as:
    (high - low) / open

    Args:
        df (pd.DataFrame): The input DataFrame, which must contain 'high',
            'low', and 'open' columns.

    Returns:
        pd.DataFrame: The DataFrame with a new 'price_range_pct' column.
    '''
    df['price_range_pct'] = (df['high'] - df['low']) / df['open']
    
    return df

def prepare_feature_ds(train_ds, valid_ds, test_ds, 
                       cols, fedfunds_path, m2sl_path):
    '''
    Orchestrates the entire feature engineering pipeline by consolidating data splits,
    adding a comprehensive suite of features, and then splitting the data back.
    
    This function implements the 'join-then-transform-then-split' best practice
    to ensure that features are calculated on a continuous time series, preventing
    data leakage and lookahead bias. It performs the following steps:
    1.  Concatenates the raw train, validation, and test sets.
    2.  Engineers a comprehensive set of features on the single, continuous
        DataFrame, including:
        -   Log returns for specified OHLCV columns.
        -   Price range percentages.
        -   A suite of technical indicators (SMA, RSI, MACD, ATR, OBV).
        -   Relationship-based features (e.g., Price vs. SMA).
    3.  Handles any resulting NaN or infinite values.
    4.  Splits the processed DataFrame back into train, validation, and test
        sets based on the original index boundaries.

    Args:
        train_ds (pd.DataFrame): The raw training data.
        valid_ds (pd.DataFrame): The raw validation data.
        test_ds (pd.DataFrame): The raw test data.
        cols (list): A list of column names (e.g., ['open', 'high', 'low',
            'close', 'volume']) to calculate log returns for.
        time_unit (str): The time unit of the dataset ('Hour' or 'Day').
        fedfunds_path (str): File path to the fedfunds.csv data.
        m2sl_path (str): File path to the m2sl.csv data.

    Returns:
        tuple: A tuple containing the three prepared DataFrames:
            (train_prep_ds, valid_prep_ds, test_prep_ds)
    '''
    print_header('Feature Engineering and Dataset Preparation')
    train_end_idx = train_ds.index[-1]
    valid_end_idx = valid_ds.index[-1]

    print('Step 1: Concatenating data splits...')
    df_full = pd.concat([train_ds, valid_ds, test_ds])
    print(f'- Combined train, validation, and test sets into a single DataFrame. Full shape: {df_full.shape}')
    
    print('\nStep 2: Engineering new features...')
    for col in cols:
        df_full = calculate_log_and_diff(df_full, col)
    print(f'- Calculated log returns for {cols}.')
    df_full = calculate_price_range_pct(df_full)
    print(f'- Calculated the high-low price range as a percentage of the open price.')

    df_full = _add_technical_indicators(df_full)
    print(f'- Calculated technical indicators.')

    df_full = _add_macroeconomic_features(df_full, fedfunds_path, m2sl_path)
    print(f'- Added macroeconomic features.')

    print('\nStep 3: Handling missing and infinite values...')
    # log(p(t)/p(t-1): If p(t) or p(t-1) is zero, the result is log(0) = -inf, or log(p(t) / 0) = inf.
    # dropna() does not remove infinity.
    df_full.replace([np.inf, -np.inf], np.nan, inplace=True)
    print(f'- Replaced {df_full.isin([np.inf, -np.inf]).sum().sum()} infinite values (inf, -inf) with NaN.')
    df_full.ffill(inplace=True)
    print('- Forward-filled missing values to maintain data continuity.')
    print(f'- Shape before dropping NaNs: {df_full.shape}')
    df_full.dropna(inplace=True)
    print(f'- Shape after dropping NaNs: {df_full.shape}')
    print('- Dropped initial rows containing NaNs that resulted from feature calculations.')

    print('\nStep 4: Splitting into final datasets...')
    train_prep_ds= df_full.loc[:train_end_idx]
    # .iloc[1:] prevents the last point of one set from being the first of the next.
    valid_prep_ds= df_full.loc[train_end_idx:valid_end_idx].iloc[1:] 
    test_prep_ds= df_full.loc[valid_end_idx:].iloc[1:]

    print('- Split the single processed DataFrame back into train, validation, and test sets.')
    print(f'- Train shape: {train_prep_ds.shape}')
    print(f'- Valid shape: {valid_prep_ds.shape}')
    print(f'- Test shape: {test_prep_ds.shape}')

    print_header('Feature Preparation Complete')

    return train_prep_ds, valid_prep_ds, test_prep_ds

def _add_technical_indicators(df, sma_period=50, rsi_period=14, atr_period=14):
    '''
    Calculates and adds a suite of technical indicators and engineered features.

    This includes indicators for trend, momentum, volatility, and volume, as well
    as relationship-based features like price vs. SMA and MACD crossovers.

    Args:
        df (pd.DataFrame): DataFrame with 'high', 'low', 'close', and 'volume' columns.
        sma_period (int): The period window for the Simple Moving Average. Default is 50.
        rsi_period (int): The period window for the Relative Strength Index. Default is 14.
        atr_period (int): The period for the Average True Range. Default is 14.

    Returns:
        pd.DataFrame: The DataFrame with all new feature columns added.
    '''
    print('- Calculated Simple Moving Average (SMA) for trend.')
    df[f'sma_{sma_period}'] = TA.SMA(df, period=sma_period)

    print('- Calculated Relative Strength Index (RSI) for momentum.')
    df[f'rsi_{rsi_period}'] = TA.RSI(df, period=rsi_period)

    print('- Calculated Average True Range (ATR) for volatility.')
    df[f'atr_{atr_period}'] = TA.ATR(df, period=atr_period)

    print('- Calculated On-Balance Volume (OBV) for volume pressure.')
    # A rising OBV shows that volume is flowing into the asset.
    df['obv'] = TA.OBV(df)

    print('- Calculated Moving Average Convergence Divergence (MACD).')
    # Captures changes in the strength, direction, momentum, and duration of a trend.
    df_macd = TA.MACD(df, period_fast=12, period_slow=26, signal=9)

    df['macd_12_26_9'] = df_macd['MACD']
    df['macd_signal_12_26_9'] = df_macd['SIGNAL']
    df['macd_hist_12_26_9'] = df['macd_12_26_9'] - df['macd_signal_12_26_9']

    print('- Engineered relationship features')
    # Price position relative to its long-term trend.
    df['price_vs_sma'] = df['close'] / df[f'sma_{sma_period}']

    # Categorical state of RSI (Oversold, Neutral, Overbought).
    rsi_bins = [0, 30, 70, 100]
    rsi_labels = ['Oversold', 'Neutral', 'Overbought']
    df['rsi_state'] = pd.cut(df[f'rsi_{rsi_period}'], bins=rsi_bins, labels=rsi_labels, right=False)

    # Binary signal for MACD line crossing its signal line.
    df['macd_crossover'] = (df['macd_12_26_9'] > df['macd_signal_12_26_9']).astype(int)

    return df

def _add_macroeconomic_features(df, fedfunds_path, m2sl_path):
    '''
    Loads, processes, and merges macroeconomic data into the main DataFrame.

    This function takes paths to the Fed Funds Effective Rate and M2 Money Supply CSVs,
    processes them, and joins them to the main daily/weekly-frequency DataFrame using a
    forward-fill method to handle the monthly-to-daily/weekly frequency mismatch. It
    also engineers a year-over-year M2 supply feature.

    Args:
        df (pd.DataFrame): DataFrame with 'high', 'low', 'close', and 'volume' columns.
        fedfunds_path (str): The file path to the FEDFUNDS.csv file.
        m2sl_path (str): The file path to the M2SL.csv file.

    Returns:
        pd.DataFrame: The main DataFrame with added macroeconomic features.       
    '''    
    df_fedfunds = pd.read_csv(fedfunds_path)
    df_m2sl = pd.read_csv(m2sl_path)

    df_fedfunds['observation_date'] = pd.to_datetime(df_fedfunds['observation_date'])
    df_fedfunds.set_index('observation_date', inplace=True)
    df_fedfunds.rename(columns={'FEDFUNDS': 'fed_rate'}, inplace=True)

    df_m2sl['observation_date'] = pd.to_datetime(df_m2sl['observation_date'])
    df_m2sl.set_index('observation_date', inplace=True)
    df_m2sl.rename(columns={'M2SL': 'm2_supply'}, inplace=True)

    df_macro = df_fedfunds.join(df_m2sl, how='inner')

    df_macro['m2_supply_mom_pct'] = df_macro['m2_supply'].pct_change(1) * 100
    # Lag the feature by 1 month to prevent lookahead bias.
    df_macro['m2_supply_mom_pct'] = df_macro['m2_supply_mom_pct'].shift(1)
    df_macro = df_macro.drop(columns=['m2_supply'])

    print('- Macro data loaded and prepared.')
    
    df_macro['month_key'] = df_macro.index.to_period('M')
    df_macro.set_index('month_key', inplace=True)

    df = df.join(df_macro, on=df.index.to_period('M'))

    cols_to_fill = ['fed_rate', 'm2_supply_mom_pct']
    df[cols_to_fill] = df[cols_to_fill].ffill()
    df[cols_to_fill] = df[cols_to_fill].fillna(0)

    df = df.drop(columns=['key_0'])

    print('- Macro data merged.')

    return df