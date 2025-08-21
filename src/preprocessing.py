import numpy as np

def resample_btc_data(df):
    '''
    Resamples the 1-minute BTC data to various timeframes (hourly, daily, weekly, monthly).

    This function aggregates the OHLCV (Open, High, Low, Close, Volume) data for
    each new timeframe. The aggregation rules are:
    - 'open': First price in the period.
    - 'high': Maximum price in the period.
    - 'low': Minimum price in theperiod.
    - 'close': Last price in the period.
    - 'volume': Sum of all volume in the period.

    Args:
        df (pd.DataFrame): The cleaned, 1-minute frequency DataFrame with a DatetimeIndex.

    Returns:
        dict[str, pd.DataFrame]: A dictionary where keys are the resampling frequencies
                                 (e.g., 'hourly', 'daily') and values are the
                                 corresponding resampled DataFrames.
    '''
    # Resampling and aggregation smooths out high-frequency noise and reveals underlying trends.
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
      print(f'Resampling to {name} frequency...')
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
    # Log transformation stabilizes the variance.
    # Differencing stabilizes the mean by removing or reducing the trend and seasonality.
    df.loc[:, f'log_{col}'] = np.log(df[col])
    df.loc[:, f'log_returns_{col}'] = df[f'log_{col}'].diff()

    return df

def calculate_returns(df, col):
    '''
    Calculates the percentage returns for a specific column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        col (str): The column for which to calculate returns.

    Returns:
        pd.DataFrame: The DataFrame with a new 'returns_{col}' column.
    '''
    # Create a temporary series where 0 is replaced by NaN.
    # dropna() does not remove infinity.
    # 0/n -> inf, 0/0 -> NaN
    df_copy = df[col].replace(0, np.nan)

    # Percentage return is more effective than a simple log transform for LSTMs because it directly represents the period-over-period change.
    df.loc[:, f'returns_{col}'] = df_copy.pct_change(fill_method=None)

    return df