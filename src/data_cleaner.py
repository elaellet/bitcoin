import numpy as np
import pandas as pd

from .utils import print_header

def _handle_missing_timestamps(prep_ds):
    '''Resamples to a 1-minute frequency and handles gaps.'''
    print('Step 1: Resampling to 1-minute frequency to identify gaps...')
    resampled_ds = prep_ds.asfreq('min') 

    missing_rows = resampled_ds['close'].isna().sum()
    missing_pct = missing_rows / len(resampled_ds) * 100
    print(f'- Identified {missing_pct:.3f}% missing minutes.')

    date_to_remove = pd.to_datetime('2025-03-15').date()
    cleaned_ds = resampled_ds[resampled_ds.index.date != date_to_remove].copy()
    print(f'- Removed known data gap (1,440) on {date_to_remove}.')
    
    if cleaned_ds.isna().values.any():
        print('Warning: NaN values still exist after cleaning. Further handling may be needed.')

    return cleaned_ds

def _verify_data_integrity(cleaned_ds):
    '''Checks for logical inconsistencies, such as negative prices or zero volume.'''
    print('\nStep 2: Verifying data integrity...')
    
    anomalies = {
        'negative_prices': (cleaned_ds[['open', 'high', 'low', 'close']] < 0).sum().sum(),
        'high_less_than_low': (cleaned_ds['high'] < cleaned_ds['low']).sum(),
        'zero_trading_volume': (cleaned_ds['volume'] == 0).sum()
    }

    print(f'- Negative prices found: {anomalies["negative_prices"]}')
    print(f'- High < Low instances: {anomalies["high_less_than_low"]}')
    print(f'- Minutes with zero volume: {anomalies["zero_trading_volume"]}')

def _engineer_features(cleaned_ds):
    '''Engineers new features required for analysis and modeling.'''
    print('\nStep 3: Engineering new features...')

    cleaned_ds.loc[:, 'is_zero_volume'] = (cleaned_ds['volume'] == 0).astype(int)

    # Percentage change, accounting for time gaps.
    cleaned_ds.loc[:, 'pct_change'] = cleaned_ds['close'].pct_change()
    # Time difference between consecutive rows in minutes.
    # Find locations where the time gap is larger than a normal interval (i.e., > 1 minute) and set the pct_change at that point to NaN.
    # This neutralizes artificial spikes/crashes that occur when a single day of data is deleted from the dataset.   
    time_diff = cleaned_ds.index.to_series().diff().dt.total_seconds() / 60.0
    gap_locs = time_diff[time_diff > 1.0].index
    cleaned_ds.loc[gap_locs, 'pct_change'] = np.nan

    # Intra-minute spread.
    cleaned_ds.loc[:, 'intra_minute_spread'] = (cleaned_ds['high'] - cleaned_ds['low']) / cleaned_ds['open']

    print('- Features engineered: is_zero_volume, pct_change, intra_minute_spread.')

    return cleaned_ds

def _impute_outliers(cleaned_ds):
    '''Identifies and imputes outlier data points based on extreme changes and spreads.'''
    print('\nStep 4: Identifying and imputing outliers...')

    # Define outlier thresholds based on the 99.99th percentile.
    # That is, 99.99% of all values are smaller than this value.
    pct_change_threshold = cleaned_ds['pct_change'].abs().quantile(0.9999)
    spread_threshold = cleaned_ds['intra_minute_spread'].abs().quantile(0.9999)

    # Identify 5% of all volumes smaller than the threshold.
    non_zero_volumes = cleaned_ds[cleaned_ds['volume'] > 0.0]['volume']
    volume_threshold = non_zero_volumes.quantile(0.05)

    # Artificial spikes/crashes from the gap are ignored.
    suspicious_pct_indices = set(cleaned_ds[cleaned_ds['pct_change'].abs() > pct_change_threshold].index)
    suspicious_spread_indices = set(cleaned_ds[cleaned_ds['intra_minute_spread'].abs() > spread_threshold].index)

    indices_with_anomalies = sorted(list(set(suspicious_pct_indices) | set(suspicious_spread_indices)))
    if not indices_with_anomalies:
        print('- No significant potential outliers found.')
        return cleaned_ds

    print(f'- Found {len(indices_with_anomalies)} potential outliers based on extreme price moves.')

    indices_to_impute = list()

    for idx in indices_with_anomalies:
        try:
            # Get the integer location of the index.
            loc = cleaned_ds.index.get_loc(idx)
            if loc == 0 or loc == len(cleaned_ds) - 1:
                continue # Skip if it's the first or last row.

            snapshot = cleaned_ds.iloc[loc-1:loc+2, 1:6]

            # Is it a one-minute event (does the price revert)?
            price_t_minus_1 = snapshot.iloc[0]['close']
            price_t = snapshot.iloc[1]['close']
            price_t_plus_1 = snapshot.iloc[2]['close']

            spike_magnitude = abs(price_t - price_t_minus_1)
            reversion_magnitude = abs(price_t - price_t_plus_1)

            is_reverted = (reversion_magnitude / spike_magnitude) > 0.75 if spike_magnitude > 0 else False

            # Is the volume zero or abnormally low?
            volume_t = snapshot.iloc[1]['volume']
            is_volume_zero = (volume_t == 0)
            is_volume_low = (volume_t < volume_threshold)

            # Classify each flagged point.
            classification = 'Possible Real Market Event'

            if is_reverted and (is_volume_zero or is_volume_low):
                classification = 'High Confidence Error'
            elif is_reverted:
                classification = 'Likely Error'
            elif is_volume_zero or is_volume_low:
                classification = 'Suspicious'

            if classification in ['High Confidence Error', 'Likely Error']:
                index_to_remove = pd.to_datetime(snapshot.index.values[1])
                indices_to_impute.append(index_to_remove)

        except Exception as e:
            print(f'- Could not process index {idx}. Error: {e}')

    if not indices_to_impute:
        print('- No significant rows found to impute.')
        return cleaned_ds    
    
    print(f'- Found {len(indices_to_impute)} suspicious rows to forward-fill.')
    cols_to_fill = ['open', 'high', 'low', 'close', 'volume'] # Use .loc to select the rows and columns to fill.
    cleaned_ds.loc[indices_to_impute, cols_to_fill] = np.nan # First set to NaN.
    cleaned_ds.loc[:, cols_to_fill] = cleaned_ds[cols_to_fill].ffill() # Then forward-fill.
    print(f'- Imputed {len(indices_to_impute)} rows using forward-fill.')

    return cleaned_ds

def fill_time_series_gaps(ds, freq):
    '''
    Fills gaps in a time series dataset using linear interpolation.

    This function ensures the dataset has a continuous DatetimeIndex by
    creating a full date range based on the data's start and end times. It
    reindexes the dataset to this new range, which introduces NaN values for
    any missing timestamps, and then fills these gaps using the 'linear' method.

    Args:
        ds (pd.DataFrame): The input dataset, which must have a DatetimeIndex
            that may contain gaps.
        freq (str): The expected frequency of the time series as a string
            (e.g., 'min', 'D', 'H') to build the complete index.

    Returns:
        pd.DataFrame: A new dataset with a continuous index and no
            missing time steps.
    '''
    full_time_range = pd.date_range(start=ds.index.min(), end=ds.index.max(), freq=freq)
    reindexed_ds = ds.reindex(full_time_range)
    imputed_ds = reindexed_ds.interpolate(method='linear')

    return imputed_ds

def clean_btc_data(raw_ds):
    '''
    Performs a comprehensive cleaning and preprocessing of the raw BTC time-series data.
    
    This function orchestrates a multi-step pipeline designed to prepare the dataset
    for rigorous time-series analysis and machine learning modeling. The process includes:
    
    1.  Standardizing Format: Renames columns for clarity and converts the UNIX
        timestamp into a proper DatetimeIndex.
    2.  Handling Missing Data: Resamples the data to a consistent 1-minute frequency,
        explicitly identifying and handling known time gaps.
    3.  Verifying Integrity: Checks for logical impossibilities in the data, such as
        negative prices or a high price being lower than a low price.
    4.  Feature Engineering: Creates new, informative features, including flags for
        zero-volume periods, minute-over-minute percentage change, and intra-minute price spread.
    5.  Outlier Imputation: Identifies and corrects anomalous data points that are likely
        errors rather than real market movements, using forward-filling.

    Args:
        raw_ds (pd.DataFrame): The raw BTC dataset, expected to have columns for
                               timestamp, open, high, low, close, and volume.

    Returns:
        pd.DataFrame: A cleaned and fully preprocessed DataFrame ready for analysis.
    '''
    print()
    print_header('Cleaning and Preprocessing BTC Data')
   
    '''Creates "date" column and sets it to index for readability, time-based indexing, and analysis'''
    prep_ds = raw_ds.set_axis(['timestamp', 'open', 'high', 'low', 'close', 'volume'], axis=1)
    prep_ds['date'] = pd.to_datetime(prep_ds['timestamp'], unit='s')
    prep_ds.set_index('date', inplace=True)
    
    cleaned_ds = _handle_missing_timestamps(prep_ds)
    _verify_data_integrity(cleaned_ds)
    cleaned_ds = _engineer_features(cleaned_ds)
    cleaned_ds = _impute_outliers(cleaned_ds)
    
    print_header('BTC Data Cleaning and Preprocessing Complete')

    return cleaned_ds