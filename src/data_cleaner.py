import numpy as np
import pandas as pd

from .utils import print_header

def _handle_missing_timestamps(df_prep):
    '''Resamples to a 1-minute frequency and handles gaps.'''
    print('Step 1: Resampling to 1-minute frequency to identify gaps...')
    # Create a new dataframe where the index is a complete 1-minute timeline.
    df_resampled = df_prep.asfreq('min') 

    # Find all rows where the 'close' price is NaN, which indicates a missing timestamp.
    missing_rows = df_resampled['close'].isna().sum()
    missing_percentage = missing_rows / len(df_resampled) * 100
    print(f'- Identified {missing_percentage:.3f}% missing minutes.')

    # Specific fix: Remove the known gap in March 2025 (2025-03-15 00:01-19:20).
    # The risk of misleading a model by introducting synthetic data is greater than the cost of deleting a single day from the dataset.
    date_to_remove = pd.to_datetime('2025-03-15').date()
    df_cleaned = df_resampled[df_resampled.index.date != date_to_remove].copy()
    print(f'- Removed known data gap on {date_to_remove}.')
    
    # Check for any remaining null values.
    if df_cleaned.isna().any().any():
        print('Warning: Null values still exist after cleaning. Further handling may be needed.')

    return df_cleaned

def _verify_data_integrity(df_cleaned):
    '''Checks for logical inconsistencies like negative prices or zero volume.'''
    print('\nStep 2: Verifying data integrity...')
    
    anomalies = {
        'negative_prices': (df_cleaned[['open', 'high', 'low', 'close']] < 0).any().any(),
        'high_less_than_low': (df_cleaned['high'] < df_cleaned['low']).any(),
        'zero_trading_volume': (df_cleaned['volume'] == 0).sum()
    }

    print(f"- Negative prices found: {anomalies['negative_prices']}")
    print(f"- High < Low instances: {anomalies['high_less_than_low']}")
    print(f"- Minutes with zero volume: {anomalies['zero_trading_volume']}")

def _engineer_features(df_cleaned):
    '''Engineers new features required for analysis and modeling.'''
    print('\nStep 3: Engineering new features...')

    df_cleaned.loc[:, 'is_zero_volume'] = (df_cleaned['volume'] == 0).astype(int)

    # Percentage change, accounting for time gaps.
    df_cleaned.loc[:, 'pct_change'] = df_cleaned['close'].pct_change()
    # Time difference between consecutive rows in minutes.
    # Find locations where the time gap is larger than a normal interval (i.e., > 1 minute) and set the pct_change at that point to NaN.
    # This neutralizes artifical spikes/crashes produced by deleting a single day of the data from the dataset.    
    time_diff = df_cleaned.index.to_series().diff().dt.total_seconds() / 60.0
    gap_locations = time_diff[time_diff > 1.0].index
    df_cleaned.loc[gap_locations, 'pct_change'] = np.nan

    # Intra-minute spread.
    df_cleaned.loc[:, 'intra_minute_spread'] = (df_cleaned['high'] - df_cleaned['low']) / df_cleaned['open']

    print('- Features engineered: is_zero_volume, pct_change, intra_minute_spread.')

    return df_cleaned

def _impute_outliers(df_cleaned):
    '''Identifies and imputes outlier data points based on extreme changes and spreads.'''
    print('\nStep 4: Identifying and imputing outliers...')

    # Define outlier thresholds based on the 99.99th percentile.
    # That is, 99.99% of all values are smaller than this value.
    pct_change_threshold = df_cleaned['pct_change'].abs().quantile(0.9999)
    spread_threshold = df_cleaned['intra_minute_spread'].abs().quantile(0.9999)

    # Identify 5% of all volumes smaller than the threshold.
    non_zero_volumes = df_cleaned[df_cleaned['volume'] > 0.0]['volume']
    volume_threshold = non_zero_volumes.quantile(0.05)

    # Artificial spikes/crashes from the gap are ignored.
    suspicious_pct_indices = set(df_cleaned[df_cleaned['pct_change'].abs() > pct_change_threshold].index)
    suspicious_spread_indices = set(df_cleaned[df_cleaned['intra_minute_spread'].abs() > spread_threshold].index)

    indices_with_anomalies = sorted(list(set(suspicious_pct_indices) | set(suspicious_spread_indices)))
    if not indices_with_anomalies:
        print('- No significant outliers found to impute.')
        return df_cleaned    

    print(f'- Found {len(indices_with_anomalies)} potential outliers based on extreme price moves.')

    indices_to_impute = list()

    for index in indices_with_anomalies:
        try:
            # Get the integer location of the index.
            loc = df_cleaned.index.get_loc(index)
            if loc == 0 or loc == len(df_cleaned) - 1:
                continue # Skip if it's the first or last row.

            snapshot = df_cleaned.iloc[loc-1:loc+2, 1:6]

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
            reasons = list()

            if is_reverted and (is_volume_zero or is_volume_low):
                classification = 'High Confidence Error'
                reasons.append('Price reverted immediately')
                reasons.append('Volume was zero or abnormally low')
            elif is_reverted:
                classification = 'Likely Error'
                reasons.append('Price reverted immediately')
            elif is_volume_zero or is_volume_low:
                classification = 'Suspicious'
                reasons.append('Volume was zero or abnormally low for a large price move')

            if classification != 'Possible Real Market Event':
                indices_to_impute.append(pd.to_datetime(snapshot.index.values[1]))

            # Print report
            # print(f"--- Investigation Report for Index: {idx} ---")
            # print(snapshot)
            # print("\nAnalysis:")
            # print(f"  - Price Reverted: {is_reverted}")
            # print(f"  - Volume Zero: {is_volume_zero}")
            # print(f"  - Volume Abnormally Low: {is_volume_low}")
            # print(f"  - Classification: **{classification}**")
            # if reasons:
            #     print(f"  - Justification: {', '.join(reasons)}.")
            # print("-" * 60, "\n")
        except Exception as e:
            print(f'- Could not process index {index}. Error: {e}')

    if not indices_to_impute:
        print('- No significant rows found to impute.')
        return df_cleaned    
    
    print(f'- Found {len(indices_to_impute)} suspicious rows to forward-fill.')
    cols_to_fill = ['open', 'high', 'low', 'close', 'volume'] # Use .loc to select the rows and columns to fill.
    df_cleaned.loc[indices_to_impute, cols_to_fill] = np.nan # First set to NaN.
    df_cleaned.loc[:, cols_to_fill] = df_cleaned[cols_to_fill].ffill() # Then forward-fill.
    print(f'- Imputed {len(indices_to_impute)} rows using forward-fill.')

    return df_cleaned

def clean_btc_df(df_raw):
    '''
    Performs a comprehensive cleaning and preprocessing of the raw BTC time-series data.
    
    This function orchestrates a multi-step pipeline designed to prepare the dataset
    for rigorous time-series analysis and machine learning modeling. The process includes:
    
    1.  **Standardizing Format:** Renames columns for clarity and converts the UNIX
        timestamp into a proper DatetimeIndex.
    2.  **Handling Missing Data:** Resamples the data to a consistent 1-minute frequency,
        explicitly identifying and handling known time gaps.
    3.  **Verifying Integrity:** Checks for logical impossibilities in the data, such as
        negative prices or a high price being lower than a low price.
    4.  **Feature Engineering:** Creates new, informative features, including flags for
        zero-volume periods, minute-over-minute percentage change, and intra-minute price spread.
    5.  **Outlier Imputation:** Identifies and corrects anomalous data points that are likely
        errors rather than real market movements, using forward-filling.
        
    Args:
        df_raw (pd.DataFrame): The raw BTC dataset, expected to have columns for
                               timestamp, open, high, low, close, and volume.

    Returns:
        pd.DataFrame: A cleaned and fully preprocessed DataFrame ready for analysis.
    '''
    print_header('Cleaning and Preprocessing BTC Data')
   
    '''Creates "date" column and set it to index for readability, time-based indexing and analysis'''
    df_prep = df_raw.set_axis(['timestamp', 'open', 'high', 'low', 'close', 'volume'], axis=1)
    df_prep['date'] = pd.to_datetime(df_prep['timestamp'], unit='s')
    df_prep = df_prep.set_index('date')
    
    df_cleaned = _handle_missing_timestamps(df_prep)
    _verify_data_integrity(df_cleaned)
    df_cleaned = _engineer_features(df_cleaned)
    df_cleaned = _impute_outliers(df_cleaned)
    
    print_header('BTC Data Cleaning and Preprocessing Complete')

    return df_cleaned