import pandas as pd
from pathlib import Path

PROJECT_DATA = Path('/content/drive/MyDrive/projects/bitcoin/data')
CSV_RAW = PROJECT_DATA / 'raw/bitcoin_274_raw.csv'
CSV_PROC = PROJECT_DATA / 'processed/bitcoin_274_proc.csv'
CSV_CLEANED = PROJECT_DATA / 'cleaned/bitcoin_274_cleaned.csv'


def load_raw_dataset():
  return pd.read_csv(CSV_RAW)

def load_proc_dataset():
  dataset = pd.read_csv(CSV_PROC,
                        index_col='date',
                        parse_dates=['date'])

  return dataset

def load_cleaned_dataset(CSV_CLEANED):
  dataset = pd.read_csv(CSV_CLEANED)
  dataset = dataset.set_index('date')

  return dataset

# Resampling and aggregation smooths out high-frequency noise and reveals underlying trends.
def resample_and_aggregate_dataset(dataset):
  ohlcv_agg = {
    'open': 'first', # First price of a timeframe
    'high': 'max',   # A timeframe high
    'low': 'min',    # A timeframe low
    'close': 'last', # Last price of a timeframe
    'volume': 'sum'  # Total timeframe trading volume
  }

  timeframes = {'hourly': 'h', 'daily': 'D', 'weekly': 'W', 'monthly': 'ME'}

  dataset_ohlcv = {}

  for period, freq in timeframes.items():
    resampled_df = dataset.resample(freq).agg(ohlcv_agg)
    # Drop rows with no data, which can occur in empty time periods.
    dataset_ohlcv[period] = resampled_df.dropna()

  return dataset_ohlcv