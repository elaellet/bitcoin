from pathlib import Path

import pandas as pd

def load_btc_ds(path, idx_col=None, parse_dates=False):
    '''
    Loads a BTC dataset from a specified CSV file path into a Pandas DataFrame.
     
    Args:
        path (str or pathlib.Path): The path to the CSV file.
        idx_col (str, optional): The column to set as the DataFrame index. Defaults to None.
        parse_dates (bool, optional): Whether to parse the index column as dates. Defaults to False.

    Returns:
        pd.DataFrame: The loaded dataset.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
    '''
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f'Error: The file was not found at {path}')
    
    print(f'\nLoading dataset from: {path.name}...')

    read_args = dict()
    if idx_col:
        read_args['index_col'] = idx_col # Set the index to unique timestamps.
    if parse_dates:
        read_args['parse_dates'] = [idx_col] if idx_col else True # Force datetime parsing(e.g., DatetimeIndex for asfreq()).

    print('Dataset loaded successfully.')

    return pd.read_csv(path, **read_args)