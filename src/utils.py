import hashlib
from pathlib import Path

import pandas as pd
import yaml

def print_header(title):
    '''
    Prints a formatted, standardized header to the console.

    Args:
        title (str): The text to display in the header.
    '''
    header_line = '=' * (len(title) + 4)
    print(f'{header_line}')
    print(f'= {title} =')
    print(f'{header_line}')

def manage_checksum(file_path, metadata_dir, main_log_path, verify=False):
    '''
    Generates, verifies, and logs SHA-256 checksums for a given file.

    This function replicates the functionality of the following shell commands:
    1. sha256sum [file] > [checksum_file]  (Generation)
    2. sha256sum [file] | diff - [checksum_file] (Verification)
    3. echo "[file]: $(sha256sum [file] | cut -d' ' -f 1)" >> [main_log_path] (Logging)

    Args:
        file_path (str or pathlib.Path): The full path to the file to be processed.
        metadata_dir (str or pathlib.Path): Directory to store individual checksum files.
        main_log_path (str or pathlib.Path): The full path to a main log file to append results.
                                        If provided in generation mode, the checksum is appended.
        verify (bool, optional): If True, verifies the file against its existing
                                 checksum file instead of generating a new one.
                                 Defaults to False.

    Returns:
        tuple: A tuple containing (bool, str).
               - The boolean is True for success (generation or successful verification)
                 and False for a verification failure.
               - The string is the calculated SHA-256 checksum of the file.

    Raises:
        FileNotFoundError: If the specified file_path does not exist.
    '''
    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(f'Error: The file was not found at \'{file_path}\'')

    sha256_hash = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for byte_block in iter(lambda: f.read(4096), b''):
            sha256_hash.update(byte_block)

    curr_checksum = sha256_hash.hexdigest()

    metadata_dir = Path(metadata_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    indiv_checksum_path = metadata_dir / f'{file_path.stem}.txt'

    if verify:
        if not indiv_checksum_path.exists():
            print(f'Verification failed: Checksum file not found at \'{indiv_checksum_path}\'')
            return False, curr_checksum

        saved_checksum = indiv_checksum_path.read_text().split()[0]
        if curr_checksum == saved_checksum:
            print(f'Verification successful for \'{file_path.name}\': OK')
            return True, curr_checksum
        else:
            print(f'Verification failed for \'{file_path.name}\':')
            print(f'- Current:  {curr_checksum}')
            print(f'- Expected: {saved_checksum}')
            return False, curr_checksum
    else:
        main_log_path = Path(main_log_path)
        
        if main_log_path.is_file():
            with open(main_log_path, 'r') as f:
                if curr_checksum in f.read():
                    print(f'Checksum \'{curr_checksum}\' already exists in main log. Skipping file writes.')
                    return True, curr_checksum        

        indiv_checksum_path.write_text(f'{curr_checksum}  {file_path.name}\n')
        print(f'Checksum for \'{file_path.name}\' saved to: \'{indiv_checksum_path}\'')

        with open(main_log_path, 'a') as f:
            f.write(f'{file_path.name}: {curr_checksum}\n')
        print(f'Appended checksum to main log: \'{main_log_path}\'')
        
        return True, curr_checksum

def save_ds(path, ds):
    '''
    Saves content to a specified file path.

    This function ensures the parent directory for the given file path exists
    and then writes the provided content to the file, overwriting it if it
    already exists.

    Args:
        ds (pd.DataFrame): The Pandas DataFrame to be saved.
        path (str or pathlib.Path): The path to the dataset file.
    '''
    try:
        path = Path(path)    
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            print(f'The dataset already exists at \'{path}\'.')
            return

        ds.to_csv(path, index=False)

        print(f'Successfully saved the dataset to \'{path}\'.')
           
    except Exception as e:
        print(f'An error occurred while saving the file: {e}')
        
def create_gitignore(path='./.gitignore'):
    '''
    Creates a .gitignore file at the specified path with a default set of rules.

    Args:
        path (str or pathlib.Path): The full path, including the filename, where the
                                    .gitignore file should be created.
                                    Defaults to './.gitignore' in the current directory.
    '''
    # Using a raw string (r''') to avoid issues with backslashes.
    gitignore_content = r'''
    # Environment and secrets
.env

    # Data folders
    # Commit the structure and metadata, but not the raw/cleaned data files.
data/raw/
data/cleaned/

    # Python cache and artifacts
__pycache__/
*.pyc
*.pkl

    # Jupyter Notebook checkpoints
.ipynb_checkpoints/

    # OS-specific files
.DS_Store
    '''

    try:
        with open(path, 'w') as f:
            f.write(gitignore_content.strip())
        print(f'Success: .gitignore file created at \'{path}\'')
    except IOError as e:
        # Handle potential file system errors (e.g., permission denied)
        print(f'Error: Could not create .gitignore file at \'{path}\'.')
        print(f'Reason: {e}')
        
def create_metadata_yaml(
    ds,
    raw_data_path,
    cleaned_data_path,
    metadata_path,
    checksum,
    timestamp_col='Timestamp',
    key_variables_notes='\nOpen/High/Low/Close prices\nVolume (BTC/USD)\nTimestamp (Unix epoch)\n',
    original_source_url='https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data'
):
    '''
    Appends or creates a central `versions.yaml` file with versioning metadata for a new dataset version.

    Args:
        ds (pd.DataFrame): The DataFrame to analyze.
        raw_data_path (str or Path): Path to the raw dataset file (e.g., 'data/raw/btc.csv').
        cleaned_data_path (str or Path): Path to the processed dataset file.
        metadata_path (str or Path): Path to the YAML file where metadata will be saved.
        checksum (str): The SHA-256 checksum of the raw data file.
        timestamp_col (str, optional): The name of the timestamp column. Defaults to 'Timestamp'.
        key_variables_notes (str, optional): A string describing the key variables.
        original_source_url (str, optional): The URL of the original data source.
    '''
    try:
        raw_path = Path(raw_data_path)
        metadata_path = Path(metadata_path)
        dataset_key = raw_path.stem

        # Load existing versions file or create a new dictionary if it doesn't exist.
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                all_metadata = yaml.safe_load(f) or {}
        else:
            all_metadata = {}        

        timestamps = pd.to_datetime(ds[timestamp_col], unit='s')
        frozen_date = timestamps.max().strftime('%Y-%m-%d %H:%M:%S')
        time_range_start = timestamps.min().strftime('%Y-%m-%d %H:%M:%S')
        num_rows, num_cols = ds.shape
        
        metadata = {
            'dataset_name': raw_path.name,
            'frozen_date': frozen_date,
            'time_range': {
                'start': time_range_start,
                'end': frozen_date
            },
            'dimensions': {
                'rows': num_rows,
                'columns': num_cols
            },
            'columns': {col: str(ds[col].dtype) for col in ds.columns},
            'key_variables': key_variables_notes,
            'paths': {
                'raw_data': str(raw_path),
                'cleaned_data': str(cleaned_data_path)
            },
            'source': {
                'name': 'Kaggle API',
                'url': original_source_url
            },
            'checksum': {
                'algorithm': 'SHA-256',
                'value': checksum
            }
        }

        if dataset_key not in all_metadata:
            all_metadata[dataset_key] = []

        # Check if a version with this checksum already exists.
        existing_checksums = {
            version['checksum']['value'] for version in all_metadata[dataset_key]
        }
        if checksum in existing_checksums:
            print(f'Info: Version with checksum {checksum[:7]}... already exists for \'{dataset_key}\'. Skipping.')
            return
        
        all_metadata[dataset_key].append(metadata)

        with open(metadata_path, 'w') as f:
            yaml.dump(all_metadata, f, sort_keys=False, indent=2)

        print(f'Successfully added new version for \'{dataset_key}\' to \'{metadata_path}\'')

    except FileNotFoundError:
        print(f'Error: The file or directory for the output path \'{metadata_path}\' was not found.')
    except KeyError:
        print(f'Error: Timestamp column \'{timestamp_col}\' not found in the DataFrame.')
    except Exception as e:
        print(f'An unexpected error occurred: {e}')


def load_config(file_path: str):
    '''
    Loads a configuration from a YAML file.

    Args:
        file_path (str): The path to the YAML configuration file.

    Returns:
        dict: The configuration as a Python dictionary.
    '''
    try:
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f'Configuration successfully loaded from \'{file_path}\'.')

        return config
    except FileNotFoundError:
        print(f'Error: The file \'{file_path}\' was not found.')
        return {}
    except yaml.YAMLError as e:
        print(f'Error parsing YAML file \'{file_path}\': {e}')
        return {}

def save_config(data, file_path):
    '''
    Loads an existing YAML config, updates it with new data, and saves it back.
    If the file doesn't exist, it creates a new one with the data.

    Args:
        data (dict): The dictionary data to add or update.
        file_path (str): The path to the YAML configuration file.
    '''
    try:
        with open(file_path, 'r') as f:
            config_data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        config_data = {}

    # The .update() method merges the dictionaries.
    # It will add new keys and overwrite any existing keys.
    config_data.update(data)

    try:
        with open(file_path, 'w') as f:
            yaml.dump(config_data, f, indent=2, sort_keys=False)
        print(f'Configuration successfully saved to \'{file_path}\'.')
    except IOError as e:
        print(f'Error writing to file \'{file_path}\': {e}')

def display_metrics(all_metrics, sort_by='da'):
    '''
    Displays MAPE and DA metrics in a neatly formatted, aligned table,
    sorted by a specific metric.

    Args:
        all_metrics (dict): A dictionary where keys are model names (str)
                            and values are metrics dictionaries (dict
                            containing 'mape' and 'da').
        sort_by (str): The metric to sort by. Must be 'mape' or 'da'.
    '''
    if not all_metrics:
        print('No metrics to display.')
        return

    sorted_items = []
    if sort_by == 'mape':
        sorted_items = sorted(all_metrics.items(), key=lambda item: item[1]['mape'])
        print('--- Model Performance (Sorted by MAPE, lower is better) ---')
    elif sort_by == 'da':
        sorted_items = sorted(all_metrics.items(), key=lambda item: item[1]['da'], reverse=True)
        print('--- Model Performance (Sorted by DA, higher is better) ---')
    else:
        print(f"Error: Invalid sort_by key. Must be 'mape' or 'da'.")
        return

    max_name_len = max(len(name) for name, _ in sorted_items)
    if max_name_len < 5:
        max_name_len = 5

    header_name = 'Model'
    print(f'\n{header_name:<{max_name_len}} | {"DA":<10} | {"MAPE":<10}')
    print(f'{"-" * max_name_len}-|------------|------------')

    for name, metrics in sorted_items:
        mape = metrics['mape']
        da = metrics['da']
        print(f'{name:<{max_name_len}} | {da:<9.4f}% | {mape:<9.4f}%')