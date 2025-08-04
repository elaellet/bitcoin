import hashlib
import shutil
from pathlib import Path

import pandas as pd
#import pickle

# TODO: How to disable during executable file creation?
#from google.colab import files

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

def compute_sha256(path):
    '''
    Computes the SHA256 hash of a file.

    This function reads the file in binary chunks, making it memory-efficient
    for files of any size.

    Args:
        path (str or pathlib.Path): The path to the file.

    Returns:
        str: The hexadecimal representation of the computed SHA256 hash, None if the file was not found.   
    '''
    print(f'\nCalculating hash for: {path.name}...')

    path = Path(path)
    sha256_hash = hashlib.sha256()
    chunk_size = 8192
    
    try:
        with path.open('rb') as f:
            while chunk := f.read(chunk_size):
                sha256_hash.update(chunk)

        print('Hash computed successfully.')
        return sha256_hash.hexdigest()
    
    except FileNotFoundError:
        print(f'Error: The file was not found at {path}. Cannot perform computation.')
        return None
    
def validate_sha256(path):
    '''
    Validates a file's integrity by comparing its SHA256 hash against a known value.

    Args:
        path (str or pathlib.Path): The path to the file to validate.

    Returns:
        bool: True if the hashes match, False otherwise.
    '''
    path = Path(path)
    ACTUAL_HASH = '2b04e4c6df4328017d586a2972e243a83b06e23144d558c92295e905837788bb'
    computed_hash = compute_sha256(path)

    print(f'\nValidating hashes for: {path.name}...')
  
    if computed_hash == ACTUAL_HASH:
        print('Data integrity check passed.')
        return True
    else:
        print('--- !!! DATA VALIDATION FAILED !!! ---')
        print(f'Error: Hashes mismatch for {path.name}. The file may be corrupt or modified.')
        print(f'Expected: {ACTUAL_HASH}')
        print(f'Got:      {computed_hash}')
        return False

def verify_and_log_checksum(
    src_path,
    dest_path,
    checksum_log_path
):
    """
    Verifies that two files are identical by comparing their SHA-256 hashes
    and logs the checksum to a file.

    Args:
        src_path: The path to the original file.
        dest_path: The path to the copied or destination file.
        checksum_log_path: The path to the file where the checksum will be logged.
    """
    print(f'Computing hash for source file: {src_path.name}...')
    src_hash = compute_sha256(src_path)
    if not src_hash:
        print('Could not compute hash for source file. Aborting.')
        return

    print(f'Computing hash for destination file: {dest_path.name}...')
    dest_hash = compute_sha256(dest_path)
    if not dest_hash:
        print('Could not compute hash for destination file. Aborting.')
        return
        
    print('\nComparing hashes...')
    assert src_hash == dest_hash, f'Data corrupted. Hashes do not match:\nSource: {src_hash}\nDestination: {dest_hash}'
    
    print('Hashes match. Data integrity verified.')

    try:
        checksum_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(checksum_log_path, 'a') as f:
            f.write(f'{dest_path.name}: {dest_hash}\n')
        
        print(f'\nSuccessfully appended checksum to {checksum_log_path}')
    except Exception as e:
        print(f'\nFailed to write to checksum file: {e}')

def setup_dataset(path):
    '''
    Checks if a dataset file exists and prompts for upload if it doesn't.

    This function ensures the parent directory for the given file path exists.
    If the file is not found, it initiates a file upload process.

    Args:
        path (str or pathlib.Path): The path to the dataset file.
    '''
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        print(f'Dataset already exists at \'{path}\'.')
        return

    expected_filename = path.name
    print(f'Dataset not found at \'{path}\'.')
    print(f'Please upload the \'{expected_filename}\' file.')

    try:
        uploaded = files.upload()

        if not uploaded:
            print('Upload cancelled. No file was received.')
            return

        uploaded_filename = list(uploaded.keys())[0]

        shutil.move(uploaded_filename, path)
        print(f'Successfully saved dataset to \'{path}\'.')

    except Exception as e:
        print(f'An error occurred: {e}')

def create_gitignore(path='./.gitignore'):
    '''
    Creates a .gitignore file at the specified path with a default set of rules.

    Args:
        path (str or pathlib.Path): The full path, including the filename, where the
                                    .gitignore file should be created.
                                    Defaults to './.gitignore' in the current directory.
    '''
    # Content for the .gitignore file.
    # Using a raw string (r''') to avoid issues with backslashes.
    gitignore_content = r'''
        # Data and local configs.
        data/raw/
        data/cleaned/

        # Jupyter Notebook checkpoints.
        .ipynb_checkpoints/

        # Python cache.
        __pycache__/
        *.pyc

        # Other common files to ignore.
        *.pkl
        *.csv
        .DS_Store
        .env
    '''

    try:
        with open(path, 'w') as f:
            # strip() removes leading/trailing whitespace from the multiline string.
            f.write(gitignore_content.strip())
        print(f'Success: .gitignore file created at \'{path}\'')
    except IOError as e:
        # Handle potential file system errors (e.g., permission denied)
        print(f'Error: Could not create .gitignore file at \'{path}\'.')
        print(f'Reason: {e}')

def create_metadata_file(
    df,
    raw_data_path,
    dataset_path,
    metadata_output_path,
    checksum,
    timestamp_col='Timestamp',
    key_variables_notes='- Open/High/Low/Close prices\n- Volume (BTC/USD)\n- Timestamp (Unix epoch)',
    original_source_url='https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data'
):
    '''
    Creates a metadata file from a pandas DataFrame and appends it to a specified file.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        raw_data_path (str or Path): Path to the raw dataset file (e.g., 'data/raw/btc.csv').
        dataset_path (str or Path): Path to the processed dataset file.
        metadata_output_path (str or Path): Path to the file where metadata will be appended.
        checksum (str): The SHA-256 checksum of the raw data file.
        timestamp_col (str, optional): The name of the timestamp column. Defaults to 'Timestamp'.
        key_variables_notes (str, optional): A string describing the key variables.
        original_source_url (str, optional): The URL of the original data source.
    '''
    try:
        raw_data_path = Path(raw_data_path)
        
        timestamps = pd.to_datetime(df[timestamp_col], unit='s')
        frozen_date = timestamps.max()
        time_range_start = timestamps.min()
        
        num_rows, num_cols = df.shape
        
        col_details = '\n'.join([f'  - {col} ({df[col].dtype})' for col in df.columns])

        metadata = f'''
        ## Dataset: {raw_data_path.name}
        - **Frozen Date**: {frozen_date}
        - **Time Range**: {time_range_start} to {frozen_date}
        - **Dimensions**: {num_rows} rows x {num_cols} columns
        - **Columns**:
        {col_details}
        - **Key Variables**:
        {key_variables_notes}
        - **Source Path**: {dataset_path}
        - **Original Source**: Kaggle API ({original_source_url})
        - **Checksum (SHA-256)**: {checksum}
        '''

        with open(metadata_output_path, 'a') as f:
            f.write(metadata)
            
        output_dir = Path(metadata_output_path).parent
        print(f'Successfully created and appended metadata to \'{output_dir}\'')

    except FileNotFoundError:
        print(f'Error: The file or directory for output path \'{metadata_output_path}\' was not found.')
    except KeyError:
        print(f'Error: Timestamp column \'{timestamp_col}\' not found in the DataFrame.')
    except Exception as e:
        print(f'An unexpected error occurred: {e}')

def process_and_save_dataset(df, output_path, timestamp_col='timestamp', date_col='date'):
    '''
    Adds a datetime column from a Unix timestamp and saves the DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): The input DataFrame to process.
        output_path (str or Path): The full path where the output CSV file will be saved.
        timestamp_col (str, optional): The name of the source timestamp column. 
                                     Defaults to 'timestamp'.
        date_col (str, optional): The name of the new date column to be created. 
                                  Defaults to 'date'.
    '''
    try:
        df_processed = df.copy()

        df_processed[date_col] = pd.to_datetime(df_processed[timestamp_col], unit='s')
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # index=False prevents pandas from writing the DataFrame index as a column.
        df_processed.to_csv(output_path, index=False)
        
        print(f'Successfully processed data and saved to \'{output_path}\'')
        
    except KeyError:
        print(f'Error: Timestamp column \'{timestamp_col}\' not found in the DataFrame.')
    except Exception as e:
        print(f'An unexpected error occurred: {e}')


# def save_best_arima_order(path, best_arima_order):
#     with open(path, 'wb') as f:
#         pickle.dump(best_arima_order, f)

# def load_best_arima_order(path):
#     best_arima_order = None

#     with open(path, 'rb') as f:
#         best_arima_order = pickle.load(f)

#     return best_arima_order