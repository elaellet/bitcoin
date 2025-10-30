def split_btc_ds(ds, unit, train_size=0.8, valid_size=0.1):
    '''
    Splits a DataFrame into training, validation, and test sets based on specified proportions.

    Args:
        ds (pd.DataFrame): The dataset to be split.
        unit (str): A string representing the unit of the dataset (e.g., 'Day', 'Week').
        train_size (float, optional): The proportion of the dataset to allocate to the training set. Defaults to 0.8.
        valid_size (float, optional): The proportion of the dataset to allocate to the validation set. Defaults to 0.1.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing the training, validation, and test DataFrames.    
    '''
    print(f'\n--- Splitting the BTC Dataset into Training, Validation, and Test Sets ({unit}) ---')

    train_idx = int(len(ds) * train_size)
    valid_idx = int(len(ds) * valid_size)

    train_ds = ds.iloc[: train_idx]
    valid_ds = ds.iloc[train_idx : train_idx + valid_idx]
    test_ds = ds.iloc[train_idx + valid_idx: ]

    print('--- BTC Dataset Splitting Complete ---')

    return train_ds, valid_ds, test_ds