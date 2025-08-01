def split_btc_dataset(df, unit, train_size=0.8, valid_size=0.1):
    print(f'\n--- Splitting BTC Dataset to Training, Validation, and Test Set ({unit}) ---')

    train_index = int(len(df) * train_size)
    valid_index = int(len(df) * valid_size)

    df_train = df.iloc[: train_index]
    df_valid = df.iloc[train_index : train_index + valid_index]
    df_test = df.iloc[train_index + valid_index: ]

    print('--- BTC Dataset Splitting Complete ---')

    return df_train, df_valid, df_test