def split_dataset(dataset, train_size=0.8, valid_size=0.1):
    train_idx = int(len(dataset) * train_size)
    valid_idx = int(len(dataset) * valid_size)

    train_set = dataset.iloc[: train_idx]
    valid_set = dataset.iloc[train_idx : train_idx + valid_idx]
    test_set = dataset.iloc[train_idx + valid_idx: ]

    return train_set, valid_set, test_set