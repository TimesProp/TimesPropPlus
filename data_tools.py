import torch
import torch.nn as nn
import pandas as pd
from nom_tool import GlobalMinMaxScaler
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np


class TimeSeriesDataset(Dataset):
    def __init__(self, x, x_mark, seq_len, pred_len):
        self.x = x
        self.x_mark = x_mark
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.x) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.x[idx:idx + self.seq_len]
        y = self.x[idx + self.seq_len:idx + self.seq_len + self.pred_len]

        if self.x_mark is not None:
            x_mark = self.x_mark[idx:idx + self.seq_len]
        else:
            x_mark = None

        return x, y, x_mark

def build_levels_from_families(families, num_nodes):
    all_nodes = set(range(num_nodes))

    child_nodes = set()
    parent2children = {}
    for fam in families:
        p = fam["parent"]
        cs = fam["children"]
        parent2children[p] = cs
        child_nodes.update(cs)

    roots = list(all_nodes - child_nodes)

    if len(roots) == 0:
        raise ValueError("No root found")
    if len(roots) > 1:
        print("Warning: multiple roots:", roots)

    levels = []
    visited = set()

    current_level = roots
    while current_level:
        levels.append(current_level)
        visited.update(current_level)

        next_level = []
        for p in current_level:
            for c in parent2children.get(p, []):
                if c not in visited and c not in next_level:
                    next_level.append(c)

        current_level = next_level

    return levels

def load_tree_families(structure_csv, data_df):
    node_names = list(data_df.columns[1:])
    node2idx = {name: i for i, name in enumerate(node_names)}

    struct_df = pd.read_csv(structure_csv, index_col=0)

    struct_df = struct_df.loc[node_names, node_names]
    families = []
    for parent in node_names:
        children = struct_df.columns[struct_df.loc[parent] == 1].tolist()
        if children:
            families.append({
                "parent": node2idx[parent],
                "children": [node2idx[c] for c in children]
            })

    return families, node2idx

def load_dataset(df):

    time_col = df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col])
    if df[time_col].dt.tz is not None:
        df[time_col] = df[time_col].dt.tz_localize(None)

    time_features = pd.DataFrame({
        "month": df[time_col].dt.month,
        "day": df[time_col].dt.day,
        "weekday": df[time_col].dt.weekday,
        "hour": df[time_col].dt.hour
    })
    x_mark = torch.tensor(time_features.values, dtype=torch.float32)

    df = df.drop(columns=[time_col])

    data = df.values.astype(np.float32)
    data = torch.tensor(data, dtype=torch.float32)
    return data, x_mark
    
def load_dataset_scaler(df):

    time_col = df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col])
    if df[time_col].dt.tz is not None:
        df[time_col] = df[time_col].dt.tz_localize(None)

    time_features = pd.DataFrame({
        "month": df[time_col].dt.month,
        "day": df[time_col].dt.day,
        "weekday": df[time_col].dt.weekday,
        "hour": df[time_col].dt.hour
    })
    x_mark = torch.tensor(time_features.values, dtype=torch.float32)

    df = df.drop(columns=[time_col])

    data = df.values.astype(np.float32)
    scaler = GlobalMinMaxScaler()
    data = scaler.fit_transform(data)
    data = torch.tensor(data, dtype=torch.float32)

    return data, x_mark, scaler

def temporal_kfold_split(data, fold=4, p_train=0.7, p_val=0.1):
    assert p_train + p_val < 1.0, "p_train + p_val have to < 1"

    N = data.shape[0]
    fold_size = N // fold
    splits = []

    for i in range(fold):
        start = i * fold_size
        end = (i + 1) * fold_size if i < fold - 1 else N

        block = data[start:end]
        L = block.shape[0]

        n_train = int(L * p_train)
        n_val = int(L * p_val)

        train_set = block[:n_train]
        val_set = block[n_train:n_train + n_val]
        test_set = block[n_train + n_val:]

    return train_set, val_set, test_set

def temporal_fold_split2(x, x_mark, fold, p_train=0.7, p_val=0.1):
    assert p_train + p_val < 1.0

    N = len(x)
    fold_size = N // fold
    folds = []

    for i in range(fold):
        start = i * fold_size
        end = (i + 1) * fold_size if i < fold - 1 else N

        x_block = x[start:end]
        x_mark_block = x_mark[start:end] if x_mark is not None else None

        L = len(x_block)
        n_train = int(L * p_train)
        n_val = int(L * p_val)

        folds.append({
            "train": (x_block[:n_train],
                      x_mark_block[:n_train] if x_mark_block is not None else None),

            "val": (x_block[n_train:n_train + n_val],
                    x_mark_block[n_train:n_train + n_val] if x_mark_block is not None else None),

            "test": (x_block[n_train + n_val:],
                     x_mark_block[n_train + n_val:] if x_mark_block is not None else None),
        })

    return folds

def temporal_fold_split3(x, x_mark, fold, p_train=0.7, p_val=0.1):
    assert p_train + p_val < 1.0

    N = len(x)
    fold_size = N // fold
    folds = []

    for i in range(fold):
        start = i * fold_size
        end = (i + 1) * fold_size if i < fold - 1 else N

        x_block = x[start:end]
        x_mark_block = x_mark[start:end] if x_mark is not None else None

        L = len(x_block)
        n_train = int(L * p_train)
        n_val = int(L * p_val)

        train_x = x_block[:n_train]
        val_x = x_block[n_train:n_train + n_val]
        test_x = x_block[n_train + n_val:]

        if x_mark_block is not None:
            train_mark = x_mark_block[:n_train]
            val_mark = x_mark_block[n_train:n_train + n_val]
            test_mark = x_mark_block[n_train + n_val:]
        else:
            train_mark, val_mark, test_mark = None, None, None

        scaler = GlobalMinMaxScaler()
        
        train_x_scaled = scaler.fit_transform(train_x)
        
        val_x_scaled = scaler.transform(val_x)
        test_x_scaled = scaler.transform(test_x)

        folds.append({
            "train": (train_x_scaled, train_mark),
            "val": (val_x_scaled, val_mark),
            "test": (test_x_scaled, test_mark),
            "scaler": scaler
        })

    return folds

def build_datasets_from_folds(
    folds,
    seq_len,
    pred_len,
):
    train_sets = []
    val_sets = []
    test_sets = []

    for fold in folds:
        for split_name, collector in [
            ("train", train_sets),
            ("val", val_sets),
            ("test", test_sets),
        ]:
            x_split, x_mark_split = fold[split_name]

            
            if len(x_split) < seq_len + pred_len:
                continue

            dataset = TimeSeriesDataset(
                x=x_split,
                x_mark=x_mark_split,
                seq_len=seq_len,
                pred_len=pred_len,
            )

            collector.append(dataset)

    return train_sets, val_sets, test_sets

def build_loaders(
    train_sets,
    val_sets,
    test_sets,
    batch_size,
    num_workers=0,
    shuffle_train=True,
):
    train_loader = DataLoader(
        ConcatDataset(train_sets),
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        drop_last=True,
    )

    val_loader = DataLoader(
        ConcatDataset(val_sets),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        ConcatDataset(test_sets),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader