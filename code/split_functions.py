import json
import os

import numpy as np
import pandas as pd
import sklearn

from db_conn import fetch_from_db


# %% List all relevant patientunitstayids
data_path = os.path.join("..", "data", "concatenated_admissions.json")

stay_ids = []
with open(data_path, "r") as f:
    for line in f:
        adm_info = json.loads(line)
        stay_ids.append(adm_info["patientunitstayid"])
stay_ids = set(stay_ids)


# %% split functions
def split_across_patients(train_ratio, val_ratio, random_state=42):
    patients = fetch_from_db("SELECT patientunitstayid, uniquepid FROM patient")
    patients = patients[patients["patientunitstayid"].isin(stay_ids)]
    grouped = patients.groupby("uniquepid")
    dfs = [group for _, group in grouped]
    # Shuffle and split the list of DataFrames
    dfs_shuffled = sklearn.utils.shuffle(dfs, random_state=random_state)
    # Calculate the split sizes based on the number of unique patients
    train_size = int(len(dfs_shuffled) * train_ratio)
    val_size = int(len(dfs_shuffled) * val_ratio)
    # Perform the split
    train_dfs = dfs_shuffled[:train_size]
    val_dfs = dfs_shuffled[train_size : train_size + val_size]
    test_dfs = dfs_shuffled[train_size + val_size :]
    # Concatenate the list of DataFrames back into single DataFrames for each split and get the patientunitstayids
    train_ids = set(pd.concat(train_dfs)["patientunitstayid"])
    val_ids = set(pd.concat(val_dfs)["patientunitstayid"])
    test_ids = set(pd.concat(test_dfs)["patientunitstayid"])

    return train_ids, val_ids, test_ids
