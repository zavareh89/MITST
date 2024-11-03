import json
import os
import pickle
from collections import defaultdict
from tqdm import tqdm

import numpy as np
import pandas as pd


# %% normalization functions
def grouped_mean_std_scaler(x, groups, scaler):
    """
    Parameters:
    - x: The input data. A list or numpy array.
    - groups: A list of keys corresponding to values in x. The length of groups should be the same as x.
    - scaler: A dictionary containing (mean, std) tuples for each possible key in groups.

    Returns:
    The normalized data.
    """
    # null vlues are replaced with mean (i.e. 0)
    output = [
        (v - scaler[k][0]) / scaler[k][1] if not pd.isna(v) else 0
        for k, v in zip(groups, x)
    ]
    return np.array(output, dtype=np.float32)


def mean_std_scaler(x, scaler):
    """
    Parameters:
    - x: The input data. numpy array or a single value.
    - scaler: (mean, std) tuple.

    Returns:
    The normalized data.
    """
    # null vlues are replaced with mean (i.e. 0)
    if isinstance(x, np.ndarray):
        output = (x - scaler[0]) / scaler[1]
        output[pd.isna(x)] = 0
        return output
    # handle single values
    if pd.isna(x):
        return 0
    return (x - scaler[0]) / scaler[1]


# %% medication repetition
freq_map = {
    "Q1H": 60,
    "Q2H": 120,
    "Q3H": 180,
    "Q4H": 240,
    "Q6H": 360,
    "Q8H": 480,
    "Q12H": 720,
    "Q24H": 1440,
}


def repeat_medication(
    cat_features, cont_features, offsets, stop_offsets, discharge_offset, frequenies
):
    """
    Parameters:
    - cat_features: Categorical features. numpy array.
    - cont_features: Continuous features. numpy array.
    - offsets: The startoffset of the medication.
    - stop_offsets: The stopoffset of the medication.
    - discharge_offset: hospital discharge offset value.
    - frequenies: The frequency of the medication. List.

    Returns:
    The repeated features and sorted by offsets.
    """
    new_cat_features = []
    new_cont_features = []
    new_offsets = []
    for i in range(len(offsets)):
        stop_offset = stop_offsets[i]
        if stop_offset is None or stop_offset > discharge_offset:
            stop_offset = discharge_offset
        freq = frequenies[i]
        offset = offsets[i]
        if freq in freq_map:
            interval = freq_map[freq]
            while offset <= stop_offset:
                new_cat_features.append(cat_features[i])
                new_cont_features.append(cont_features[i])
                new_offsets.append(offset)
                offset += interval
        else:  # if the frequency is "Once" or "Misc"
            new_cat_features.append(cat_features[i])
            new_cont_features.append(cont_features[i])
            new_offsets.append(offset)

    # sort the features by offsets and return them
    new_offsets = np.array(new_offsets, dtype=np.float32)
    new_cat_features = np.array(new_cat_features, dtype=np.int32)
    new_cont_features = np.array(new_cont_features, dtype=np.float32)
    idx = np.argsort(new_offsets)

    return new_offsets[idx], new_cat_features[idx], new_cont_features[idx]


# %% Mapping categories to integers and vice versa
categories_to_int_path = os.path.join("..", "data", "categories_to_int.pkl")
int_to_categories_path = os.path.join("..", "data", "int_to_categories.pkl")


def get_categories_mappings():
    if os.path.exists(categories_to_int_path) and os.path.exists(
        int_to_categories_path
    ):
        with open(categories_to_int_path, "rb") as f:
            categories_to_int = pickle.load(f)
        with open(int_to_categories_path, "rb") as f:
            int_to_categories = pickle.load(f)
        return categories_to_int, int_to_categories
    # if the mappings do not exist, create them
    categories_path = os.path.join("..", "data", "categories.json")
    with open(categories_path, "r") as f:
        categories = json.load(f)

    categories_to_int = {}
    for key in categories:
        if "discharge" in key or key == "add_medication":
            # exclude discharge-related categories and combine add_medication categories with medication
            continue
        if key == "medication":
            # combine add_medication categories with medication
            combined_categories = set(categories["add_medication"]) | set(
                categories["medication"]
            )
            combined_categories = list(combined_categories)
            categories_to_int[key] = {
                combined_categories[i]: i for i in range(len(combined_categories))
            }
        else:
            categories_to_int[key] = {
                categories[key][i]: i for i in range(len(categories[key]))
            }

    # map integers to categories
    int_to_categories = {}
    for key in categories_to_int:
        int_to_categories[key] = {v: k for k, v in categories_to_int[key].items()}

    # save the mappings
    with open(categories_to_int_path, "wb") as f:
        pickle.dump(categories_to_int, f)
    with open(int_to_categories_path, "wb") as f:
        pickle.dump(int_to_categories, f)

    return categories_to_int, int_to_categories


# %% calculating mean and std for each continuous feature
continuous_features = defaultdict(list)


def append_features(adm_info, feature_name, label_name, value_name):
    feature = adm_info[feature_name]
    if feature:
        df = pd.DataFrame(
            {
                "label": feature[label_name],
                "value": feature[value_name],
            }
        )
        continuous_features[feature_name].append(df)


# collect continuous features
def calculate_mean_std(data_path, train_ids):
    """
    Parameters:
    - data_path: The path to the data file.
    - train_ids: The set of training patientunitstayids.

    Returns:
    The mean and std for each (grouped) continuous feature.
    """
    scaler_path = os.path.join("..", "data", "scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            return pickle.load(f)

    # if the scaler does not exist, create it.
    scaler = {}

    with open(data_path, "r") as f:
        for line in tqdm(f):
            adm_info = json.loads(line)
            pu_id = adm_info["patientunitstayid"]
            if pu_id not in train_ids:
                continue
            # static features
            static = adm_info["static"]
            for f_name in ("weight", "age", "height"):
                continuous_features[f_name].append(static[f_name])
            # nursecharting features
            nc = adm_info["nurse_charting"]
            for f_name in (
                "SpO2",
                "nibp_mean",
                "ibp_mean",
                "nibp_systolic",
                "ibp_systolic",
                "nibp_diastolic",
                "ibp_diastolic",
                "HR",
                "RR",
            ):
                if nc[f_name]:
                    continuous_features[f_name].append(
                        np.array(nc[f_name]["value"], dtype=np.float32)
                    )
            # IO_num_reg features
            IO_num_reg = adm_info["IO_num_reg"]
            if IO_num_reg:
                for f_name in ("num_registrations", "intake", "output", "dialysis"):
                    continuous_features[f_name].append(
                        np.array(IO_num_reg[f_name], dtype=np.float32)
                    )

            # lab features
            append_features(adm_info, "lab", "labname", "labresult")

            # IO features
            append_features(adm_info, "IO", "celllabel", "cellvalue")

            # medication features
            append_features(adm_info["med"], "med", "drugname", "dosage")

            # addmission medication features
            append_features(adm_info["med"], "ad_med", "drugname", "drugdosage")

            # infusion features
            append_features(adm_info["med"], "infusion", "drugname", "drugrate")

            # temperature features
            append_features(
                adm_info["nurse_charting"], "Temp", "temp_location", "temp_value"
            )

    # concatenate the features
    for f_name in ("weight", "age", "height"):
        continuous_features[f_name] = np.array(
            continuous_features[f_name], dtype=np.float32
        )
    for f_name in (
        "SpO2",
        "nibp_mean",
        "ibp_mean",
        "nibp_systolic",
        "ibp_systolic",
        "nibp_diastolic",
        "ibp_diastolic",
        "HR",
        "RR",
        "num_registrations",
        "intake",
        "output",
        "dialysis",
    ):
        continuous_features[f_name] = np.concatenate(
            continuous_features[f_name], dtype=np.float32
        )
    for f_name in ("lab", "IO", "med", "ad_med", "infusion", "Temp"):
        continuous_features[f_name] = pd.concat(continuous_features[f_name])

    # calculate mean and std for each continuous feature
    for f_name, features in continuous_features.items():
        if isinstance(features, pd.DataFrame):
            mean_ = features.groupby("label").mean()
            std_ = features.groupby("label").std()
            # scaler is a dictionary with label as feature name and (mean, std) tuple as value
            scaler[f_name] = {}
            for label in mean_.index:
                mean_val = mean_.loc[label, "value"]
                std_val = std_.loc[label, "value"]
                scaler[f_name][label] = (mean_val, std_val)
            features = features["value"]
        else:
            scaler[f_name] = (np.nanmean(features), np.nanstd(features))

    # save the scaler
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    return scaler


# %% functions related to feature generation
categories_to_int, int_to_categories = get_categories_mappings()


def generate_categorical_features(source_info, cat_keys, cat_to_int_keys):

    n_cat_features = len(cat_keys)  # number of categorical features
    L = len(source_info[cat_keys[0]])  # number of observations
    cat_features = np.zeros((L, n_cat_features), dtype=np.int32)
    for i, (k1, k2) in enumerate(zip(cat_keys, cat_to_int_keys)):
        cat_features[:, i] = np.array(
            [categories_to_int[k2][source_info[k1][j]] for j in range(L)],
            dtype=np.int32,
        )
    return cat_features


def generate_continuous_features(
    source_info,
    cont_keys,
    cont_scalers,
    grouped_cont_keys,
    grouped_cont_scalers,
):

    # number of continuous features
    n_cont_keys = len(cont_keys) if cont_keys else 0
    # number of grouped continuous features
    n_grouped_cont_keys = len(grouped_cont_keys) if grouped_cont_keys else 0

    # continuous features
    if n_cont_keys > 0:
        L = len(source_info[cont_keys[0]])  # number of observations
        cont_features = np.zeros((L, n_cont_keys), dtype=np.float32)
        for i, k in enumerate(cont_keys):
            x = np.array(source_info[k], dtype=np.float32)
            cont_features[:, i] = mean_std_scaler(x, cont_scalers[i])

    # grouped continuous features
    if n_grouped_cont_keys > 0:
        L = len(source_info[grouped_cont_keys[0][0]])  # number of observations
        g_cont_features = np.zeros((L, n_grouped_cont_keys), dtype=np.float32)
        for i, (label_col, val_col) in enumerate(grouped_cont_keys):
            g_cont_features[:, i] = grouped_mean_std_scaler(
                source_info[val_col], source_info[label_col], grouped_cont_scalers[i]
            )

    if n_cont_keys > 0 and n_grouped_cont_keys > 0:
        return np.concatenate([cont_features, g_cont_features], axis=1)
    elif n_cont_keys > 0:
        return cont_features
    else:
        return g_cont_features


def generate_features(
    source_info,
    dynamic_features,
    dynamic_key,
    offset_col,
    cont_keys=None,
    cont_scalers=None,
    grouped_cont_keys=None,
    grouped_cont_scalers=None,
    cat_keys=None,
    cat_to_int_keys=None,
):
    """This function generates numpy features for a single patient stay
    and updates the dynamic_features dictionary, corresponding to the dynamic_key.

    Args:
        source_info (dict): dictionary containing the source information.
        dynamic_features (dict): all features for a single patient stay.
        dynamic_key (str): the key used to save the source info in dynamic_features.
        offset_col (str): column name for the offset in the source_info.
        cont_keys (list of str, optional): list of continuous feature names in the source_info.
        cont_scalers (list of 2-tuples, optional): mean and std for each continuous feature.
        grouped_cont_keys (list of 2-tuples, optional): label_name and value_name for each grouped continuous feature.
        grouped_cont_scalers (list of dict, optional): mean and std for each grouped continuous features.
            Each dict contains (mean, std) tuples for each possible key in the grouped continuous feature.
        cat_keys (list of str, optional): list of categorical feature names in the source_info.
        cat_to_int_keys (list of str, optional): list of keys to map the categorical features to integers (keys should be available in "categories_to_int" variable).
    """
    # if source_info is empty, do not process it
    if not source_info:
        return
    offsets = np.array(source_info[offset_col], dtype=np.float32)
    L = len(offsets)
    d_features = {"offsets": offsets}

    # continuous features
    if cont_keys or grouped_cont_keys:
        cont_features = generate_continuous_features(
            source_info,
            cont_keys,
            cont_scalers,
            grouped_cont_keys,
            grouped_cont_scalers,
        )
        d_features["cont_features"] = cont_features

    # categorical features
    if cat_keys:
        cat_features = generate_categorical_features(
            source_info, cat_keys, cat_to_int_keys
        )
        d_features["cat_features"] = cat_features

    dynamic_features.update({dynamic_key: d_features})


# %% extract windows for a single patient stay (exclude next BG measurements larger than 10 hours)
def extract_windows(adm_info, dynamic):
    # We only keep window lengths and all data (i.e. dynamic) to eliminate the need for storing each window
    stay_data = {}
    labels, label_offsets, window_lengths = [], [], []
    lab = adm_info["lab"]
    lab_offsets = np.array(lab["labresultoffset"], dtype=np.float32)
    labnames = pd.Series(lab["labname"])
    labresults = np.array(lab["labresult"], dtype=np.float32)
    bg_idx = np.where(labnames == "glucose")[0][4:]
    for i in range(len(bg_idx) - 1):
        bg_index = bg_idx[i]
        bg_offset_index = lab_offsets[bg_index]
        label_offset = lab_offsets[bg_idx[i + 1]]
        if label_offset - bg_offset_index > 10 * 60:
            continue
        # add labels
        label_offsets.append(label_offset)
        bg_value = labresults[bg_idx[i + 1]]
        if bg_value <= 70:
            labels.append(1)  # hypoglycemia
        elif bg_value > 180:
            labels.append(2)  # hyperglycemia
        else:
            labels.append(0)  # normal BG
        # add window
        window_length = {}
        for k, v in dynamic.items():
            offsets = v["offsets"]
            incl_window = offsets <= bg_offset_index
            win_len = np.sum(incl_window)
            if np.sum(win_len) == 0:
                continue
            window_length[k] = win_len
        window_lengths.append(window_length)

        if len(window_lengths) > 0:
            stay_data = dynamic

    return stay_data, window_lengths, labels, label_offsets
