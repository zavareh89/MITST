"""
This script first concatenates all the admissions of the same patient 
to the same hospital at the same time (e.g. transfers from floor to ICU) and
updates the offsets, accordingly. Furthermore, to have a fair comparison with
[1], all blood glucose (BG) measurements that were
followed by a BG in 5 min were excluded. Also, for the same reason, only admissions
with at least 6 BG measurements were included.
Note that all offset data are updated according to hospitaladmitoffset
(i.e. the hospitaladmitoffset will be converted to 0).

structure of new data is as follows:
    {
        patientunitstayid: value,
        "static": {information related to patient demographics and hospital admission...},
        "unit_info": {information related to unit admissions...}, 
        "diagnosis": {information related to diagnosis and admission diagnosis...},
        "lab": {information related to lab tests...},
        "med": {information related to medication and admissiondrug and infusion drug...},
        "IO": {information related to input and output...},
        "IO_num_reg": {meta features for the number of IOs recorded at each timestamp (all cases not only the relavant ones) ...},
        "nurse_charting": {information related to nursecharting...},
        "past_history": {information related to past history of the patient...},
        "treatment": {information related to treatment...},
    }

    [1]: A. D. Zale, M. S. Abusamaan, J. McGready, N. Mathioudakis, Development and validation of a machine learning model 
        for classification of next glucose measurement in hospitalized patients, eClinicalMedicine 44 (2022) 

"""

import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

data_path = os.path.join("..", "data", "data.json")
concat_data_path = os.path.join("..", "data", "concatenated_admissions.json")


def update_offsets(adm_info, offset):
    for key in adm_info:
        if isinstance(adm_info[key], list):
            if "offset" in key:
                adm_info[key] = list(map(lambda x: x - offset, adm_info[key]))
        else:  # if it is a dictionary
            adm_info[key] = update_offsets(adm_info[key], offset)
    return adm_info


def sort_by_offsets(adm_info):
    keys = list(adm_info.keys())
    if len(keys) == 0:
        return adm_info
    if isinstance(adm_info[keys[0]], dict):
        for key in keys:
            adm_info[key] = sort_by_offsets(adm_info[key])
    else:
        # find the offset column
        for key in keys:
            if "offset" in key and "revised" not in key and "stop" not in key:
                offset_key = key
                break
        keys.remove(offset_key)
        # sort the values based on the offset
        list_of_columns = [adm_info[offset_key]] + [adm_info[key] for key in keys]
        zipped_lists = zip(*list_of_columns)
        sorted_lists = sorted(zipped_lists)
        list_of_sorted_columns = list(map(list, zip(*sorted_lists)))
        adm_info[offset_key] = list_of_sorted_columns[0]
        for i, key in enumerate(keys):
            adm_info[key] = list_of_sorted_columns[i + 1]
    return adm_info


def merge_dicts(d1, d2):
    """
    Merge two dictionaries, preserving keys and appending list values.

    Parameters:
    - d1: The first dictionary.
    - d2: The second dictionary, whose values will be merged into d1.

    Returns:
    A new dictionary with merged values from d1 and d2.
    """
    merged = {}
    for key in d1:
        if key in d2:
            # If both values are dictionaries, merge them recursively
            if isinstance(d1[key], dict) and isinstance(d2[key], dict):
                merged[key] = merge_dicts(d1[key], d2[key])
            # If both values are lists, append them
            elif isinstance(d1[key], list) and isinstance(d2[key], list):
                merged[key] = d1[key] + d2[key]
        else:
            # If the key is only in d1
            merged[key] = d1[key]
    # Check for any keys that are only in d2 and add them
    for key in d2:
        if key not in d1:
            merged[key] = d2[key]
    return merged


total_stays = 0
total_unique_stays = 0
total_stays_included = 0  # after removing stays with less than 5 BG measurements

with open(data_path, "r") as f:
    # write the concatenated data to a new file
    with open(concat_data_path, "w") as f2:
        for line in tqdm(f):
            # concatenate admissions
            adms = json.loads(line.strip())
            total_stays += len(adms["admissions"])
            total_unique_stays += 1
            all_admissions = {"static": {}, "unit_info": defaultdict(list)}
            all_admissions["patientunitstayid"] = adms["patientunitstayid"]
            for k, v in adms["demographics"].items():
                all_admissions["static"][k] = v
            # hospital info is same across admissions
            for k, v in adms["admissions"][0]["other"].items():
                if "hospital" in k:
                    if "offset" in k:
                        all_admissions["static"][k] = (
                            v - adms["admissions"][0]["other"]["hospitaladmitoffset"]
                        )
                    else:
                        all_admissions["static"][k] = v

            # but unit info is different for each admission (like other admission information)
            for i, adm in enumerate(adms["admissions"]):
                offset = adm["other"]["hospitaladmitoffset"]
                for k, v in adm["other"].items():
                    if "unit" in k:
                        if "offset" in k:
                            all_admissions["unit_info"][k].append(v - offset)
                        else:
                            all_admissions["unit_info"][k].append(v)
                all_admissions["unit_info"]["unitadmitoffset"].append(0 - offset)

                # add other admission info
                for k in (
                    "diagnosis",
                    "med",
                    "lab",
                    "IO",
                    "IO_num_reg",
                    "nurse_charting",
                    "past_history",
                    "treatment",
                ):
                    adm_info = adm[k]
                    adm_info = update_offsets(adm_info, offset)
                    if i == 0:
                        all_admissions[k] = adm_info
                    else:
                        all_admissions[k] = merge_dicts(all_admissions[k], adm_info)
            
            # sort the values based on the offset (if there are multiple admissions)
            if i > 0:
                for k in (
                    "diagnosis",
                    "med",
                    "lab",
                    "IO",
                    "IO_num_reg",
                    "nurse_charting",
                    "past_history",
                    "treatment",
                ):
                    all_admissions[k] = sort_by_offsets(all_admissions[k])

            # remove BG measurements that were followed by a BG in 5 min or less
            lab = all_admissions["lab"]
            if not lab:
                continue  # skip this patient stay
            df = pd.DataFrame(
                {
                    "labname": lab["labname"],
                    "labresult": lab["labresult"],
                    "labresultoffset": lab["labresultoffset"],
                    "labresultrevisedoffset": lab["labresultrevisedoffset"],
                }
            )
            df2 = df.loc[df["labname"] == "glucose"]
            if len(df2) < 6:
                continue  # skip this patient stay
            diff = np.abs(
                np.array(df2["labresultoffset"].iloc[1:])
                - np.array(df2["labresultoffset"].iloc[:-1])
            )
            idx = np.nonzero(diff <= 5)[0]
            if len(idx) > 0:
                if len(df2) - len(idx) < 6:
                    continue  # skip this patient stay
                df = df.drop(df2.index[idx])
                all_admissions["lab"] = {
                    "labname": df["labname"].tolist(),
                    "labresult": df["labresult"].tolist(),
                    "labresultoffset": df["labresultoffset"].tolist(),
                    "labresultrevisedoffset": df["labresultrevisedoffset"].tolist(),
                }
            total_stays_included += 1
            f2.write(json.dumps(all_admissions) + "\n")

print(f"Total stays: {total_stays}")
print(f"Total unique stays: {total_unique_stays}")
print(f"Total stays included in this study: {total_stays_included}")

"""output:
    Total stays: 200859
    Total unique stays: 166355
    Total stays included in this study: 114921 (before excluding horizons > 10 hours)
"""
