import json
import os
import pickle
from collections import defaultdict
from contextlib import ExitStack

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from feature_generation_utilities import (
    calculate_mean_std,
    generate_features,
    get_categories_mappings,
    grouped_mean_std_scaler,
    mean_std_scaler,
    repeat_medication,
    extract_windows,
)
from split_functions import split_across_patients


data_path = os.path.join("..", "data", "concatenated_admissions.json")
# %% split the data into training (70%), validation (10%) and testing sets (20%) with mutually exclusive patients in each set
print("Splitting data across patients into training, validation and testing sets")
train_ids, val_ids, test_ids = split_across_patients(
    train_ratio=0.7, val_ratio=0.1, random_state=42
)

# %% Mapping categories to integers and vice versa
print("Mapping categories to integers")
categories_to_int, int_to_categories = get_categories_mappings()

# %% calculating mean and std for each continuous feature
print("Calculating mean and std for each continuous feature")
scaler = calculate_mean_std(data_path, train_ids)


# %% Normalizing continuous features, windowing, and saving the data on HDF5 files
print("Normalizing continuous features, windowing, and saving the data on HDF5 files")
input_schema = {
    "static_cont": ("weight", "age", "height"),
    "static_cat": (
        "gender",
        "ethnicity",
        "hospitalid",
        "hospital_numbeds",
        "hospital_region",
        "hospitaladmitsource",
        "hospital_teachingstatus",
    ),
    "unit_info_cat": ("unittype", "unitstaytype", "unitadmitsource"),
    "diagnosis_cat": ("diagnosis", "diagnosispriority"),
    "addx_cat": ("addx",),
    "lab_cont": ("labresult",),
    "lab_cat": ("labname",),
    "IO_cont": ("cellvalue",),
    "IO_cat": ("celllabel",),
    "IO_num_reg_cont": ("num_registrations", "intake", "output", "dialysis"),
    "past_history_cat": ("pasthistoryvalue",),
    "treatment_cat": ("treatmentstring",),
    "SpO2_cont": ("value",),
    "HR": ("value",),
    "RR": ("value",),
    "nibp_mean_cont": ("value",),
    "ibp_mean_cont": ("value",),
    "nibp_systolic_cont": ("value",),
    "ibp_systolic_cont": ("value",),
    "nibp_diastolic_cont": ("value",),
    "ibp_diastolic_cont": ("value",),
    "sedation_cat": ("score",),
    "GCS_cat": ("score",),
    "Temp_cont": ("temp_value",),
    "Temp_cat": ("temp_location",),
    "med_cont": ("dosage",),
    "med_cat": ("drugname", "frequency", "routeadmin", "prn", "drugivadmixture"),
    "ad_med_cont": ("drugdosage",),
    "ad_med_cat": (
        "drugname",
        "drugadmitfrequency",
        "drugnotetype",
        "rxincluded",
        "writtenineicu",
    ),
    "infusion_cont": ("drugrate",),
    "infusion_cat": ("drugname",),
}


n_windows = {}

"""
For each input source, continuous and categorical features are extracted, normalized,
windowed according the BG offsets, and then finally written to the corresponding HDF5 file.
"""
n_examples = {"train": 0, "val": 0, "test": 0}
# we also save the patientunitstayid and window number for each class label for future sampling
example_ids = {"train": {}, "val": {}, "test": {}}
for name in example_ids:
    example_ids[name]["normal"] = []
    example_ids[name]["hypoglycemia"] = []
    example_ids[name]["hyperglycemia"] = []

with ExitStack() as stack:
    files = {
        name: stack.enter_context(
            h5py.File(os.path.join("..", "data", f"{name}_features.hdf5"), "w")
        )
        for name in ("train", "val", "test")
    }

    # create a group for the schema
    for file in files.values():
        schema_group = file.create_group("schema")
        for k, v in input_schema.items():
            schema_group.create_dataset(k, data=v)

    f = stack.enter_context(open(data_path, "r"))
    for line in tqdm(f):
        adm_info = json.loads(line)
        pu_id = adm_info["patientunitstayid"]
        if pu_id in train_ids:
            split = "train"
        elif pu_id in val_ids:
            split = "val"
        else:
            split = "test"
        h5_file = files[split]

        # static features
        static = adm_info["static"]
        discharge_offset = static["hospitaldischargeoffset"]
        static_cont = np.zeros((1, 3), dtype=np.float32)
        static_cat = np.zeros((1, 7), dtype=np.int32)
        static_cont[0, :] = [
            mean_std_scaler(static[k], scaler[k]) for k in ("weight", "age", "height")
        ]
        for i, (k1, k2) in enumerate(
            zip(
                (
                    "gender",
                    "ethnicity",
                    "hospitalid",
                    "hospital_numbeds",
                    "hospital_region",
                    "hospitaladmitsource",
                ),
                (
                    "gender",
                    "ethnicity",
                    "hospital_id",
                    "num_beds",
                    "region",
                    "admitsource",
                ),
            )
        ):
            static_cat[0, i] = categories_to_int[k2][static[k1]]
        static_cat[0, -1] = 1 if static["hospital_teachingstatus"] else 0

        # dynamic features (incluing offsets, continuous and categorical features)
        dynamic = {}

        ## unit_info features
        cat_keys = ("unittype", "unitstaytype", "unitadmitsource")
        cat_to_int_keys = ("unittype", "unitstaytype", "admitsource")
        generate_features(
            adm_info["unit_info"],
            dynamic,
            "unit_info",
            "unitadmitoffset",
            cat_keys=cat_keys,
            cat_to_int_keys=cat_to_int_keys,
        )

        ## diagnosis features
        cat_keys = ("diagnosis", "diagnosispriority")
        cat_to_int_keys = ("diagnosis", "diagnosispriority")
        generate_features(
            adm_info["diagnosis"]["diagnosis"],
            dynamic,
            "diagnosis",
            "diagnosisoffset",
            cat_keys=cat_keys,
            cat_to_int_keys=cat_to_int_keys,
        )

        ## admission diagnosis features
        cat_keys = ("addx",)
        cat_to_int_keys = ("addx",)
        generate_features(
            adm_info["diagnosis"]["addx"],
            dynamic,
            "addx",
            "admitdxenteredoffset",
            cat_keys=cat_keys,
            cat_to_int_keys=cat_to_int_keys,
        )

        ## lab features
        grouped_cont_keys = (("labname", "labresult"),)
        grouped_cont_scalers = (scaler["lab"],)
        cat_keys = ("labname",)
        cat_to_int_keys = ("labname",)
        generate_features(
            adm_info["lab"],
            dynamic,
            "lab",
            "labresultoffset",
            grouped_cont_keys=grouped_cont_keys,
            grouped_cont_scalers=grouped_cont_scalers,
            cat_keys=cat_keys,
            cat_to_int_keys=cat_to_int_keys,
        )

        ## IO features
        grouped_cont_keys = (("celllabel", "cellvalue"),)
        grouped_cont_scalers = (scaler["IO"],)
        cat_keys = ("celllabel",)
        cat_to_int_keys = ("IO_cell_label",)
        generate_features(
            adm_info["IO"],
            dynamic,
            "IO",
            "intakeoutputoffset",
            grouped_cont_keys=grouped_cont_keys,
            grouped_cont_scalers=grouped_cont_scalers,
            cat_keys=cat_keys,
            cat_to_int_keys=cat_to_int_keys,
        )

        ## IO_num_reg
        cont_keys = ("num_registrations", "intake", "output", "dialysis")
        cont_scalers = tuple(scaler[f_name] for f_name in cont_keys)
        generate_features(
            adm_info["IO_num_reg"],
            dynamic,
            "IO_num_reg",
            "intakeoutputoffset",
            cont_keys=cont_keys,
            cont_scalers=cont_scalers,
        )

        ## past history features
        cat_keys = ("pasthistoryvalue",)
        cat_to_int_keys = ("past_history",)
        generate_features(
            adm_info["past_history"],
            dynamic,
            "past_history",
            "pasthistoryoffset",
            cat_keys=cat_keys,
            cat_to_int_keys=cat_to_int_keys,
        )

        ## treatment features
        cat_keys = ("treatmentstring",)
        cat_to_int_keys = ("treatment",)
        generate_features(
            adm_info["treatment"],
            dynamic,
            "treatment",
            "treatmentoffset",
            cat_keys=cat_keys,
            cat_to_int_keys=cat_to_int_keys,
        )

        ## nurse charting features
        nc = adm_info["nurse_charting"]
        for f_name in (
            "SpO2",
            "HR",
            "RR",
            "nibp_mean",
            "ibp_mean",
            "nibp_systolic",
            "ibp_systolic",
            "nibp_diastolic",
            "ibp_diastolic",
        ):
            cont_keys = ("value",)
            cont_scalers = (scaler[f_name],)
            generate_features(
                nc[f_name],
                dynamic,
                f_name,
                "nursingchartoffset",
                cont_keys=cont_keys,
                cont_scalers=cont_scalers,
            )

        ### scores
        for f_name in ("sedation", "GCS"):
            cat_keys = ("score",)
            cat_to_int_keys = (f"{f_name}_scores",)
            generate_features(
                nc[f_name],
                dynamic,
                f_name,
                "nursingchartoffset",
                cat_keys=cat_keys,
                cat_to_int_keys=cat_to_int_keys,
            )

        ### temperature
        grouped_cont_keys = (("temp_location", "temp_value"),)
        grouped_cont_scalers = (scaler["Temp"],)
        cat_keys = ("temp_location",)
        cat_to_int_keys = ("temp_location",)
        generate_features(
            nc["Temp"],
            dynamic,
            "Temp",
            "nursingchartoffset",
            grouped_cont_keys=grouped_cont_keys,
            grouped_cont_scalers=grouped_cont_scalers,
            cat_keys=cat_keys,
            cat_to_int_keys=cat_to_int_keys,
        )

        ## medication features
        med = adm_info["med"]["med"]
        grouped_cont_keys = (("drugname", "dosage"),)
        grouped_cont_scalers = (scaler["med"],)
        cat_keys = ("drugname", "frequency", "routeadmin")
        cat_to_int_keys = ("medication", "frequency", "route_admin")
        generate_features(
            med,
            dynamic,
            "med",
            "drugstartoffset",
            grouped_cont_keys=grouped_cont_keys,
            grouped_cont_scalers=grouped_cont_scalers,
            cat_keys=cat_keys,
            cat_to_int_keys=cat_to_int_keys,
        )

        if adm_info["med"]["med"]:
            ### prn and drugivadmixture are boolean features
            bool_features = []
            bool_features.append(np.array(med["prn"], dtype=np.int32))
            bool_features.append(np.array(med["drugivadmixture"], dtype=np.int32))
            bool_features = np.stack(bool_features, axis=1)

            ### concatenate boolean and categorical features
            offsets, cat_features, cont_features = (
                dynamic["med"]["offsets"],
                dynamic["med"]["cat_features"],
                dynamic["med"]["cont_features"],
            )
            cat_features = np.concatenate([cat_features, bool_features], axis=1)

            ### repeat medications based on the stop offset and the frequency
            stop_offsets = np.array(med["drugstopoffset"], dtype=np.float32)
            offsets, cat_features, cont_features = repeat_medication(
                cat_features,
                cont_features,
                offsets,
                stop_offsets,
                discharge_offset,
                med["frequency"],
            )

            dynamic["med"] = {
                "offsets": offsets,
                "cat_features": cat_features,
                "cont_features": cont_features,
            }

        ## admission medication features
        grouped_cont_keys = (("drugname", "drugdosage"),)
        grouped_cont_scalers = (scaler["ad_med"],)
        cat_keys = ("drugname", "drugadmitfrequency", "drugnotetype")
        cat_to_int_keys = ("medication", "frequency", "drug_note_type")
        generate_features(
            adm_info["med"]["ad_med"],
            dynamic,
            "ad_med",
            "drugoffset",
            grouped_cont_keys=grouped_cont_keys,
            grouped_cont_scalers=grouped_cont_scalers,
            cat_keys=cat_keys,
            cat_to_int_keys=cat_to_int_keys,
        )

        if adm_info["med"]["ad_med"]:
            ### rxincluded and writtenineicu are boolean features
            bool_features = []
            bool_features.append(
                np.array(adm_info["med"]["ad_med"]["rxincluded"], dtype=np.int32)
            )
            bool_features.append(
                np.array(adm_info["med"]["ad_med"]["writtenineicu"], dtype=np.int32)
            )
            bool_features = np.stack(bool_features, axis=1)

            ### concatenate boolean and categorical features
            offsets, cat_features, cont_features = (
                dynamic["ad_med"]["offsets"],
                dynamic["ad_med"]["cat_features"],
                dynamic["ad_med"]["cont_features"],
            )
            cat_features = np.concatenate([cat_features, bool_features], axis=1)

            dynamic["ad_med"] = {
                "offsets": offsets,
                "cat_features": cat_features,
                "cont_features": cont_features,
            }

        ## infusion features
        grouped_cont_keys = (("drugname", "drugrate"),)
        grouped_cont_scalers = (scaler["infusion"],)
        cat_keys = ("drugname",)
        cat_to_int_keys = ("infusion_drug",)
        generate_features(
            adm_info["med"]["infusion"],
            dynamic,
            "infusion",
            "infusionoffset",
            grouped_cont_keys=grouped_cont_keys,
            grouped_cont_scalers=grouped_cont_scalers,
            cat_keys=cat_keys,
            cat_to_int_keys=cat_to_int_keys,
        )

        ## extract windows and class labels based on BG offsets and values
        stay_data, window_lengths, labels, label_offsets = extract_windows(
            adm_info, dynamic
        )
        n_windows[pu_id] = len(window_lengths)
        n_examples[split] += n_windows[pu_id]

        # Create a group for each patient stay
        pu_group = h5_file.create_group(f"patient_{pu_id}")
        pu_group.create_dataset("static_cont", data=static_cont)
        pu_group.create_dataset("static_cat", data=static_cat)
        pu_group.create_dataset("num_windows", data=n_windows[pu_id])

        # Create a group for each data source
        for k, source_data in stay_data.items():
            pu_group.create_group(k)
            for k2, v2 in source_data.items():
                pu_group[k].create_dataset(k2, data=v2)

        # Create a group and relevant data for each window
        for i, (window_length, label, label_offset) in enumerate(
            zip(window_lengths, labels, label_offsets)
        ):
            window_group = pu_group.create_group(f"window_{i}")
            window_group.create_dataset("label", data=label)
            window_group.create_dataset("label_offset", data=label_offset)
            example_id = (pu_id, i)
            if label == 0:
                example_ids[split]["normal"].append(example_id)
            elif label == 1:
                example_ids[split]["hypoglycemia"].append(example_id)
            else:
                example_ids[split]["hyperglycemia"].append(example_id)
            for k, v in window_length.items():
                window_group.create_dataset(k, data=v)


# convert the example ids to numpy arrays
for name in example_ids:
    for k, v in example_ids[name].items():
        example_ids[name][k] = np.array(v, dtype=np.int32)


# %% save the splits and the number of windows for each patient stay
with open(os.path.join("..", "data", "splits.pkl"), "wb") as f:
    pickle.dump({"train": train_ids, "val": val_ids, "test": test_ids}, f)

with open(os.path.join("..", "data", "num_windows.pkl"), "wb") as f:
    pickle.dump(n_windows, f)

# %% save the example ids for each class label
for split_name in example_ids:
    np.savez(
        os.path.join("..", "data", f"{split_name}_example_ids.npz"),
        normal=example_ids[split_name]["normal"],
        hypoglycemia=example_ids[split_name]["hypoglycemia"],
        hyperglycemia=example_ids[split_name]["hyperglycemia"],
    )
