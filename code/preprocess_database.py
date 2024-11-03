# extract the useful events from the database and convert them to a json file.
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from db_conn import fetch_from_db
from preprocessing_functions import (
    categorize_freq,
    categorize_route,
    drug_dosage_calc,
    extract_number,
)

"""Here is the final jsonl (i.e. each line one valid json file) 
structure of patient data (we also have hospital data):
    {
        patientunitstayid: value,
        "demographics": {...}, 
        "admissions": array of objects of admissions
    }

    each object of admissions is:
    {
        "other": {information related to hospital and unit admission...},
        "diagnosis": {information related to diagnosis and admission diagnosis...},
        "lab": {information related to lab tests...},
        "med": {information related to medication and admissiondrug and infusion drug...},
        "IO": {information related to input and output...},
        "IO_num_reg": {meta features for the number of IOs recorded at each timestamp (all cases not only the relavant ones) ...},
        "nurse_charting": {information related to nursecharting...},
        "past_history": {information related to past history of the patient...},
        "treatment": {information related to treatment...},
    }

    Also, categorical variables are saved in a separate dictionary called categories.
    
    """

# %% Preprocess patient + hospital table
print("Data cleaning and feature extraction for Patient table ...")
# hospital data is merged into patient admission information
hospital = fetch_from_db("SELECT * FROM hospital")
## keep categories for categorical variables
num_beds_categories = set(hospital["numbedscategory"].unique())
region_categories = set(hospital["region"].unique())
hospital_ids = set(hospital["hospitalid"].unique())

hospital.set_index("hospitalid", inplace=True)

# script to preprocess weight values
script_path = os.path.join("..", "concepts", "pivoted", "pivoted-weight.sql")


def sql_reader(script_path):
    with open(script_path, "r") as file:
        sql_content = file.read()
    return sql_content


weights = fetch_from_db(sql_reader(script_path))
weights["weight"] = weights["weight"].astype(np.float32)


# Select the most representative weight for each patient
def f(weight_series):
    # the most probable weight to be correct
    closest_value = weight_series.iloc[(weight_series - 80).abs().argsort()[:1]]
    if closest_value.values[0] > 300 or closest_value.values[0] < 5:
        return None
    else:
        return closest_value


selected_weights = weights.groupby("patientunitstayid").agg({"weight": f})

# merge the selected weights with the patient table
patients = fetch_from_db("SELECT * FROM patient")
patients.set_index("patientunitstayid", inplace=True)
patients = patients.merge(selected_weights, on="patientunitstayid", how="left")

# filling null values for weights for patients with consecutive admissions
n_null = patients["weight"].isnull().sum()
n_null_previous = -1
while n_null != n_null_previous:
    n_null_previous = n_null
    idx = patients["weight"].isnull() & patients["previous_visit"].notnull()
    patients.loc[idx, "weight"] = list(
        patients.loc[patients.loc[idx, "previous_visit"], "weight"]
    )
    idx = patients["weight"].isnull() & patients["next_visit"].notnull()
    patients.loc[idx, "weight"] = list(
        patients.loc[patients.loc[idx, "next_visit"], "weight"]
    )
    n_null = patients["weight"].isnull().sum()

# At this point, there are only 1661 null values for weight

# cleaning demographic features
patients["gender"].replace(["Unknown", "Other", ""], "Other/Unknown", inplace=True)
patients["age"].replace(["> 89", "", pd.NA], ["90", np.nan, np.nan], inplace=True)
patients["age"] = patients["age"].astype(np.float32)
patients["ethnicity"].replace("", "Other/Unknown", inplace=True)
patients["admissionheight"].replace(pd.NA, np.nan, inplace=True)
patients["admissionheight"] = patients["admissionheight"].astype(np.float32)

# cleaning other features from patient table
patients["apacheadmissiondx"].replace("", pd.NA, inplace=True)
patients["hospitaladmitsource"].replace("", "Other", inplace=True)
patients["hospitaldischargelocation"].replace("", "Other", inplace=True)
patients["hospitaldischargestatus"].replace("", pd.NA, inplace=True)
patients["unitadmitsource"].replace("", "Other", inplace=True)
patients["unitdischargelocation"].replace("", "Other", inplace=True)
patients["unitdischargestatus"].replace("", pd.NA, inplace=True)

# data type conversion
patients = patients.astype(
    {
        "hospitalid": np.int32,
        "hospitaladmitoffset": np.float32,
        "hospitaldischargeoffset": np.float32,
        "unitdischargeoffset": np.float32,
        "hospitaldischargeyear": np.int32,
    }
)

# keep categories for categorical variables
gender_categories = set(patients["gender"].unique())
ethnicity_categories = set(patients["ethnicity"].unique())
admitsource_categories = set(patients["hospitaladmitsource"].unique()) | set(
    patients["unitadmitsource"].unique()
)
dischargelocation_categories = set(
    patients["hospitaldischargelocation"].unique()
) | set(patients["unitdischargelocation"].unique())
dischargestatus_categories = set(patients["hospitaldischargestatus"].unique()) | set(
    patients["unitdischargestatus"].unique()
)
unittype_categories = set(patients["unittype"].unique())
unitstaytype_categories = set(patients["unitstaytype"].unique())

# map the first admission to list of all consecutive admissions for the same patient
admisson_list = defaultdict(list)
idx = patients.loc[patients["previous_visit"].isnull()].index
for i in idx:
    admisson_list[i].append(i)
del idx

for _id in admisson_list.keys():
    row = patients.loc[_id]
    next_visit = row["next_visit"]
    while pd.notnull(next_visit):
        admisson_list[_id].append(int(next_visit))
        next_visit = patients.loc[next_visit, "next_visit"]

# All data will be saved in a big dictionary called data
data = {}
for _id in tqdm(patients.index):
    if _id not in admisson_list:
        continue
    admission_ids = admisson_list[_id]
    first_adm = admission_ids[0]
    row = patients.loc[first_adm]
    data[first_adm] = {
        "demographics": {
            "gender": row["gender"],
            "age": row["age"],
            "ethnicity": row["ethnicity"],
            "weight": row["weight"],
            "height": row["admissionheight"],
        }
    }

    admissions = []
    for adm in admission_ids:
        row = patients.loc[adm]
        hospital_row = hospital.loc[row["hospitalid"]]
        admission = {
            "other": {
                "hospitalid": row["hospitalid"],
                "hospital_numbeds": hospital_row["numbedscategory"],
                "hospital_region": hospital_row["region"],
                "hospital_teachingstatus": hospital_row["teachingstatus"],
                "apacheadmissiondx": row["apacheadmissiondx"],
                "hospitaladmittime24": row["hospitaladmittime24"],
                "hospitaladmitoffset": row["hospitaladmitoffset"],
                "hospitaladmitsource": row["hospitaladmitsource"],
                "hospitaldischargeyear": row["hospitaldischargeyear"],
                "hospitaldischargetime24": row["hospitaldischargetime24"],
                "hospitaldischargeoffset": row["hospitaldischargeoffset"],
                "hospitaldischargelocation": row["hospitaldischargelocation"],
                "hospitaldischargestatus": row["hospitaldischargestatus"],
                "unittype": row["unittype"],
                "unitadmittime24": row["unitadmittime24"],
                "unitadmitsource": row["unitadmitsource"],
                "unitstaytype": row["unitstaytype"],
                "unitdischargeoffset": row["unitdischargeoffset"],
                "unitdischargelocation": row["unitdischargelocation"],
                "unitdischargestatus": row["unitdischargestatus"],
            }
        }
        admissions.append(admission)

    data[first_adm]["admissions"] = admissions

# %% Preprocess diagnosis table
print("Data cleaning and feature extraction for Diagnosis table ...")
diagnosis = fetch_from_db("SELECT * FROM diagnosis")

# data type conversion
diagnosis = diagnosis.astype(
    {"activeupondischarge": bool, "diagnosisoffset": np.float32}
)

# filtering out less freuqnet diagnoses
L = diagnosis["diagnosisstring"].value_counts()
L = pd.Series(L[L > 50].index)

# Filtering the diagnoses based on the terms related to blood glucose regulation directly or indirectly
related_terms = [
    "diabetes",
    "hyperglycemia",
    "hypoglycemia",
    "glucose",
    "insulin",
    "kidney",
    "pancrea",
    "sepsis",
    "liver",
    "congestive heart failure",
    "hypertension",
]
related_diagnoses = L[L.str.contains("|".join(related_terms), case=False)]
related_diagnoses = set(related_diagnoses)

# keep only relevant diagnoses
diagnosis = diagnosis[diagnosis["diagnosisstring"].isin(related_diagnoses)]

# remove duplicates
diagnosis = diagnosis.drop_duplicates(
    subset=[
        "patientunitstayid",
        "diagnosisstring",
        "diagnosisoffset",
        "diagnosispriority",
    ],
)

# keep categories for categorical variables
diagnosispriority_categories = set(diagnosis["diagnosispriority"].unique())

# indexing
diagnosis.set_index("patientunitstayid", inplace=True)
diagnosis.sort_index(inplace=True)
idx = diagnosis.index.to_numpy().astype(np.int32)  # used for efficient seaerch


# Selecting the relevant diagnoses for each admission
for _id, admission_ids in tqdm(admisson_list.items()):
    for c, adm in enumerate(admission_ids):
        if adm in idx:
            start_idx = np.searchsorted(idx, adm, side="left")
            end_idx = np.searchsorted(idx, adm, side="right")
            rows = diagnosis.iloc[start_idx:end_idx].sort_values(by="diagnosisoffset")
        else:
            data[_id]["admissions"][c]["diagnosis"] = {"diagnosis": {}}
            continue

        data[_id]["admissions"][c]["diagnosis"] = {
            "diagnosis": {
                "diagnosis": list(rows["diagnosisstring"]),
                "diagnosisoffset": list(rows["diagnosisoffset"]),
                "activeupondischarge": list(rows["activeupondischarge"]),
                "diagnosispriority": list(rows["diagnosispriority"]),
            }
        }

del diagnosis

# %% Preprocess admissiondx table
print("Data cleaning and feature extraction for admissiondx table ...")
addx = fetch_from_db("SELECT * FROM admissiondx")[
    ["patientunitstayid", "admitdxenteredoffset", "admitdxpath"]
]
addx["admitdxenteredoffset"] = addx["admitdxenteredoffset"].astype(np.float32)

# filtering out less freuqnet diagnoses
L = addx["admitdxpath"].value_counts()
L = pd.Series(L[L > 50].index)

# Filtering the diagnoses based on the terms related to blood glucose regulation directly or indirectly
related_addx = L[L.str.contains("|".join(related_terms), case=False)]
related_addx = set(related_addx)

# keep only relevant admission diagnoses
addx = addx[addx["admitdxpath"].isin(related_addx)]

# indexing
addx.set_index("patientunitstayid", inplace=True)
addx.sort_index(inplace=True)
idx = addx.index.to_numpy().astype(np.int32)  # used for efficient seaerch

# Selecting the relevant admission diagnoses for each admission
for _id, admission_ids in tqdm(admisson_list.items()):
    for c, adm in enumerate(admission_ids):
        if adm in idx:
            start_idx = np.searchsorted(idx, adm, side="left")
            end_idx = np.searchsorted(idx, adm, side="right")
            rows = addx.iloc[start_idx:end_idx].sort_values(by="admitdxenteredoffset")
        else:
            data[_id]["admissions"][c]["diagnosis"]["addx"] = {}
            continue

        data[_id]["admissions"][c]["diagnosis"]["addx"] = {
            "addx": list(rows["admitdxpath"]),
            "admitdxenteredoffset": list(rows["admitdxenteredoffset"]),
        }

del addx

# %% Preprocess lab table
print("Data cleaning and feature extraction for lab table ...")

# labs extraction
script_path = os.path.join("..", "concepts", "lab_extraction.sql")
lab_results = fetch_from_db(sql_reader(script_path))

# integrate same lab results
lab_results.loc[lab_results["labname"] == "bicarbonate", "labname"] = "HCO3"
idx = lab_results["labname"] == "Base Excess"
lab_results.loc[idx, "labname"] = "Base Deficit"
lab_results.loc[idx, "labresult"] = -lab_results.loc[idx, "labresult"]
lab_results.loc[lab_results["labname"] == "bedside glucose", "labname"] = "glucose"

# we also consider glucose measuerements from nursecharting table (around 500k new records)
bg_nurse = fetch_from_db(
    """
WITH lab_bg AS
(
SELECT DISTINCT patientunitstayid, labresultoffset
FROM lab
WHERE labname in ('bedside glucose', 'glucose')
)

SELECT nc.patientunitstayid, nc.nursingchartoffset as labresultoffset, CAST(nc.nursingchartvalue AS float4) as labresult
FROM nurseCharting nc
WHERE nc.nursingchartcelltypevalname = 'Bedside Glucose'
AND NOT EXISTS (SELECT 1 FROM lab_bg l WHERE l.patientunitstayid = nc.patientunitstayid AND l.labresultoffset = nc.nursingchartoffset);
"""
)

# concatenate lab and blood glucose from nursecharting data
bg_nurse["labresultrevisedoffset"] = bg_nurse["labresultoffset"]
bg_nurse["labname"] = "glucose"
lab_results = pd.concat([lab_results, bg_nurse])
lab_results.reset_index(drop=True, inplace=True)

related_labs = set(lab_results["labname"].unique())

# data type conversion
lab_results = lab_results.astype(
    {
        "labresult": np.float32,
        "labresultoffset": np.float32,
        "labresultrevisedoffset": np.float32,
    }
)

# removing null values and outliers
lab_results = lab_results[lab_results["labresult"].notnull()]


def remove_outliers(group):
    low = group["labresult"].quantile(0.0005)
    high = group["labresult"].quantile(0.9995)
    return group[(group["labresult"] >= low) & (group["labresult"] <= high)]


lab_results = (
    lab_results.groupby("labname").apply(remove_outliers).reset_index(drop=True)
)

# indexing
lab_results.set_index("patientunitstayid", inplace=True)
lab_results.sort_index(inplace=True)
idx = lab_results.index.to_numpy().astype(np.int32)  # for efficient seaerch


# Selecting the relevant lab tests for each admission
for _id, admission_ids in tqdm(admisson_list.items()):
    for c, adm in enumerate(admission_ids):
        if adm in idx:
            start_idx = np.searchsorted(idx, adm, side="left")
            end_idx = np.searchsorted(idx, adm, side="right")
            rows = lab_results.iloc[start_idx:end_idx].sort_values(by="labresultoffset")
        else:
            data[_id]["admissions"][c]["lab"] = {}
            continue

        data[_id]["admissions"][c]["lab"] = {
            "labresult": list(rows["labresult"]),
            "labresultoffset": list(rows["labresultoffset"]),
            "labresultrevisedoffset": list(rows["labresultrevisedoffset"]),
            "labname": list(rows["labname"]),
        }

del lab_results


# %% Preprocess medication table
print("Data cleaning and feature extraction for medication table ...")
med = fetch_from_db("SELECT * FROM medication")

# filtering out less freuqnet drugs
L = med["drugname"].value_counts()
L = pd.Series(L[L > 50].index)

# Filtering the medications based on the terms related to blood glucose regulation directly or indirectly
# and based on the hicl codes assigned to only one drug
related_med_terms = [
    "insulin",
    "regular",
    "lispro",
    "aspart",
    "glargine",
    "detemir",
    "humalog",
    "novolog",
    "lantus",
    "metformin",
    "glipizide",
    "glyburide",
    "glimepiride",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "sitagliptin",
    "saxagliptin",
    "linagliptin",
    "exenatide",
    "liraglutide",
    "levemir",
    "dapagliflozin",
    "canagliflozin",
    "empagliflozin",
    "prednisone",
    "dexamethasone",
    "Decadron",
    "hydrocortisone",
    "methylprednisolone",
    "medrol",
    "solumedrol",
    "beta-blockers",
    "thiazide diuretics",
    "niacin",
    "atypical antipsychotics",
    "statins",
    "protease inhibitors",
    "pentamidine",
    "glucagon",
    "quinolones",
    "corticosteroids",
    "dext",  # dextrose
    "D5",  # dextrose 5%
    "D10",  # dextrose 10%
    "D50",  # dextrose 50%
    "glucose",
]

related_meds = L[L.str.contains("|".join(related_med_terms), case=False)]

# hicl codes assigned to only one drug can be used to retrieve missing drug names
related_hicl = {
    36875: "HYDROCORTISONE",
    2866: "HYDROCORTISONE",
    36808: "METHYLPREDNISOLONE",
    2876: "METHYLPREDNISOLONE",
    2877: "METHYLPREDNISOLONE",
    37537: "METHYLPREDNISOLONE",
    2879: "PREDNISONE",
    2888: "DEXAMETHASONE",
    2889: "DEXAMETHASONE",
    34381: "DEXAMETHASONE",
    38176: "DEXAMETHASONE",
    38645: "DEXAMETHASONE",
    4763: "METFORMIN",
    19078: "GLUCAGON",
    11528: "INSULIN LISPRO",
    13633: "INSULIN LISPRO",
    20769: "INSULIN ASPART",
    35487: "INSULIN ASPART",
    920: "INSULIN DETEMIR",
    26407: "INSULIN DETEMIR",
    22025: "INSULIN GLARGINE",
    915: "DEXTROSE5",
    918: "DEXTROSE5",
    929: "DEXTROSE5",
    934: "DEXTROSE5",
    936: "DEXTROSE5",
    940: "DEXTROSE5",
    6071: "DEXTROSE5",
    14778: "DEXTROSE5",
    926: "DEXTROSE50",
    807: "GLUCOSE",
}

med = med[
    med["drugname"].isin(related_meds)
    | (med["drugname"].isna() & med["drughiclseqno"].isin(related_hicl))
]

for hicl, drug in related_hicl.items():
    med.loc[(med["drughiclseqno"] == hicl) & (med["drugname"].isna()), "drugname"] = (
        drug
    )

# remove medications with startoffset > stopoffset
med = med[med["drugstartoffset"] <= med["drugstopoffset"]]

# replace stopoffset 0 with null
med["drugstopoffset"].replace(0, np.nan, inplace=True)

# data type conversion
med = med.astype(
    {
        "drugstartoffset": np.float32,
        "drugstopoffset": np.float32,
        "drugorderoffset": np.float32,
    }
)

# Categorize routeadmin
med["routeadmin"] = med["routeadmin"].apply(categorize_route)

# Categorize frequency
med["frequency"] = med["frequency"].apply(categorize_freq)

# preprocess and clean the drug names and dosage values
med["drugname"] = med["drugname"].str.upper()
med["dosage2"] = med["dosage"].str.upper()
med["dosage2"] = med["dosage2"].fillna("")

## METFORMIN
drug_idx = med["drugname"].str.contains("METFORMIN")

med.loc[
    drug_idx & med["dosage2"].str.contains("EACH"),
    "dosage2",
] = ""

args = (r"\d+(?:,\d+)*", True, r"(\d+)\s*MG")

med.loc[drug_idx, "dosage2"] = med.loc[drug_idx].apply(
    drug_dosage_calc, axis=1, args=args
)

med.loc[
    drug_idx & (med["dosage2"].isin(("1", "", "PYXIS"))),
    "dosage2",
] = 500  # fix "1 Each", "PYXIS" and empty values to the common dosage

med.loc[drug_idx, "drugname"] = "METFORMIN"

## PREDNISONE
drug_idx = med["drugname"].str.contains("PREDNISONE")

med.loc[
    drug_idx
    & (
        med["dosage2"].str.contains("TAB")
        | med["dosage2"].str.contains("EA")
        | med["dosage2"].str.contains("DOSE")
        | med["dosage2"].str.contains("PYXIS")
    ),
    "dosage2",
] = ""

args = (
    r"(\d+(?:\.\d+)?)(?=\s*MG\b)?(?!.*MG/KG)",
    True,
    r"(\d+)\s*MG",
)

med.loc[drug_idx, "dosage2"] = med.loc[drug_idx].apply(
    drug_dosage_calc, axis=1, args=args
)

med.loc[
    drug_idx & med["dosage2"].str.contains("KG"),
    "dosage2",
] = ""  # remove dosage values in MG/KG

med.loc[drug_idx, "drugname"] = "PREDNISONE"

## METHYLPREDNISOLONE
drug_idx = med["drugname"].str.contains("METHYLPREDNISOLONE|MEDROL")

med.loc[
    drug_idx
    & (
        med["dosage2"].str.contains("EA")
        | med["dosage2"].str.contains("DOSE\(S\)")
        | med["dosage2"].str.contains("DOSE")
        | med["dosage2"].str.contains("MANUAL CHARGE")
        | med["dosage2"].str.contains("PYXIS")
        | med["dosage2"].str.contains("DROP")
    ),
    "dosage2",
] = ""

args = (
    r"(\d+(?:(\.|,)\d+)?)(?=\s*(MG|ML|GM)\b)?",
    True,
    r"(\d+)\s*MG",
)

med.loc[drug_idx, "dosage2"] = med.loc[drug_idx].apply(
    drug_dosage_calc, axis=1, args=args
)

### convert g to mg
GM_idx = med["dosage"].str.upper().str.endswith(("GM", "G"), na=False) & ~(
    med["dosage"].str.upper().str.endswith("MG", na=False)
)

med.loc[drug_idx & GM_idx, "dosage2"] = (med.loc[drug_idx & GM_idx, "dosage2"]).astype(
    float
) * 1000

med.loc[drug_idx, "drugname"] = "METHYLPREDNISOLONE"

## HYDROCORTISONE
drug_idx = med["drugname"].str.contains("HYDROCORTISONE")

med.loc[
    drug_idx
    & (
        med["dosage2"].str.contains("EA")
        | med["dosage2"].str.contains("MANUAL CHARGE")
        | med["dosage2"].str.contains("PYXIS")
        | med["dosage2"].str.contains("ML")
    ),
    "dosage2",
] = ""

args = (
    r"(\d+(?:(\.|,)\d+)?)(?=\s*(MG|MCG)\b)?",
    True,
    r"(\d+)\s*MG",
)

med.loc[drug_idx, "dosage2"] = med.loc[drug_idx].apply(
    drug_dosage_calc, axis=1, args=args
)

med.loc[drug_idx, "drugname"] = "HYDROCORTISONE"

## DEXAMETHASONE
drug_idx = med["drugname"].str.contains("DEXAMETHASONE|DECADRON")

med.loc[
    drug_idx
    & (
        med["dosage2"].str.contains("EA")
        | med["dosage2"].str.contains("MANUAL CHARGE")
        | med["dosage2"].str.contains("PYXIS")
        | med["dosage2"].str.contains("DROP")
        | med["dosage2"].str.contains("TAB")
        | med["dosage2"].str.contains("VL")
    ),
    "dosage2",
] = ""

args = (
    r"(\d+(?:(\.|,)\d+)?)(?=\s*(MG|ML)\b)?",
    True,
    r"(\d+)\s*MG",
)

med.loc[drug_idx, "dosage2"] = med.loc[drug_idx].apply(
    drug_dosage_calc, axis=1, args=args
)

### convert ML to mg (1 ML equivalent to 4 MG)
ML_idx = med["dosage"].str.contains("ML", case=False)

med.loc[drug_idx & ML_idx, "dosage2"] = (med.loc[drug_idx & ML_idx, "dosage2"]).astype(
    float
) * 4

med.loc[drug_idx, "drugname"] = "DEXAMETHASONE"

## GLUCAGON
drug_idx = med["drugname"].str.contains("GLUCAGON|GLUCAGEN")

med.loc[
    drug_idx
    & (
        med["dosage2"].str.contains("PYXIS")
        | med["dosage2"].str.contains("MANUAL CHARGE")
        | med["dosage2"].str.contains("100 ML")
        | med["dosage2"].str.contains("0.5-1")
    ),
    "dosage2",
] = ""

args = (r"(\d+(?:(\.|,)\d+)?)(?=\s*(MG|ML)\b)?",)

med.loc[drug_idx, "dosage2"] = med.loc[drug_idx].apply(
    drug_dosage_calc, axis=1, args=args
)

med.loc[drug_idx, "drugname"] = "GLUCAGON"

## Insulin Detemir
drug_idx = med["drugname"].str.contains("DETEMIR|LEVEMIR")

med.loc[
    drug_idx
    & (
        med["dosage2"].str.contains("PYXIS")
        | med["dosage2"].str.contains("MANUAL CHARGE")
        | med["dosage2"].str.contains("ML/HR")
        | med["dosage2"].str.contains("MCG")
        | med["dosage2"].str.contains("EA")
    ),
    "dosage2",
] = ""

args = (r"(\d+(?:(\.|,)\d+)?)(?=\s*UNIT)?",)

med.loc[drug_idx, "dosage2"] = med.loc[drug_idx].apply(
    drug_dosage_calc, axis=1, args=args
)

### convert MG to UNIT (1 UNIT equivalent to 0.142 MG)
MG_idx = med["dosage"].str.contains("MG", case=False)

med.loc[drug_idx & MG_idx, "dosage2"] = (med.loc[drug_idx & MG_idx, "dosage2"]).astype(
    float
) / 0.142

### convert ML to UNIT (1 ML equivalent to 100 UNIT)
ML_idx = med["dosage"].str.contains("ML$", case=False)

med.loc[drug_idx & ML_idx, "dosage2"] = (med.loc[drug_idx & ML_idx, "dosage2"]).astype(
    float
) * 100

### remove outliers
med.loc[
    drug_idx & (pd.to_numeric(med["dosage2"], errors="coerce") > 100), "dosage2"
] = ""

med.loc[drug_idx, "drugname"] = "INSULIN DETEMIR"

## Insulin GLARGINE
drug_idx = med["drugname"].str.contains("GLARGINE|LANTUS")

med.loc[
    drug_idx
    & (med["dosage2"].str.contains("PYXIS") | med["dosage2"].str.contains("MG")),
    "dosage2",
] = ""

args = (r"(\d+(?:(\.|,)\d+)?)(?=\s*UNIT)?",)

med.loc[drug_idx, "dosage2"] = med.loc[drug_idx].apply(
    drug_dosage_calc, axis=1, args=args
)

med.loc[drug_idx, "drugname"] = "INSULIN GLARGINE"

## Insulin REGULAR (HUMAN)
drug_idx = med["drugname"].str.contains("REG")

med.loc[
    drug_idx
    & (
        med["dosage2"].str.contains("PYXIS")
        | med["dosage2"].str.contains("MG")
        | med["dosage2"].str.contains("DOSE")
    ),
    "dosage2",
] = ""

### First, we extract all values like 0-5 or 0-5 8. Only the maximum dosage is considered.
args = (r"(\d+-\d+)(?=\s|$)",)

med.loc[drug_idx, "dosage2"] = med.loc[drug_idx].apply(
    drug_dosage_calc, axis=1, args=args
)

med.loc[drug_idx, "dosage2"] = med.loc[drug_idx, "dosage2"].apply(
    lambda x: x.split("-")[-1] if "-" in x else x
)

### Then, we extract other dosage values
args = (r"(\d+(?:(\.|,)\d+)?)(?=\s*(UNIT|ML))?",)

med.loc[drug_idx, "dosage2"] = med.loc[drug_idx].apply(
    drug_dosage_calc, axis=1, args=args
)

### convert ML to UNIT for vials
ML_idx = med["dosage"].str.contains("ML$", case=False) & med["drugname"].str.contains(
    "ML VIAL"
)

med.loc[drug_idx & ML_idx, "dosage2"] = (med.loc[drug_idx & ML_idx, "dosage2"]).astype(
    float
) * 100

med.loc[drug_idx, "drugname"] = "INSULIN REGULAR"

## INSULIN LISPRO (HumaLOG)
drug_idx = med["drugname"].str.contains("LISPRO|HUMALOG|BAG CUSTOM NDC")

med.loc[
    drug_idx
    & (
        med["dosage2"].str.contains("PYXIS")
        | med["dosage2"].str.contains("MANUAL CHARGE")
        | med["dosage2"].str.contains("EA")
        | med["dosage2"].str.contains("DOSE")
        | med["dosage2"].str.contains("DROP")
        | med["dosage2"].str.contains("APPLY")
        | med["dosage2"].str.contains("MG")
        | med["dosage2"].str.contains("ML")
        | med["dosage2"].str.contains("PACKET")
        | med["dosage2"].str.contains("PATCH")
        | med["dosage2"].str.contains("PUFF")
        | med["dosage2"].str.contains("SPRAY")
        | med["dosage2"].str.contains("TAB")
    ),
    "dosage2",
] = ""

### First, we extract all values like 0-5 or 0-5 8. Only the maximum dosage is considered.
args = (r"(\d+-\d+)(?=\s|$)",)

med.loc[drug_idx, "dosage2"] = med.loc[drug_idx].apply(
    drug_dosage_calc, axis=1, args=args
)

med.loc[drug_idx, "dosage2"] = med.loc[drug_idx, "dosage2"].apply(
    lambda x: x.split("-")[-1] if "-" in x else x
)

### Then, we extract other dosage values
args = (r"(\d+(?:(\.|,)\d+)?)(?=\s*UNIT)?",)

med.loc[drug_idx, "dosage2"] = med.loc[drug_idx].apply(
    drug_dosage_calc, axis=1, args=args
)

### set dosage for custom bags to 100
med.loc[drug_idx & med["dosage"].str.contains("BAG", case=False), "dosage2"] = "100"

med.loc[drug_idx, "drugname"] = "INSULIN LISPRO"

## INSULIN ASPART (NovoLog)
drug_idx = med["drugname"].str.contains("ASPART|NOVOLOG")

med.loc[
    drug_idx
    & (
        med["dosage2"].str.contains("PYXIS")
        | med["dosage2"].str.contains("MANUAL CHARGE")
        | med["dosage2"].str.contains("DOSE")
        | med["dosage2"].str.contains("EACH")
        | med["dosage2"].str.contains("ML")
    ),
    "dosage2",
] = ""

### First, we extract all values like 0-5 or 0-5 8. Only the maximum dosage is considered.
args = (r"(\d+-\d+)(?=\s|$)",)

med.loc[drug_idx, "dosage2"] = med.loc[drug_idx].apply(
    drug_dosage_calc, axis=1, args=args
)

med.loc[drug_idx, "dosage2"] = med.loc[drug_idx, "dosage2"].apply(
    lambda x: x.split("-")[-1] if "-" in x else x
)

### Then, we extract other dosage values
args = (r"(\d+(?:(\.|,)\d+)?)(?=\s*UNIT)?",)

med.loc[drug_idx, "dosage2"] = med.loc[drug_idx].apply(
    drug_dosage_calc, axis=1, args=args
)

med.loc[drug_idx, "drugname"] = "INSULIN ASPART"

## Glucose
med.loc[(med["drughiclseqno"] == 807) & (med["drugname"] == "DEXTROSE"), "drugname"] = (
    "GLUCOSE"  # fix drugname for ORAL DEXTROSE
)

drug_idx = med["drugname"].str.contains("GLUCOSE|GEL")

med.loc[
    drug_idx & med["dosage2"].str.contains("DOSE"),
    "dosage2",
] = ""

### First, we extract all values like 0-5 or 0-5 8. Only the maximum dosage is considered.
args = (r"(\d+-\d+)(?=\s|$)",)

med.loc[drug_idx, "dosage2"] = med.loc[drug_idx].apply(
    drug_dosage_calc, axis=1, args=args
)

med.loc[drug_idx, "dosage2"] = med.loc[drug_idx, "dosage2"].apply(
    lambda x: x.split("-")[-1] if "-" in x else x
)

### Then, we extract other dosage values
args = (r"(\d+(?:(\.|,)\d+)?)(?=\s*(GM|G|ML))?",)

med.loc[drug_idx, "dosage2"] = med.loc[drug_idx].apply(
    drug_dosage_calc, axis=1, args=args
)

med.loc[drug_idx, "drugname"] = "GLUCOSE"

## DEXTROSE 50 (D50)
drug_idx = med["drugname"].str.contains("D50|DEXTROSE50|DEXTROSE 50")

med.loc[
    drug_idx
    & (
        med["dosage2"].str.contains("MANUAL CHARGE")
        | med["dosage2"].str.contains("DOSE")
        | med["dosage2"].str.contains("PYXIS")
        | med["dosage2"].str.contains("VL")
        | med["dosage2"].str.contains("EA")
    ),
    "dosage2",
] = ""

### First, we extract all values like 0-5 or 0-5 8. Only the maximum dosage is considered.
args = (r"(\d+-\d+)(?=\s|$)",)

med.loc[drug_idx, "dosage2"] = med.loc[drug_idx].apply(
    drug_dosage_calc, axis=1, args=args
)

med.loc[drug_idx, "dosage2"] = med.loc[drug_idx, "dosage2"].apply(
    lambda x: x.split("-")[-1] if "-" in x else x
)

### Then, we extract other dosage values
args = (r"(\d+(?:(\.|,)\d+)?)(?=\s*(ML|G|GRAM|GM))?",)

med.loc[drug_idx, "dosage2"] = med.loc[drug_idx].apply(
    drug_dosage_calc, axis=1, args=args
)

### convert GM to ML based on 50% concentration
GM_idx = med["dosage"].str.upper().str.endswith(("GM", "G", "GRAM"), na=False)

med.loc[drug_idx & GM_idx, "dosage2"] = (med.loc[drug_idx & GM_idx, "dosage2"]).astype(
    float
) * 2

med.loc[drug_idx, "drugname"] = "DEXTROSE50"

## All Drugs dissolved in DEXTROSE 5 % (D5W) are mapped to Dexrose5
drug_idx = (
    (med["drugname"].str.contains("D5W") & ~med["drugname"].str.contains("DEXTROSE"))
    | med["drugname"].str.contains("IN DEXTROSE")
    | med["drugname"].isin(
        [
            "HEPARIN/DEXTROSE 5% 25000 UNITS/250 ML BAG",
            "DEXT 5%-NACL 0.45%-KCL 20MEQ LVP",
        ]
    )
    | med["drughiclseqno"].isin(
        [
            14778,
            6071,
        ]
    )
)

med.loc[
    drug_idx
    & (~med["dosage"].str.contains("ML", case=False, na=False))
    & (
        ~med["dosage"].apply(
            lambda x: " " in str(x) and str(x).replace(" ", "").isdigit()
        )  # also keep "x y" dosage format
    ),
    "dosage2",
] = ""

# The following drug dosage is in gm and is excluded separately.
med.loc[
    med["drugname"] == "100 ML  -  MAGNESIUM SULFATE IN D5W 10-5 MG/ML-% IV SOLN",
    "dosage2",
] = ""

args = (
    r"(\d+(?:(\.|,)\d+)?)(?=\s*(ML))?",
    True,
    r"(\d+)\s*ML",
)

med.loc[drug_idx, "dosage2"] = med.loc[drug_idx].apply(
    drug_dosage_calc, axis=1, args=args
)

med.loc[drug_idx, "drugname"] = "DEXTROSE5"

## All other dextrose drugs are mapped to Dexrose5. These are usually direct administration of dextrose
drug_idx = (
    med["drugname"].str.contains("DEXT") & ~med["drugname"].str.contains("DEXTROSE5")
) | med["drughiclseqno"].isin([915, 918, 929, 934, 936, 940])

med.loc[
    drug_idx
    & (~med["dosage"].str.contains("ML", case=False, na=False))
    & (
        ~med["dosage"].apply(
            lambda x: " " in str(x) and str(x).replace(" ", "").isdigit()
        )  # also keep "x y" dosage format
    ),
    "dosage2",
] = ""

args = (r"(\d+(?:(\.|,)\d+)?)(?=\s*(ML))?",)

med.loc[drug_idx, "dosage2"] = med.loc[drug_idx].apply(
    drug_dosage_calc, axis=1, args=args
)

med.loc[drug_idx, "drugname"] = "DEXTROSE5"

# make all dosage values as string for further processing
med["dosage2"] = med["dosage2"].apply(lambda x: str(x))

# replace ","  and "0" with empty value in dosage2 column to remove thousands separator and unknown values
med["dosage2"] = med["dosage2"].str.replace(",", "").replace("0", "")

# convert dosage2 column to float
med["dosage2"] = (
    pd.to_numeric(med["dosage2"], errors="coerce").fillna(np.nan).astype(np.float32)
)

# take into consideration the number of EAs and TABLETS
EA_TAB_idx = med["dosage"].str.contains("EA|TAB", case=False, na=False)
med.loc[EA_TAB_idx, "dosage2"] = med.loc[EA_TAB_idx].apply(
    lambda row: (
        (row["dosage2"] * int(row["dosage"].split()[0]))
        if row["dosage2"]
        else row["dosage2"]
    ),
    axis=1,
)

# keep max dosage for every drug and categories for other categorical features
max_dosage = med.groupby("drugname").agg({"dosage2": "max"}).reset_index()
med_categories = set(med["drugname"].unique())
freq_categories = set(med["frequency"].unique())
route_categories = set(med["routeadmin"].unique())

# map binary values to boolean
for col in ["prn", "drugivadmixture", "drugordercancelled"]:
    med[col] = med[col].apply(lambda x: True if x == "Yes" else False).astype(bool)

# remove unnecessary columns
med.drop(
    columns=["medicationid", "drughiclseqno", "gtc", "dosage", "loadingdose"],
    inplace=True,
)
med.rename(columns={"dosage2": "dosage"}, inplace=True)

# set index
med.set_index("patientunitstayid", inplace=True)
med.sort_index(inplace=True)
idx = med.index.to_numpy().astype(np.int32)  # used for efficient seaerch

# Selecting the relevant medications for each admission
for _id, admission_ids in tqdm(admisson_list.items()):
    for c, adm in enumerate(admission_ids):
        if adm in idx:
            start_idx = np.searchsorted(idx, adm, side="left")
            end_idx = np.searchsorted(idx, adm, side="right")
            rows = med.iloc[start_idx:end_idx].sort_values(by="drugstartoffset")
        else:
            data[_id]["admissions"][c]["med"] = {"med": {}}
            continue

        data[_id]["admissions"][c]["med"] = {
            "med": {
                "drugstartoffset": list(rows["drugstartoffset"]),
                "drugstopoffset": list(rows["drugstopoffset"]),
                "drugorderoffset": list(rows["drugorderoffset"]),
                "drugname": list(rows["drugname"]),
                "dosage": list(rows["dosage"]),
                "prn": list(rows["prn"]),
                "drugivadmixture": list(rows["drugivadmixture"]),
                "drugordercancelled": list(rows["drugordercancelled"]),
                "frequency": list(rows["frequency"]),
                "routeadmin": list(rows["routeadmin"]),
            }
        }

del med


# %% Preprocess admissiondrug table
print("Data cleaning and feature extraction for admissiondrug table ...")
ad_med = fetch_from_db("SELECT * FROM admissiondrug")

# filtering out less freuqnet drugs
L = ad_med["drugname"].value_counts()
L = pd.Series(L[L > 50].index)

# Filtering the medications based on the terms related to blood glucose regulation directly or indirectly
## dextrose drugs were excluded here (threre are only a few of them in the admissiondrug table)
related_meds = L[L.str.contains("|".join(related_med_terms[:-5]), case=False)]
ad_med = ad_med[ad_med["drugname"].isin(related_meds)]

# map drug names to similar names used in medication table (if possible)
# unspecified insulin are mapped to "INSULIN OTHERS"
drug_mapping = {
    "LANTUS": "INSULIN GLARGINE",
    "LANTUS SOLOSTAR": "INSULIN GLARGINE",
    "INSULIN GLARGINE,HUMAN RECOMBINANT ANALOG": "INSULIN GLARGINE",
    "METFORMIN HCL": "METFORMIN",
    "METFORMIN HCL ER": "METFORMIN",
    "NOVOLOG": "INSULIN ASPART",
    "NOVOLOG MIX 70-30": "INSULIN ASPART",
    "NOVOLOG FLEXPEN": "INSULIN ASPART",
    "HUMALOG": "INSULIN LISPRO",
    "LEVEMIR": "INSULIN DETEMIR",
    "INSULIN REGULAR, HUMAN": "INSULIN REGULAR",
    "INSULIN GLARGINE, HUMAN RECOMBINANT ANALOG": "INSULIN GLARGINE",
    "NPH, HUMAN INSULIN ISOPHANE": "NPH INSULIN",
    "INSULIN NPH HUMAN ISOPHANE": "NPH INSULIN",
    "PIOGLITAZONE HCL": "PIOGLITAZONE",
    "MEDROL": "METHYLPREDNISOLONE",
    "SOLU-MEDROL": "METHYLPREDNISOLONE",
    "GLIPIZIDE ER": "GLIPIZIDE",
    "GLUCAGON EMERGENCY KIT": "GLUCAGON",
    "SAXAGLIPTIN HCL": "SAXAGLIPTIN",
    "INSULIN SYRINGE": "INSULIN OTHERS",
    "INSULIN PUMP": "INSULIN OTHERS",
    "INSULIN ADMIN. SUPPLIES": "INSULIN OTHERS",
}

ad_med["drugname"] = ad_med["drugname"].str.strip().replace(drug_mapping)

# since all records related to "INSULIN OTHERS" and "PIOGLITAZONE" 
## have null drugrates, we exclude them from the dataset.
ad_med = ad_med[~ad_med["drugname"].isin(["INSULIN OTHERS", "PIOGLITAZONE"])]

# map drug frequencies to similar frequencies used in medication table
freq_mapping = {
    "daily": "Q24H",
    "twice a day": "Q12H",
    "at bedtime": "Once",
    "three times a day": "Q8H",
    "four times a day": "Q6H",
    "PRN": "Once",
    " ": "Misc",
    "every other day": "Misc",
}

ad_med["drugadmitfrequency"] = ad_med["drugadmitfrequency"].replace(freq_mapping)

# replace uncommon units with empty values
ad_med.loc[
    ad_med["drugunit"].isin(
        ["ml", "mmol", "tablet(s) / capsule(s)", "drops", "application"]
    ),
    "drugdosage",
] = 0  # Some values are internally stored as float here!

# convert gm to mg
gm_idx = ad_med["drugunit"] == "gm"
ad_med.loc[gm_idx, "drugdosage"] = ad_med.loc[gm_idx, "drugdosage"].astype(float) * 1000

# remove outliers from insulin dosages
insulin_idx = ad_med["drugname"].str.contains("INSULIN")
outlier_idx = pd.to_numeric(ad_med["drugdosage"], errors="coerce") > 100
ad_med.loc[insulin_idx & outlier_idx, "drugdosage"] = 0

# replace all 0 values in dosage column with NaN
ad_med["drugdosage"].replace(0, np.nan, inplace=True)

# data type conversion
ad_med = ad_med.astype(
    {
        "drugdosage": np.float32,
        "drugoffset": np.float32,
    }
)

# keep max dosage for every drug and categories for other categorical features
# Note: categories for freuqency are the same with medication table
ad_max_dosage = ad_med.groupby("drugname").agg({"drugdosage": "max"}).reset_index()
# After preprocessing, many drugs have very few records. We will only keep the drugs with more than 200 records.
group_sizes = ad_med.groupby("drugname").size()
ad_med = ad_med[ad_med["drugname"].isin(group_sizes[group_sizes >= 200].index)]

ad_med_categories = set(ad_med["drugname"].unique())
drug_note_type = set(ad_med["drugnotetype"].unique())


# map binary values to boolean
for col in ["rxincluded", "writtenineicu"]:
    ad_med[col] = (
        ad_med[col].apply(lambda x: True if x == "True" else False).astype(bool)
    )

# remove unnecessary columns
ad_med.drop(
    columns=[
        "admissiondrugid",
        "drugenteredoffset",
        "specialtytype",
        "usertype",
        "drugunit",
        "drughiclseqno",
    ],
    inplace=True,
)

# set index
ad_med.set_index("patientunitstayid", inplace=True)
ad_med.sort_index(inplace=True)
idx = ad_med.index.to_numpy().astype(np.int32)  # used for efficient seaerch

# Selecting the relevant admission medications for each admission
for _id, admission_ids in tqdm(admisson_list.items()):
    for c, adm in enumerate(admission_ids):
        if adm in idx:
            start_idx = np.searchsorted(idx, adm, side="left")
            end_idx = np.searchsorted(idx, adm, side="right")
            rows = ad_med.iloc[start_idx:end_idx].sort_values(by="drugoffset")
        else:
            data[_id]["admissions"][c]["med"]["ad_med"] = {}
            continue

        data[_id]["admissions"][c]["med"]["ad_med"] = {
            "drugoffset": list(rows["drugoffset"]),
            "drugname": list(rows["drugname"]),
            "drugnotetype": list(rows["drugnotetype"]),
            "drugdosage": list(rows["drugdosage"]),
            "drugadmitfrequency": list(rows["drugadmitfrequency"]),
            "rxincluded": list(rows["rxincluded"]),
            "writtenineicu": list(rows["writtenineicu"]),
        }


del ad_med

# %% Preprocess infusiondrug table
print("Data cleaning and feature extraction for infusiondrug table ...")
infusion = fetch_from_db("SELECT * FROM infusiondrug")

# list of infusion drugs
L = infusion["drugname"].value_counts()
L = pd.Series(L.index)

# Since more than 99% of relevant records are related to insulin and dextrose, we filter out the rest.
# To find oher relevant recrods, we could use related_med_terms from previous section.
related_inf_drugs = L[
    (
        L.str.contains("insulin", case=False)
        & ~(L.str.contains("kg/hr", case=False) | L.str.contains("mg|mcg", case=False))
    )
    | (
        L.str.contains("Dextrose|D5|D10|D50", case=False)
        & L.str.contains("ml/hr", case=False)
    )
]

infusion = infusion[infusion["drugname"].isin(related_inf_drugs)]

# extract numerical values from the text and remove null values
infusion["drugrate"] = infusion["drugrate"].apply(extract_number)
infusion = infusion[infusion["drugrate"].notnull()]

# insulin infusions
insulin_idx = infusion["drugname"].str.contains("insulin", case=False)
infusion.loc[insulin_idx, "drugname"] = "INSULIN INFUSION"
infusion = infusion[(~insulin_idx) | (insulin_idx & infusion["drugrate"] <= 100)]

# Dextrose 5 % infusions
dextrose5_idx = (
    infusion["drugname"].str.contains("D5", case=False)
    & (~infusion["drugname"].str.contains("D50", case=False))
) | infusion["drugname"].str.contains("IN DEXTROSE|5% DEXTROSE|DEXTROSE 5%", case=False)

infusion.loc[dextrose5_idx, "drugname"] = "DEXTROSE5 INFUSION"

# Dextrose 10 % infusions
dextrose10_idx = infusion["drugname"].str.contains("D10|DEXTROSE 10", case=False)
infusion.loc[dextrose10_idx, "drugname"] = "DEXTROSE10 INFUSION"

# remove outliers and remaining drugnames
infusion = infusion[infusion["drugrate"] <= 1000]
infusion = infusion[
    infusion["drugname"].isin(
        ["INSULIN INFUSION", "DEXTROSE5 INFUSION", "DEXTROSE10 INFUSION"]
    )
]

# select the relevant columns
infusion = infusion[["patientunitstayid", "infusionoffset", "drugname", "drugrate"]]
infusion["infusionoffset"] = infusion["infusionoffset"].astype(np.float32)

# set index
infusion.set_index("patientunitstayid", inplace=True)
infusion.sort_index(inplace=True)
idx = infusion.index.to_numpy().astype(np.int32)  # used for efficient seaerch

# Selecting the relevant admission medications for each admission
for _id, admission_ids in tqdm(admisson_list.items()):
    for c, adm in enumerate(admission_ids):
        if adm in idx:
            start_idx = np.searchsorted(idx, adm, side="left")
            end_idx = np.searchsorted(idx, adm, side="right")
            rows = infusion.iloc[start_idx:end_idx].sort_values(by="infusionoffset")
        else:
            data[_id]["admissions"][c]["med"]["infusion"] = {}
            continue

        data[_id]["admissions"][c]["med"]["infusion"] = {
            "infusionoffset": list(rows["infusionoffset"]),
            "drugname": list(rows["drugname"]),
            "drugrate": list(rows["drugrate"]),
        }

del infusion

# %% Preprocess intakeoutput table
# %% Preprocess lab table
print("Data cleaning and feature extraction for intakeoutput table ...")

# the number of registrations for each patient and timestamp (can be used as a meta feature)
IO_num_registration = fetch_from_db(
    """
    SELECT patientunitstayid, intakeoutputoffset, count(*) as num_registrations,
    CAST(avg(intakeTotal) AS float4) as intake, CAST(avg(outputTotal) AS float4) as output, 
    CAST(avg(dialysisTotal) AS float4) as dialysis
    FROM intakeoutput
    where celllabel not in ('Bodyweight (kg)', 'Bodyweight (lb)')
    group by patientunitstayid, intakeoutputoffset
    """
)

IO_num_registration["intakeoutputoffset"] = IO_num_registration[
    "intakeoutputoffset"
].astype(np.float32)

# intake and output extraction
script_path = os.path.join("..", "concepts", "intakeoutput_extraction.sql")
IO = fetch_from_db(sql_reader(script_path))

# aggregate similar cell labels
## insulin
IO.loc[
    IO["celllabel"].str.contains("insulin", case=False)
    & (~IO["celllabel"].str.contains("TPN", case=False)),
    "celllabel",
] = "insulin"

## TPN
IO.loc[IO["celllabel"].str.contains("TPN", case=False), "celllabel"] = "TPN"

## albumin
IO.loc[IO["celllabel"].str.contains("albumin", case=False), "celllabel"] = "albumin"

## hypertonic solution
IO.loc[IO["celllabel"].str.contains("hypertonic", case=False), "celllabel"] = (
    "hypertonic"
)

## electrolytes
IO.loc[
    IO["celllabel"].str.contains(
        "NS|saline|lactated|LR|NaCl|sodium chloride", case=False
    )
    & (~IO["celllabel"].str.contains("insulin|dextrose|D5|D10", case=False))
    & IO["cellpath"].str.contains("Crystalloids|Generic Intake", case=False),
    "celllabel",
] = "electrolytes"

## dextrose 10 % and 15 %
IO.loc[
    IO["celllabel"].str.contains("D10|dextrose 10|dextrose 15", case=False), "celllabel"
] = "D10"

## dextrose 5 %
IO.loc[IO["celllabel"].str.contains("D5|dextrose", case=False), "celllabel"] = "D5"

## CRRT (output)
IO.loc[
    IO["celllabel"].str.contains("CRRT", case=False)
    & IO["cellpath"].str.contains("output", case=False),
    "celllabel",
] = "CRRT"

## Foley and Urine
IO.loc[
    IO["celllabel"].str.contains("Urin|Urethral|Foley", case=False),
    "celllabel",
] = "urine"

## dialysis
IO.loc[
    IO["cellpath"] == "flowsheet|Flowsheet Cell Labels|I&O|Dialysis (ml)|In",
    "celllabel",
] = "dialysis_in"
IO.loc[
    IO["cellpath"] == "flowsheet|Flowsheet Cell Labels|I&O|Dialysis (ml)|Out",
    "celllabel",
] = "dialysis_out"

# remove outliers
IO = IO[~((IO["cellvalue"] <= 0) | (IO["cellvalue"] > 10000))]

related_IO = set(IO["celllabel"].unique())
IO["intakeoutputoffset"] = IO["intakeoutputoffset"].astype(np.float32)

# remove duplicates
IO = IO[
    [
        "patientunitstayid",
        "intakeoutputoffset",
        "celllabel",
        "cellvalue",
    ]
].drop_duplicates()

# indexing
IO.set_index("patientunitstayid", inplace=True)
IO.sort_index(inplace=True)
idx = IO.index.to_numpy().astype(np.int32)  # for efficient seaerch
IO_num_registration.set_index("patientunitstayid", inplace=True)
IO_num_registration.sort_index(inplace=True)
idx_n_reg = IO_num_registration.index.to_numpy().astype(
    np.int32
)  # for efficient seaerch

# Selecting the relevant intake and output values for each admission
for _id, admission_ids in tqdm(admisson_list.items()):
    for c, adm in enumerate(admission_ids):
        if adm in idx:
            start_idx = np.searchsorted(idx, adm, side="left")
            end_idx = np.searchsorted(idx, adm, side="right")
            rows = IO.iloc[start_idx:end_idx].sort_values(by="intakeoutputoffset")
        else:
            data[_id]["admissions"][c]["IO"] = {}
            continue

        data[_id]["admissions"][c]["IO"] = {
            "celllabel": list(rows["celllabel"]),
            "cellvalue": list(rows["cellvalue"]),
            "intakeoutputoffset": list(rows["intakeoutputoffset"]),
        }

# Storing the number of registration for each admission
for _id, admission_ids in tqdm(admisson_list.items()):
    for c, adm in enumerate(admission_ids):
        if adm in idx_n_reg:
            start_idx = np.searchsorted(idx_n_reg, adm, side="left")
            end_idx = np.searchsorted(idx_n_reg, adm, side="right")
            rows = IO_num_registration.iloc[start_idx:end_idx].sort_values(
                by="intakeoutputoffset"
            )
        else:
            data[_id]["admissions"][c]["IO_num_reg"] = {}
            continue

        data[_id]["admissions"][c]["IO_num_reg"] = {
            "num_registrations": list(rows["num_registrations"]),
            "intake": list(rows["intake"]),
            "output": list(rows["output"]),
            "dialysis": list(rows["dialysis"]),
            "intakeoutputoffset": list(rows["intakeoutputoffset"]),
        }

del IO, IO_num_registration


# %% Preprocess nursecharting table
print("Data cleaning and feature extraction for nursecharting table ...")
# extract all potential useful nursecharting records
## sedation score (only RASS is used here because most of the sedation value are reported in RASS scale)
sedation = fetch_from_db(
    """
WITH filtered_groups AS (
  SELECT patientunitstayid, nursingchartoffset
  FROM nurseCharting
  WHERE nursingchartcelltypevallabel = 'Sedation Scale/Score/Goal'
  GROUP BY patientunitstayid, nursingchartoffset
  HAVING NOT BOOL_OR(nursingchartcelltypevalname = 'Sedation Scale' AND nursingchartvalue != 'RASS')
)
SELECT t.patientunitstayid, t.nursingchartoffset, t.nursingchartvalue as score
FROM nurseCharting t
JOIN filtered_groups fg ON t.patientunitstayid = fg.patientunitstayid AND t.nursingchartoffset = fg.nursingchartoffset
WHERE nursingchartcelltypevalname = 'Sedation Score'

UNION ALL

SELECT t2.patientunitstayid, t2.nursingchartoffset, t2.nursingchartvalue as score
FROM nurseCharting t2
WHERE nursingchartcelltypevallabel = 'SEDATION SCORE' OR nursingchartcelltypevallabel = 'RASS'
"""
)

sedation["score"] = (
    sedation["score"].str.replace("00", "0").replace("01", "1")
)  # range is from -5 to 4

sedation["nursingchartoffset"] = sedation["nursingchartoffset"].astype(np.float32)

## GSC (Glasgow Coma Scale) score
GCS_score = fetch_from_db(
    """
SELECT patientunitstayid, nursingchartoffset, nursingchartvalue as score
FROM nurseCharting
WHERE nursingchartcelltypevallabel = 'Score (Glasgow Coma Scale)' OR nursingchartcelltypevalname = 'GCS Total'
"""
)

### remove empty values
GCS_score = GCS_score.loc[
    GCS_score["score"] != ""
]  # range is from 3 to 15 + "Unable to score due to medication"

GCS_score["nursingchartoffset"] = GCS_score["nursingchartoffset"].astype(np.float32)

## SpO2
SpO2 = fetch_from_db(
    """
    SELECT patientunitstayid, nursingchartoffset, CAST(nursingchartvalue AS float4) as value
    FROM nurseCharting
    WHERE nursingchartcelltypevallabel in ('SpO2', 'O2 Saturation') AND CAST(nursingchartvalue AS float4) BETWEEN 40 AND 100;
    """
)

SpO2["value"] = SpO2["value"].astype(np.float32)
SpO2["nursingchartoffset"] = SpO2["nursingchartoffset"].astype(np.float32)


## blood pressure
bp = fetch_from_db(
    """
    SELECT patientunitstayid, nursingchartoffset, CAST(nursingchartvalue AS float4) as value,
        case
      when nursingchartcelltypevallabel = 'MAP (mmHg)'
       or nursingchartcelltypevallabel = 'Arterial Line MAP (mmHg)'
          then 'ibp_mean'
      when nursingchartcelltypevalname = 'Invasive BP Mean'
          then 'ibp_mean'
      when nursingchartcelltypevalname = 'Non-Invasive BP Mean'
          then 'nibp_mean'
      when nursingchartcelltypevalname = 'Invasive BP Diastolic'
          then 'ibp_diastolic'
      when nursingchartcelltypevalname = 'Invasive BP Systolic'
          then 'ibp_systolic'
      when nursingchartcelltypevalname = 'Non-Invasive BP Diastolic'
          then 'nibp_diastolic'
      when nursingchartcelltypevalname = 'Non-Invasive BP Systolic'
          then 'nibp_systolic'
      else null end
    as bp
    FROM nurseCharting
    WHERE (
        (nursingchartcelltypevallabel in ('MAP (mmHg)', 'Arterial Line MAP (mmHg)') AND CAST(nursingchartvalue AS float4) BETWEEN 1 AND 250)
    OR (nursingchartcelltypevalname in ('Non-Invasive BP Mean', 'Invasive BP Mean') AND CAST(nursingchartvalue AS float4) BETWEEN 1 AND 250)
    OR (nursingchartcelltypevalname in ('Non-Invasive BP Systolic', 'Invasive BP Systolic') AND CAST(nursingchartvalue AS float4) BETWEEN 1 AND 250)
    OR (nursingchartcelltypevalname in ('Non-Invasive BP Diastolic', 'Invasive BP Diastolic') AND CAST(nursingchartvalue AS float4) BETWEEN 1 AND 300)
    ) AND nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$';
    """
)

bp["value"] = bp["value"].astype(np.float32)
bp["nursingchartoffset"] = bp["nursingchartoffset"].astype(np.float32)


### separate the blood pressure values based on the type
col = ["patientunitstayid", "nursingchartoffset", "value"]
nibp_mean = bp.loc[bp["bp"] == "nibp_mean", col]
ibp_mean = bp.loc[bp["bp"] == "ibp_mean", col]
nibp_systolic = bp.loc[bp["bp"] == "nibp_systolic", col]
nibp_diastolic = bp.loc[bp["bp"] == "nibp_diastolic", col]
ibp_systolic = bp.loc[bp["bp"] == "ibp_systolic", col]
ibp_diastolic = bp.loc[bp["bp"] == "ibp_diastolic", col]

del bp

## heart rate
HR = fetch_from_db(
    """
    SELECT patientunitstayid, nursingchartoffset, CAST(nursingchartvalue AS float4) as value
    FROM nurseCharting
    WHERE nursingchartcelltypevalname = 'Heart Rate' AND CAST(nursingchartvalue AS float4) BETWEEN 25 AND 225;
    """
)

HR["value"] = HR["value"].astype(np.float32)
HR["nursingchartoffset"] = HR["nursingchartoffset"].astype(np.float32)


## respiratory rate
RR = fetch_from_db(
    """
    SELECT patientunitstayid, nursingchartoffset, CAST(nursingchartvalue AS float4) as value
    FROM nurseCharting
    WHERE nursingchartcelltypevalname = 'Respiratory Rate' AND CAST(nursingchartvalue AS float4) BETWEEN 0 AND 60;
    """
)

RR["value"] = RR["value"].astype(np.float32)
RR["nursingchartoffset"] = RR["nursingchartoffset"].astype(np.float32)

## temperature (save Temperature and Temperature location as pairs)
Temp = fetch_from_db(
    """
with nc as
(
select
    patientunitstayid, nursingchartoffset,
  case
    when nursingchartcelltypevalname = 'Temperature (C)'
        then cast(nursingchartvalue as float4)
    else null end
    as temp_value
  , case
    when nursingchartcelltypevalname = 'Temperature Location'
        then LOWER(nursingchartvalue)
    else null end
    as temp_location
  from nursecharting
  where nursingchartcelltypevalname in ('Temperature (C)', 'Temperature Location')
)
select patientunitstayid, nursingchartoffset, 
avg(case when temp_value >= 25 and temp_value <= 46 then temp_value else null end) as temp_value, 
max(temp_location) as temp_location
from nc
GROUP BY patientunitstayid, nursingchartoffset
ORDER BY patientunitstayid, nursingchartoffset;
    """
)

Temp["temp_value"] = Temp["temp_value"].astype(np.float32)
Temp["nursingchartoffset"] = Temp["nursingchartoffset"].astype(np.float32)

### Mapping similar temperature locations (the modst frequent ones) to one name
location_mapping = {
    "oral": "Oral",
    "axillary": "Axillary",
    "axilla": "Axillary",
    "ax": "Axillary",
    "axllry": "Axillary",
    "core": "Core",
    "core urinary catheter": "Core",
    "pa catheter": "Core",
    "esophageal": "Core",
    "intravascular (swan)": "Core",
    "blood": "Core",
    "esophageal probe": "Core",
    "eso": "Core",
    "temporal": "Temporal",
    "temporal artery": "Temporal",
    "temp artery": "Temporal",
    "temporal scan": "Temporal",
    ".ta": "Temporal",
    "ta": "Temporal",
    "temporal artery scan": "Temporal",
    "temprl": "Temporal",
    "tympanic": "Tympanic",
    "rectal": "Rectal",
    "rec": "Rectal",
    "bladder": "Bladder",
    "bladr": "Bladder",
    "skin sensor": "Skin",
    "skin": "Skin",
    "forehead": "Skin",
    "f.": "Skin",
    "undocumented": "Undocumented",
}

Temp["temp_location"] = Temp["temp_location"].map(location_mapping)
### We discriminate between undocumented and NaN locations
Temp.loc[Temp["temp_location"].isnull(), "temp_location"] = "Empty"
Temp = Temp.loc[Temp["temp_value"].notnull()]

# keep categories for categorical features
sedation_scores = set(sedation["score"].unique())
GCS_scores = set(GCS_score["score"].unique())
temp_locations = set(Temp["temp_location"].unique())

# indexing
idx = {}
nurse_dfs = {
    "sedation": sedation,
    "GCS": GCS_score,
    "SpO2": SpO2,
    "nibp_mean": nibp_mean,
    "ibp_mean": ibp_mean,
    "nibp_systolic": nibp_systolic,
    "nibp_diastolic": nibp_diastolic,
    "ibp_systolic": ibp_systolic,
    "ibp_diastolic": ibp_diastolic,
    "HR": HR,
    "RR": RR,
    "Temp": Temp,
}


for name, nurse_df in nurse_dfs.items():
    nurse_df.set_index("patientunitstayid", inplace=True)
    nurse_df.sort_index(inplace=True)
    idx[name] = nurse_df.index.to_numpy().astype(np.int32)  # for efficient seaerch

# Selecting the relevant nursecharting values for each admission
for _id, admission_ids in tqdm(admisson_list.items()):
    for c, adm in enumerate(admission_ids):
        data[_id]["admissions"][c]["nurse_charting"] = {}
        for name, nurse_df in nurse_dfs.items():
            if adm in idx[name]:
                start_idx = np.searchsorted(idx[name], adm, side="left")
                end_idx = np.searchsorted(idx[name], adm, side="right")
                rows = nurse_df.iloc[start_idx:end_idx].sort_values(
                    by="nursingchartoffset"
                )
            else:
                data[_id]["admissions"][c]["nurse_charting"][name] = {}
                continue

            data[_id]["admissions"][c]["nurse_charting"][name] = {
                "nursingchartoffset": list(rows["nursingchartoffset"])
            }
            if name == "Temp":
                data[_id]["admissions"][c]["nurse_charting"][name]["temp_value"] = list(
                    rows["temp_value"]
                )
                data[_id]["admissions"][c]["nurse_charting"][name]["temp_location"] = (
                    list(rows["temp_location"])
                )
            elif name in ("sedation", "GCS"):
                data[_id]["admissions"][c]["nurse_charting"][name]["score"] = list(
                    rows["score"]
                )
            else:
                data[_id]["admissions"][c]["nurse_charting"][name]["value"] = list(
                    rows["value"]
                )

del (
    nurse_dfs,
    idx,
    sedation,
    GCS_score,
    SpO2,
    nibp_mean,
    ibp_mean,
    nibp_systolic,
    nibp_diastolic,
    ibp_systolic,
    ibp_diastolic,
    HR,
    RR,
    Temp,
)


# %% Preprocess pasthistory table
print("Data cleaning and feature extraction for pasthistory table ...")
history = fetch_from_db("SELECT * FROM pasthistory")

# finding relevant past history records to blood glucose regulation
ph_groups = fetch_from_db(
    """
    SELECT pastHistoryPath, pastHistoryValue, count(*)
    FROM pasthistory
    GROUP BY pastHistoryPath, pastHistoryValue
    """
)

keywords = [
    "diabetes",
    "insulin",
    "hypoglycemia",
    "hyperglycemia",
    "thyroid",
    "hypertension",
    "chf",
    "renal",
    "liver",
    "pancreas",
    "cirrhosis",
]


## Function to determine if any keyword is in the pasthistoryvalue
def is_relevant(value):
    return any(keyword.lower() in value.lower() for keyword in keywords)


## Filter the pasthistory based on relevance
relevant_ph_groups = ph_groups[ph_groups["pasthistorypath"].apply(is_relevant)]
history = history[
    history["pasthistorypath"].isin(relevant_ph_groups["pasthistorypath"])
]

# map similar values to the same category
history.loc[
    history["pasthistoryvalue"].str.contains("CHF", case=False), "pasthistoryvalue"
] = "CHF"

history.loc[
    history["pasthistoryvalue"].str.contains("pancreas", case=False),
    "pasthistoryvalue",
] = "pancreas"

history.loc[
    history["pasthistoryvalue"].str.contains("renal insufficiency", case=False),
    "pasthistoryvalue",
] = "renal insufficiency"

history.loc[
    history["pasthistoryvalue"].str.contains("renal failure", case=False),
    "pasthistoryvalue",
] = "renal failure"


for v in ["non-medication dependent", "medication dependent"]:
    history.loc[
        history["pasthistoryvalue"] == v,
        "pasthistoryvalue",
    ] = f"Non-Insulin Dependent Diabetes|{v}"

history.loc[
    history["pasthistoryvalue"].isin(
        [
            "clinical diagnosis",
            "ascites",
            "varices",
            "UGI bleeding",
            "encephalopathy",
            "jaundice",
            "biopsy proven",
            "coma",
        ]
    ),
    "pasthistoryvalue",
] = "Cirrhosis"

# remove the less frequent values
history = history[history["pasthistoryvalue"] != "renal tubular acidosis"]

# keep relevant columns
history = history[["patientunitstayid", "pasthistoryvalue", "pasthistoryoffset"]]

# remove duplicates and only keep the earliest past history record
history = history.sort_values(by="pasthistoryoffset").drop_duplicates(
    subset=["patientunitstayid", "pasthistoryvalue"], keep="first"
)

# keep categories for categorical features
history_categories = set(history["pasthistoryvalue"].unique())

# data type conversion and indexing
history["pasthistoryoffset"] = history["pasthistoryoffset"].astype(np.float32)
history.set_index("patientunitstayid", inplace=True)
history.sort_index(inplace=True)
idx = history.index.to_numpy().astype(np.int32)  # used for efficient seaerch

# Selecting the relevant pasthistory values for each admission
for _id, admission_ids in tqdm(admisson_list.items()):
    for c, adm in enumerate(admission_ids):
        if adm in idx:
            start_idx = np.searchsorted(idx, adm, side="left")
            end_idx = np.searchsorted(idx, adm, side="right")
            rows = history.iloc[start_idx:end_idx].sort_values(by="pasthistoryoffset")
        else:
            data[_id]["admissions"][c]["past_history"] = {}
            continue

        data[_id]["admissions"][c]["past_history"] = {
            "pasthistoryvalue": list(rows["pasthistoryvalue"]),
            "pasthistoryoffset": list(rows["pasthistoryoffset"]),
        }

del history


# %% Preprocess treatment table
print("Data cleaning and feature extraction for treatment table ...")
treatment = fetch_from_db("SELECT * FROM treatment")

treatment_mapping = {
    "subcutaneous dose of regular insulin": "subcutaneous regular insulin",
    "insulin\\|continuous infusion": "insulin continuous infusion",
    "insulin\\|sliding scale administration": "insulin sliding scale administration",
    "subcutaneous dose of longer-acting insulin": "subcutaneous longer-acting insulin",
    "insulin / glucose": "insulin_dextrose",
    "insulin/dextrose": "insulin_dextrose",
    "\\|insulin": "insulin",  # other types of insulin
    "D10W": "dextrose 10",
    "D50": "dextrose 50",
    "D5": "dextrose 5",
    "oral hypoglycemic administration": "oral hypoglycemic administration",
    "glucose": "glucose",  # other types of glucose
    "hemodialysis": "hemodialysis",
    "\\|dialysis": "dialysis",  # other types of dialysis
    "nutrition": "nutrition",
}

# keep only relevant treatments
regex_exp = "|".join(treatment_mapping.keys())
treatment = treatment.loc[treatment["treatmentstring"].str.contains(regex_exp)]

# map similar treatments to the same category
for k, v in treatment_mapping.items():
    treatment.loc[treatment["treatmentstring"].str.contains(k), "treatmentstring"] = v

# remove duplicates
treatment = treatment[
    ["patientunitstayid", "treatmentstring", "treatmentoffset"]
].drop_duplicates()

# keep categories for categorical features
treatment_categories = set(treatment["treatmentstring"].unique())

# data type conversion and indexing
treatment["treatmentoffset"] = treatment["treatmentoffset"].astype(np.float32)
treatment.set_index("patientunitstayid", inplace=True)
treatment.sort_index(inplace=True)
idx = treatment.index.to_numpy().astype(np.int32)  # used for efficient seaerch


# Selecting the relevant treatments for each admission
for _id, admission_ids in tqdm(admisson_list.items()):
    for c, adm in enumerate(admission_ids):
        if adm in idx:
            start_idx = np.searchsorted(idx, adm, side="left")
            end_idx = np.searchsorted(idx, adm, side="right")
            rows = treatment.iloc[start_idx:end_idx].sort_values(by="treatmentoffset")
        else:
            data[_id]["admissions"][c]["treatment"] = {}
            continue

        data[_id]["admissions"][c]["treatment"] = {
            "treatmentstring": list(rows["treatmentstring"]),
            "treatmentoffset": list(rows["treatmentoffset"]),
        }

del treatment


# %% Save the data and the categorical features
categories = {}
categories["hospital_id"] = list(hospital_ids)
categories["region"] = list(region_categories)
categories["num_beds"] = list(num_beds_categories)
categories["gender"] = list(gender_categories)
categories["ethnicity"] = list(ethnicity_categories)
categories["admitsource"] = list(admitsource_categories)
categories["dischargelocation"] = list(dischargelocation_categories)
categories["dischargestatus"] = list(dischargestatus_categories)
categories["unittype"] = list(unittype_categories)
categories["unitstaytype"] = list(unitstaytype_categories)
categories["diagnosis"] = list(related_diagnoses)
categories["diagnosispriority"] = list(diagnosispriority_categories)
categories["addx"] = list(related_addx)
categories["labname"] = list(related_labs)
categories["medication"] = list(med_categories)
categories["frequency"] = list(freq_categories)
categories["route_admin"] = list(route_categories)
categories["add_medication"] = list(ad_med_categories)
categories["drug_note_type"] = list(drug_note_type)
categories["infusion_drug"] = [
    "INSULIN INFUSION",
    "DEXTROSE5 INFUSION",
    "DEXTROSE10 INFUSION",
]
categories["IO_cell_label"] = list(related_IO)
categories["sedation_scores"] = list(sedation_scores)
categories["GCS_scores"] = list(GCS_scores)
categories["temp_location"] = list(temp_locations)
categories["past_history"] = list(history_categories)
categories["treatment"] = list(treatment_categories)


# Custom JSON Encoder that converts pd.NA, null values, and NumPy data types to Python-friendly types
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if pd.isna(obj):  # Handles pd.NA and np.nan
            return None
        elif isinstance(obj, np.generic):  # Checks if obj is a NumPy scalar
            return obj.item()  # Converts NumPy scalar to a Python native type
        return json.JSONEncoder.default(self, obj)


# # saving categories and saving each admission as a JSON string per line
data_path = os.path.join("..", "data", "data.json")
categories_path = os.path.join("..", "data", "categories.json")

if not os.path.exists(data_path):
    with open(data_path, "w") as f:
        for _id, adm in data.items():
            adm["patientunitstayid"] = _id
            json_string = json.dumps(adm, cls=CustomJSONEncoder)
            f.write(json_string + "\n")
else:
    print("data already exists!")

if not os.path.exists(categories_path):
    with open(categories_path, "w") as f:
        json.dump(categories, f, cls=CustomJSONEncoder)
else:
    print("categories already exists!")

print("Data and categories were saved successfully!")
