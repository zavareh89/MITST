import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassConfusionMatrix,
    MulticlassPrecision,
    MulticlassSpecificity,
    MulticlassRecall,
    MulticlassROC,
    MulticlassPrecisionRecallCurve,
    MulticlassAUROC,
    MulticlassAveragePrecision,
)
from tqdm import tqdm

from dataset import PatientDataset, collate_fn, move_to_device
from feature_generation_utilities import get_categories_mappings
from models import EHRTransformer


# set parameters
batch_size = 16
n_workers = 20  # The main overhead is for GPU computations not data loading
shared_dim = 32
n_classes = 3
tf_n_heads = (8, 8, 8)
tf_dim_head = (8, 8, 8)
tf_depths = (4, 4, 4)
n_sources = 23

# prefix format: batch_size_shared_dim_tf_depths_tf_dim_head_tf_n_heads
save_prefix = f"{batch_size}_{shared_dim}_444_888_888_seq"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# best model on validation is used for final evaluation
model_path = os.path.join("..", "saved_models", f"weights_epoch31_{save_prefix}.pth")
test_file = os.path.join("..", "data", "test_features.hdf5")
test_example_id_path = os.path.join("..", "data", "test_example_ids.npz")

# find the categorical index for BG measurments
scaler_path = os.path.join("..", "data", "scaler.pkl")
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)
(bg_mean, bg_std) = scaler["lab"]["glucose"]
categories_to_int, int_to_categories = get_categories_mappings()
bg_cat_idx = categories_to_int["labname"]["glucose"]

# define the model and load the weights
model = EHRTransformer(
    shared_dim=shared_dim,
    tf_n_heads=tf_n_heads,
    tf_depths=tf_depths,
    tf_dim_head=tf_dim_head,
    n_classes=n_classes,
)
model.load_state_dict(torch.load(model_path))
model.to(device)

# build the test dataset
test_dataset = PatientDataset(
    test_file,
    test_example_id_path,
    do_subsampling=False,
    shuffle=False,
    load_to_memory=True,
    num_workers=n_workers,
)
test_dataloader = DataLoader(
    test_dataset, batch_size=batch_size * 8, collate_fn=collate_fn
)

# we also compute the horizon for prediction (i.e. label offset - the last BG measurement offset)
print("Computing the prediction horizon for every window...")
t_to_next_bg = np.zeros((len(test_dataset),))
for i in tqdm(range(len(test_dataset))):
    (sources, offsets), (label, label_offset) = test_dataset[i]
    last = offsets["lab"][-1]
    t_to_next_bg[i] = label_offset - last


# %% evaluate the model on the test set
test_logits = []
test_labels = []
with torch.no_grad():
    for i, model_input in enumerate(tqdm(test_dataloader)):
        # Move tensors to the appropriate device
        source_data, label_data, seq_len_per_source, mask = move_to_device(
            model_input, device
        )

        # Forward pass
        logits = model(source_data, mask, seq_len_per_source)
        test_logits.append(logits)
        labels, label_offsets = label_data
        test_labels.append(labels)


test_logits = torch.cat(test_logits, dim=0).cpu().numpy()
test_labels = torch.cat(test_labels, dim=0).cpu().numpy()

test_results_path = os.path.join(
    "..", "test_results", f"test_results_{save_prefix}.npz"
)
np.savez(
    test_results_path,
    test_logits=test_logits,
    test_labels=test_labels,
    t_to_next_bg=t_to_next_bg,
)

# %% showcase the results
# Define metrics for each class and collect them
metrics = MetricCollection(
    {
        "confusion_mat": MulticlassConfusionMatrix(num_classes=n_classes),
        "precision": MulticlassPrecision(num_classes=n_classes, average=None),
        "specificity": MulticlassSpecificity(num_classes=n_classes, average=None),
        "recall": MulticlassRecall(num_classes=n_classes, average=None),
        "auroc": MulticlassAUROC(num_classes=n_classes, average=None),
        "auprc": MulticlassAveragePrecision(num_classes=n_classes, average=None),
        "roc": MulticlassROC(num_classes=n_classes, average=None),
        "prc": MulticlassPrecisionRecallCurve(num_classes=n_classes, average=None),
    }
)

# test_results = np.load(test_results_path)
# logits = test_results["test_logits"]
# labels = test_results["test_labels"]
# t_to_next_bg = np.load(t_to_next_bg_path)
logits, labels, t_to_next_bg = [
    torch.from_numpy(x) for x in (test_logits, test_labels, t_to_next_bg)
]

print("results for test features")
r3 = metrics(logits.softmax(axis=-1), labels)
print(
    f"test_AUROC: controlled:{r3['auroc'][0]:.4f}, hypo:{r3['auroc'][1]:.4f}, hyper:{r3['auroc'][2]:.4f}"
)
print(
    f"test_AUPRC: controlled:{r3['auprc'][0]:.4f}, hypo:{r3['auprc'][1]:.4f}, hyper:{r3['auprc'][2]:.4f}"
)
