import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassROC,
    MulticlassPrecisionRecallCurve,
    MulticlassAUROC,
    MulticlassAveragePrecision,
)
from tqdm import tqdm

from dataset import PatientDataset, collate_fn, move_to_device
from models import EHRTransformer

batch_size = 16  # the actual batch size
inner_batch_size = 16  # used along with gradient accumulation
accumulation_steps = batch_size // inner_batch_size
n_epochs = 50
# number of workers for data loading into memory (set this to None if you don't want to use it)
# This is used for loading data into memory (load_to_memory=True) to speed up training
n_workers = 20
shared_dim = 32
n_classes = 3
tf_n_heads = (8, 8, 8)
tf_dim_head = (8, 8, 8)
tf_depths = (4, 4, 4)
n_sources = 23

# prefix format: batch_size_shared_dim_tf_depths_tf_dim_head_tf_n_heads_seq
save_prefix = f"{batch_size}_{shared_dim}_444_888_888_seq"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define the model
model = EHRTransformer(
    shared_dim=shared_dim,
    tf_n_heads=tf_n_heads,
    tf_depths=tf_depths,
    tf_dim_head=tf_dim_head,
    n_classes=n_classes
)
model.to(device)

# Define metrics for each class and collect them
metrics = MetricCollection(
    {
        "auroc": MulticlassAUROC(num_classes=n_classes, average=None),
        "auprc": MulticlassAveragePrecision(num_classes=n_classes, average=None),
    }
)
metrics.to(device)

# train and validate the model
train_file = os.path.join("..", "data", "train_features.hdf5")
train_example_id_path = os.path.join("..", "data", "train_example_ids.npz")
val_file = os.path.join("..", "data", "val_features.hdf5")
val_example_id_path = os.path.join("..", "data", "val_example_ids.npz")

dataset = PatientDataset(
    train_file,
    train_example_id_path,
    do_subsampling=True,
    load_to_memory=True,
    num_workers=n_workers,
)
dataloader = DataLoader(dataset, batch_size=inner_batch_size, collate_fn=collate_fn)
val_dataset = PatientDataset(
    val_file,
    val_example_id_path,
    do_subsampling=False,
    load_to_memory=True,
    num_workers=n_workers,
)
val_dataloader = DataLoader(
    val_dataset, batch_size=batch_size * 2, collate_fn=collate_fn
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
train_time_per_epoch = []
val_time_per_epoch = []
train_auroc, val_auroc, train_auprc, val_auprc = (
    np.zeros((n_epochs, n_classes), dtype=np.float32) for _ in range(4)
)

# variables for early stopping
best_metric = 0
patience = 5
trigger_times = 0

for epoch in range(n_epochs):
    print(f"Epoch {epoch+1}")
    start_time = time.time()
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()

    for i, model_input in enumerate(tqdm(dataloader)):
        # Move tensors to the appropriate device
        source_data, label_data, seq_len_per_source, mask = move_to_device(
            model_input, device
        )

        # Forward pass
        logits = model(source_data, mask, seq_len_per_source)

        # Compute loss
        labels, label_offsets = label_data
        loss = criterion(logits, labels)

        # Backward and optimize
        (loss / accumulation_steps).backward()  # Accumulate gradients
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            loss_value = loss.detach().item()
            total_loss += loss_value
            num_batches += 1
            optimizer.zero_grad()

        # update metrics
        metrics.update(logits.softmax(dim=-1).detach(), labels.detach())

        # clear cache
        torch.cuda.empty_cache()

        if (i + 1) % (500 * accumulation_steps) == 0:

            print(
                f"Epoch [{epoch+1}/{n_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss_value:.4f}"
            )
            results = metrics.compute()
            auroc = results["auroc"].cpu().numpy()
            auprc = results["auprc"].cpu().numpy()
            print(
                f"AUROC: normal:{auroc[0]:.4f} hypo:{auroc[1]:.4f} hyper:{auroc[2]:.4f}"
            )
            print(
                f"AUPRC: normal:{auprc[0]:.4f} hypo:{auprc[1]:.4f} hyper:{auprc[2]:.4f}"
            )

    # compute average loss and metrics for the epoch
    average_loss = total_loss / num_batches
    print(f"Epoch [{epoch+1}/{n_epochs}], Average Loss: {average_loss:.4f}")
    results = metrics.compute()
    metrics.reset()
    train_auroc[epoch] = results["auroc"].cpu().numpy()
    train_auprc[epoch] = results["auprc"].cpu().numpy()
    train_time_per_epoch.append(time.time() - start_time)

    # Validation
    with torch.no_grad():
        start_time = time.time()
        for i, model_input in enumerate(tqdm(val_dataloader)):
            # Move tensors to the appropriate device
            source_data, label_data, seq_len_per_source, mask = move_to_device(
                model_input, device
            )

            # Forward pass
            logits = model(source_data, mask, seq_len_per_source)

            # update metrics
            labels, label_offsets = label_data
            metrics.update(logits.softmax(dim=-1), labels)

    results = metrics.compute()
    metrics.reset()
    auroc = results["auroc"].cpu().numpy()
    auprc = results["auprc"].cpu().numpy()
    val_auroc[epoch] = auroc
    val_auprc[epoch] = auprc
    print(
        f"Epoch [{epoch+1}/{n_epochs}], val_AUROC: normal:{auroc[0]:.4f} hypo:{auroc[1]:.4f} hyper:{auroc[2]:.4f}"
    )
    print(
        f"Epoch [{epoch+1}/{n_epochs}], val_AUPRC: normal:{auprc[0]:.4f} hypo:{auprc[1]:.4f} hyper:{auprc[2]:.4f}"
    )

    val_time_per_epoch.append(time.time() - start_time)

    # save the model
    model_path = os.path.join(
        "..", "saved_models", f"weights_epoch{epoch+1}_{save_prefix}.pth"
    )
    torch.save(model.state_dict(), model_path)

    # Update subsampled indices for the next epoch
    dataset.on_epoch_end()
    val_dataset.on_epoch_end()

    # save the training and validation metrics
    metrics_path = os.path.join("..", "saved_models", f"metrics_{save_prefix}.npz")
    train_time_per_epoch2 = np.array(train_time_per_epoch, dtype=np.float32)
    val_time_per_epoch2 = np.array(val_time_per_epoch, dtype=np.float32)
    np.savez(
        metrics_path,
        train_auroc=train_auroc,
        val_auroc=val_auroc,
        train_auprc=train_auprc,
        val_auprc=val_auprc,
        train_time_per_epoch=train_time_per_epoch2,
        val_time_per_epoch=val_time_per_epoch2,
    )

    # early stopping
    current_metric = np.mean(val_auroc[epoch]) + np.mean(val_auprc[epoch])
    if current_metric > best_metric:
        best_metric = current_metric
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping!")
            break

    # Clear cache
    torch.cuda.empty_cache()
