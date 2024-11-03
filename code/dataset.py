from multiprocessing import Pool

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# For handling missing sources, specify number of categorical and contniuous
# features here as a tuple (n_cat, n_cont).
sources_to_num_features = {
    "static": (7, 3),
    "unit_info": (3, 0),
    "addx": (1, 0),
    "diagnosis": (2, 0),
    "lab": (1, 1),
    "IO": (1, 1),
    "IO_num_reg": (0, 4),
    "past_history": (1, 0),
    "treatment": (1, 0),
    "med": (5, 1),
    "infusion": (1, 1),
    "GCS": (1, 0),
    "sedation": (1, 0),
    "HR": (0, 1),
    "RR": (0, 1),
    "SpO2": (0, 1),
    "Temp": (1, 1),
    "nibp_mean": (0, 1),
    "ibp_mean": (0, 1),
    "nibp_systolic": (0, 1),
    "ibp_systolic": (0, 1),
    "nibp_diastolic": (0, 1),
    "ibp_diastolic": (0, 1),
}

# max_n_features = max(ncat + ncont for ncat, ncont in sources_to_num_features.values())


def worker_cleanup():
    _hdf5_file.close()


def worker_init(hdf5_path):
    global _hdf5_file
    _hdf5_file = h5py.File(hdf5_path, "r")
    import atexit

    atexit.register(worker_cleanup)


# Custom dataset class for blood glucose prediction task
class PatientDataset(Dataset):

    def __init__(
        self,
        hdf5_file,
        example_id_path,
        max_seq_len=511,
        do_subsampling=False,
        shuffle=True,
        load_to_memory=False,
        num_workers=None,
    ):
        """
        Args:
            hdf5_file (string): Path to the HDF5 file containing patient data.
            example_id_path (string): Path to the numpy file containing example IDs
                (patientunitstayid, window number) for each class.
            max_seq_len (int): Maximum sequence length for each source.
            do_subsampling (bool): Whether to subsample the data to have the same number of samples per class.
            shuffle (bool): Whether to shuffle the indices at the beginning of each epoch.
            load_to_memory (bool): Whether to load the entire dataset to memory.
            num_workers (int): Number of workers for data loading into memory.
        """
        self.file = hdf5_file
        self.do_subsampling = do_subsampling
        self.max_seq_len = max_seq_len  # Maximum sequence length for each source
        self.shuffle = shuffle
        self.load_to_memory = load_to_memory
        self.num_workers = num_workers
        # For training, the same number of samples per class will be used.
        self.example_ids = np.load(example_id_path)
        self.all_indices = list(self.example_ids.values())
        if do_subsampling:
            self.n_samples_per_class = min(
                *(len(self.example_ids[k]) for k in self.example_ids)
            )
            self.indices = self.subsample_indices()
        else:
            self.indices = np.concatenate(self.all_indices, axis=0)
        if self.shuffle:
            np.random.shuffle(self.indices)  # Shuffle to mix classes
        if self.load_to_memory:
            self.load_data()

    @staticmethod
    def load_patient_data(p_key, filehandler=None):
        if filehandler is None:
            global _hdf5_file
            f = _hdf5_file
        else:
            f = filehandler
        result = {}
        p = f[p_key]
        n_win = p["num_windows"][()]
        if n_win == 0:
            return
        result[p_key] = {}
        result[p_key]["static_cat"] = p["static_cat"][:]
        result[p_key]["static_cont"] = p["static_cont"][:]
        for k in sources_to_num_features:  # iterate over sources
            if k in p:
                result[p_key][k] = {}
                for k2, v2 in p[k].items():  # iterate over features and offsets
                    result[p_key][k][k2] = v2[:]
        # collect window information
        for i in range(n_win):
            window_group = p[f"window_{i}"]
            window = {}
            window["label"] = window_group["label"][()]
            window["label_offset"] = window_group["label_offset"][()]
            # iterate over sources to collect window length for every source
            for k in sources_to_num_features:
                if k in window_group:
                    window[k] = window_group[k][()]
            result[p_key][f"window_{i}"] = window
        return result

    def load_data(self):
        self.data = {}
        with h5py.File(self.file, "r") as f:
            p_keys = [p_key for p_key in f.keys() if p_key.startswith("patient_")]
        if self.num_workers:
            with Pool(
                processes=self.num_workers,
                initializer=worker_init,
                initargs=(self.file,),
            ) as pool:
                results = list(
                    tqdm(
                        pool.imap(self.load_patient_data, p_keys),
                        total=len(p_keys),
                        desc="Loading data to memory...",
                    )
                )
                # Merge the results into a single data dictionary
                for result in results:
                    if result is not None:
                        self.data.update(result)

        else:
            with h5py.File(self.file, "r") as f:
                for p_key in tqdm(p_keys, desc="Loading data to memory..."):
                    result = self.load_patient_data(p_key, filehandler=f)
                    if result is not None:
                        self.data.update(result)

    def subsample_indices(self):
        """Subsamples indices for each class."""
        subsampled_indices = []
        for indices in self.all_indices:
            idx = np.random.choice(
                len(indices), self.n_samples_per_class, replace=False
            )
            subsampled_indices.append(indices[idx])

        subsampled_indices = np.concatenate(subsampled_indices, axis=0)
        return subsampled_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # get patientunitstayid and window number
        pu_id, w_num = self.indices[idx]
        if self.load_to_memory:
            f = self.data
        else:
            f = h5py.File(self.file, "r")

        p = f[f"patient_{pu_id}"]
        window = p[f"window_{w_num}"]
        if not self.load_to_memory:
            window = {k: v[()] for k, v in window.items()}
        label = torch.tensor(window["label"], dtype=torch.int32)
        label_offset = torch.tensor(window["label_offset"], dtype=torch.float32)

        sources, offsets = {}, {}

        for k, (n_cat, n_cont) in sources_to_num_features.items():
            if k == "static":
                sources[k] = {
                    "categorical": torch.tensor(p["static_cat"][:], dtype=torch.int32),
                    "continuous": torch.tensor(
                        p["static_cont"][:], dtype=torch.float32
                    ),
                }
                offsets[k] = torch.zeros(1, dtype=torch.float32)
                continue
            if k in window:
                win_len = window[k]
                categorical = None
                if n_cat:
                    categorical = p[k]["cat_features"][0:win_len, :][
                        -self.max_seq_len :
                    ]
                    categorical = torch.tensor(categorical, dtype=torch.int32)
                continuous = None
                if n_cont:
                    continuous = p[k]["cont_features"][0:win_len, :][
                        -self.max_seq_len :
                    ]
                    continuous = torch.tensor(continuous, dtype=torch.float32)
                sources[k] = {
                    "categorical": categorical,
                    "continuous": continuous,
                }
                offsets[k] = torch.tensor(
                    p[k]["offsets"][0:win_len][-self.max_seq_len :],
                    dtype=torch.float32,
                )
            else:  # handle missing sources
                # use special token (-1) for categorical features and zeros (mean) for continuous features
                sources[k] = {
                    "categorical": (
                        -1 * torch.ones((1, n_cat), dtype=torch.int32)
                        if n_cat
                        else None
                    ),
                    "continuous": (
                        torch.zeros((1, n_cont), dtype=torch.float32)
                        if n_cont
                        else None
                    ),
                }
                offsets[k] = torch.zeros(1, dtype=torch.float32)

        if not self.load_to_memory:
            f.close()

        return (sources, offsets), (label, label_offset)

    def on_epoch_end(self):
        """Update indices after each epoch."""
        if self.do_subsampling:
            self.indices = self.subsample_indices()
        if self.shuffle:
            np.random.shuffle(self.indices)


def collate_fn(batch):
    source_data, label_data = zip(*batch)
    batch_size = len(source_data)
    seq_len_per_source = {}
    max_seq_len_per_source = {}
    mask = {}  # True for masked values
    batched_sources = {}
    batched_offsets = {}
    for k, (n_cat, n_cont) in sources_to_num_features.items():
        seq_len_per_source[k] = torch.tensor(
            [len(offsets[k]) for _, offsets in source_data]
        )
        max_seq_len_per_source[k] = max(seq_len_per_source[k]).item()
        mask[k] = torch.ones(batch_size, max_seq_len_per_source[k], dtype=torch.bool)
        for i, l in enumerate(seq_len_per_source[k]):
            mask[k][i, :l] = False
        # concatenate sources and offsets across the batch
        batched_sources[k] = {
            "categorical": (
                torch.cat([s[k]["categorical"] for s, _ in source_data], dim=0)
                if n_cat
                else None
            ),
            "continuous": (
                torch.cat([s[k]["continuous"] for s, _ in source_data], dim=0)
                if n_cont
                else None
            ),
        }
        batched_offsets[k] = torch.cat(
            [offsets[k] for _, offsets in source_data], dim=0
        )

    # convert labels to batched tensors
    labels = torch.tensor([l[0] for l in label_data], dtype=torch.int64)
    label_offsets = torch.tensor([l[1] for l in label_data], dtype=torch.float32)
    label_data = (labels, label_offsets)

    source_data = (batched_sources, batched_offsets)
    return source_data, label_data, seq_len_per_source, mask


def move_to_device(model_input, device):
    source_data, label_data, seq_len_per_source, mask = model_input
    labels, label_offsets = label_data
    sources, offsets = source_data
    # labels
    labels, label_offsets = labels.to(device), label_offsets.to(device)
    label_data = (labels, label_offsets)
    # sources and offsets
    for k in sources:
        if sources[k]["categorical"] is not None:
            sources[k]["categorical"] = sources[k]["categorical"].to(device)
        if sources[k]["continuous"] is not None:
            sources[k]["continuous"] = sources[k]["continuous"].to(device)
        offsets[k] = offsets[k].to(device)
    for k in mask:
        mask[k] = mask[k].to(device)

    return source_data, label_data, seq_len_per_source, mask
