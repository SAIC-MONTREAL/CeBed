from cebed.datasets.sionna import (
    MultiDomainOfflineSionnaDataset,
    OfflineSionnaDataset,
)

DATASETS = {
    "SionnaOfflineMD": MultiDomainOfflineSionnaDataset,
    "SionnaOffline": OfflineSionnaDataset,
}


def get_dataset_class(dataset_name: str):
    """Return the dataset class with the given name.

    :param dataset_name: Name of the dataset to get the function of.
    (Must be a part of the DATASETS dict)

    return: The dataset class

    """

    if dataset_name not in DATASETS:
        raise NotImplementedError(f"Dataset not found: {dataset_name}")

    return DATASETS[dataset_name]
