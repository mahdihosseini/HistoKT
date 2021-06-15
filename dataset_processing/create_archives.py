import os
from utils import archive_subdataset
from datasets import TransformedDataset


def main(root):
    for dataset_name in ["CRC_transformed",
                         "PCam_transformed"]:
        split = "train"
        for num in [100, 200, 300, 500, 1000]:
            dataset = TransformedDataset(os.path.join(root, dataset_name), split=split)
            archive_subdataset(dataset, os.path.join(root, dataset_name+f"_{num}_per_class"), num_per_class=num)
        split = "valid"
        for num in [100, 200, 300, 500, 1000]:
            dataset = TransformedDataset(os.path.join(root, dataset_name), split=split)
            archive_subdataset(dataset, os.path.join(root, dataset_name + f"_{num}_per_class"), num_per_class=200)


if __name__ == "__main__":
    root_dir = "/home/zhan8425/scratch/HistoKTdata"
    main(root_dir)
