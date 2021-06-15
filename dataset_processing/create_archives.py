import os
from utils import archive_subdataset
from datasets import TransformedDataset


def main(root):
    for dataset_name in ["CRC_transformed",
                         "PCam_transformed"]:
        num_per_class_list = [100, 200, 300, 500, 1000]
        archive_names = [os.path.join(root, dataset_name + f"_{num_per_class}_per_class")
                         for num_per_class in num_per_class_list]
        split = "train"
        archive_subdataset(os.path.join(root, dataset_name),
                           split,
                           archive_names=archive_names,
                           num_per_class_list=num_per_class_list)
        split = "valid"
        valid_num_per_class_list = [200 for _ in range(len(num_per_class_list))]
        archive_subdataset(os.path.join(root, dataset_name),
                           split,
                           archive_names=archive_names,
                           num_per_class_list=valid_num_per_class_list)


if __name__ == "__main__":
    root_dir = "/home/zhan8425/scratch/HistoKTdata"
    # root_dir = "C:\\Users\\ryanr\\Desktop\\Summer_Research\\HistoKT\\.adas-data"
    main(root_dir)
