from utils import get_mean_and_std
from datasets import TransformedDataset
from torchvision.transforms import ToTensor
import os
import time


def main(root):
    output_dict = {}
    for dataset in ["AIDPATH_transformed",
                    "AJ-Lymph_transformed",
                    "BACH_transformed",
                    "CRC_transformed",
                    "GlaS_transformed",
                    "MHIST_transformed",
                    "OSDataset_transformed",
                    "PCam_transformed"]:

        print(f"Loading dataset: {dataset}")
        start_time = time.time()
        dataset_object = TransformedDataset(transform=ToTensor(),
                                            root=os.path.join(root, dataset),
                                            split='train')
        load_time = time.time()
        print(f"Loaded dataset, took time:", load_time-start_time)
        mean, std = get_mean_and_std(dataset_object, num_workers=4)
        print("calculated mean and std, took time:", time.time()-load_time)
        output_dict[dataset] = {"mean": mean, "std": std}

    return output_dict


if __name__ == "__main__":
    home = os.environ.get("HOME")
    root_dir = f"{home}/scratch/HistoKTdata"
    print(main(root_dir))
