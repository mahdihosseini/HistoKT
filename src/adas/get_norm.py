from utils import get_mean_and_std
from datasets import TransformedDataset
from torchvision.transforms import ToTensor
import os


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
        dataset_object = TransformedDataset(transform=ToTensor(),
                                            root=os.path.join(root, dataset),
                                            split='train')

        mean, std = get_mean_and_std(dataset_object, num_workers=4)
        output_dict[dataset] = {"mean": mean, "std": std}

    return output_dict


if __name__ == "__main__":
    root_dir = "~/scratch/HistoKTdata"
    print(main(root_dir))