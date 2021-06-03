import os

from preprocessing.transforms import ProcessImages
from datasets import AIDPATH, AJLymph, BACH, CRC, GlaS, MHIST, OSDataset, PCam

pixel_res = {
    "ADP": 1.0,
    "MHIST": 1.25,
    "BACH": 0.42,
    "AIDPATH": 0.5,
    "AJLymph": 0.25,
    "PCam": 0.972,
    "CRC": 0.5,
    "GlaS": 0.62005,
    "Osteosarcoma": 1.0,
}


def main(root):

    dataset_list = [
                    # (AIDPATH(root, split="train", transform=None),
                    #  "AIDPATH_transformed",
                    #  {"train": 80, "valid": 10, "test": 10},
                    #  pixel_res["AIDPATH"]),
                    # (AJLymph(root, split="train", transform=None),
                    #  "AJ-Lymph_transformed",
                    #  {"train": 80, "valid": 10, "test": 10},
                    #  pixel_res["AJLymph"]),
                    # (BACH(root, split="train", transform=None),
                    #  "BACH_transformed",
                    #  {"train": 80, "valid": 10, "test": 10},
                    #  pixel_res["BACH"]),
                    # (CRC(root, split="train", transform=None),
                    #  "CRC_transformed",
                    #  {"train": 80, "valid": 20},
                    #  pixel_res["CRC"]),
                    # (CRC(root, split="test", transform=None),
                    #  "CRC_transformed",
                    #  {"test": 100},
                    #  pixel_res["CRC"]),
                    # (GlaS(root, split="train", transform=None),
                    #  "GlaS_transformed",
                    #  {"train": 100},
                    #  pixel_res["GlaS"]),
                    # (GlaS(root, split="valid", transform=None),
                    #  "GlaS_transformed",
                    #  {"valid": 100},
                    #  pixel_res["GlaS"]),
                    # (GlaS(root, split="test", transform=None),
                    #  "GlaS_transformed",
                    #  {"test": 100},
                    #  pixel_res["GlaS"]),
                    # (MHIST(root, split="train", transform=None),
                    #  "MHIST_transformed",
                    #  {"train": 80, "valid": 20},
                    #  pixel_res["MHIST"]),
                    # (MHIST(root, split="test", transform=None),
                    #  "MHIST_transformed",
                    #  {"test": 100},
                    #  pixel_res["MHIST"]),
                    # (OSDataset(root, split="train", transform=None),
                    #  "OSDataset_transformed",
                    #  {"train": 100},
                    #  pixel_res["Osteosarcoma"]),
                    # (OSDataset(root, split="valid", transform=None),
                    #  "OSDataset_transformed",
                    #  {"valid": 100},
                    #  pixel_res["Osteosarcoma"]),
                    # (OSDataset(root, split="test", transform=None),
                    #  "OSDataset_transformed",
                    #  {"test": 100},
                    #  pixel_res["Osteosarcoma"]),
                    (PCam(root, split="train", transform=None),
                     "PCam_transformed",
                     {"train": 100},
                     pixel_res["PCam"]),
                    (PCam(root, split="valid", transform=None),
                     "PCam_transformed",
                     {"valid": 100},
                     pixel_res["PCam"]),
                    (PCam(root, split="test", transform=None),
                     "PCam_transformed",
                     {"test": 100},
                     pixel_res["PCam"]),
                    ]

    for dataset, rel_target_folder, split_dict, scale in dataset_list:
        print("Dataset information")
        print(dataset)
        print(f"target folder: {rel_target_folder}, Split: {split_dict}, Pixel Res: {scale}")
        image_processor = ProcessImages(dataset,
                                        target_folder=os.path.join(root, rel_target_folder),
                                        split_dict=split_dict)

        image_processor.process(scale=scale, target_dim=(272, 272),
                                percent_overlap=0.25,
                                show_imgs=False,
                                low_contrast=True,
                                lower_percentile=5,
                                upper_percentile=99)


if __name__ == "__main__":
    #root_dir = ".adas-data"
    home = os.environ.get("HOME")
    root_dir = f"{home}/scratch/HistoKTdata"
    main(root_dir)
