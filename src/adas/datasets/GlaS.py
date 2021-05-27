import torch

from torchvision.datasets.utils import verify_str_arg
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
from typing import Any

import pandas as pd
import os


class GlaS(Dataset):
    """
    Dataset definition for GlaS
    """

    db_name = 'Warwick QU Dataset (Released 2016_07_08)'
    csv_file = 'Grade.csv'

    classes = [' benign', ' malignant']

    # classes = [' adenomatous', ' healthy', ' poorly differentiated', ' moderately differentiated', ' moderately-to-poorly differentated']

    def __init__(self, root, transform=None, split="train", loader=default_loader) -> None:
        """
        Args:
            transform (callable, optional): A function/transform that  takes in an
                PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            root (string): Root directory of the ImageNet Dataset.
            split (string, optional): The dataset split, supports ``train``,
                ``valid``, or ``test``.
            loader (callable, optional): A function to load an image given its
                path. Defaults to default_loader defined in torchvision

        Attributes:
            self.samples (list): a list of (image_path, label)
        """
        self.root = root
        self.transform = transform
        self.split = verify_str_arg(split, "split", ("train", "valid", "test"))
        self.loader = loader

        # get the csv file path
        csv_file_path = os.path.join(self.root, self.db_name, self.csv_file)

        # read the csv file
        GlaS_data = pd.read_csv(csv_file_path)

        if self.split == "train":
            out_df = GlaS_data.iloc[80:, :]

        elif self.split == "valid":
            out_df = GlaS_data.iloc[:60, :]

        elif self.split == "test":
            out_df = GlaS_data.iloc[60:80, :]

        # get the image paths
        self.full_image_paths = [os.path.join(self.root, self.db_name, image_name + ".bmp")
                                 for image_name in out_df["name"]]

        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}

        self.samples = [(self.full_image_paths[image_name], self.class_to_idx[class_name])
                        for image_name, class_name in zip(out_df["name"], out_df[" grade (GlaS)"])]


    def __getitem__(self, item) -> [Any, torch.Tensor]:

        path, label = self.samples[item]

        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, torch.tensor(label)


    def __len__(self) -> int:
        return len(self.samples)
