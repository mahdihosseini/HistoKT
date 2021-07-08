import torch
import os
import numpy as np
import pandas as pd
from torchvision.datasets.utils import verify_str_arg
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
from typing import Any


class BCSSDataset(Dataset):

    def __init__(self,
                 root,
                 split="train",
                 transform=None,
                 loader=default_loader,
                 multi_labelled=True,
                 class_labels=False) -> None:
        """
        Retrieved from: https://bcsegmentation.grand-challenge.org/
        Args:
            root (string):
                Directory of the transformed dataset, e.g. "/home/BCSS_transformed"
            split (string, optional): The dataset split, supports ``train``,
                ``valid``, or ``test``.
            transform (callable, optional): A function/transform that  takes in an
                PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            loader (callable, optional): A function to load an image given its
                path. Defaults to default_loader defined in torchvision
            multi_labelled (bool): a boolean controlling whether the output labels are a multilabelled array
                or an index corresponding to the single label
        """

        self.root = root
        self.split = verify_str_arg(split, "split", ("train", "valid", "test"))
        self.transform = transform
        self.loader = loader

        # getting samples from preprocessed pickle file
        if multi_labelled:
            df = pd.read_csv(os.path.join(self.root, self.split+".csv"), index_col="image")
        else:
            df = pd.read_csv(os.path.join(self.root, self.split+"_with_norm_mass.csv"), index_col="image")
        self.samples = [(image, label) for image, label in zip(df.index, df.to_records(index=False))]

        if class_labels:
            self.class_labels = df.to_numpy(dtype=np.float32)

        if multi_labelled:
            self.samples = [(os.path.join(self.root, path), list(label)) for path, label in self.samples]
        else:
            self.samples = [(os.path.join(self.root, path), np.argmax(list(label))) for path, label in self.samples]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(df.columns)}

    def __getitem__(self, idx) -> [Any, torch.Tensor]:

        path, label = self.samples[idx]

        sample = self.loader(path)  # Loading image
        if self.transform is not None:  # PyTorch implementation
            sample = self.transform(sample)

        return sample, torch.tensor(label, dtype=torch.int64)

    def __len__(self) -> int:
        return len(self.samples)
