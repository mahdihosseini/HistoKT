import pickle
import torch
from torchvision.datasets.utils import verify_str_arg
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
import os
from typing import Any


class CRC_transformed(Dataset):

    def __init__(self, root, split="train", transform=None, loader=default_loader) -> None:
        """

        Args:
            root (string):
                Directory of the transformed dataset, e.g. /home/CRC_transformed
            split (string, optional): The dataset split, supports ``train``,
                ``valid``, or ``test``.
            transform (callable, optional): A function/transform that  takes in an
                PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            loader (callable, optional): A function to load an image given its
                path. Defaults to default_loader defined in torchvision
        """

        self.root = root
        self.split = verify_str_arg(split, "split", ("train", "valid", "test"))
        self.transform = transform
        self.loader = loader

        # getting samples from preprocessed pickle file
        self.samples = pickle.load(open(os.path.join(self.root, self.split+".pickle"), "rb"))
        self.samples = [(os.path.join(self.root, path), label) for path, label in self.samples]
        self.class_to_idx = pickle.load(open(os.path.join(self.root, "class_to_idx.pickle"), "rb"))

    def __getitem__(self, idx) -> [Any, torch.Tensor]:

        path, label = self.samples[idx]

        sample = self.loader(path)  # Loading image
        if self.transform is not None:  # PyTorch implementation
            sample = self.transform(sample)

        return sample, torch.tensor(label)

    def __len__(self) -> int:
        return len(self.samples)
