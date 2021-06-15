
import torch

from torchvision.datasets.utils import verify_str_arg
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
from typing import Any

import pandas as pd
import os
import pickle
import h5py
from skimage import io
from pathlib import Path
import gzip
import shutil


class PCam(Dataset):
    """
    Dataset definition for PCam
    """

    db_folder = "PCam"
    db_name = "camelyonpatch_level_2_split_"

    classes = ['non_tumor', 'tumor']

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
        self.root = os.path.join(root, self.db_folder)
        self.transform = transform
        self.split = verify_str_arg(split, "split", ("train", "valid", "test"))
        self.loader = loader

        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}

        path_to_images = os.path.join(self.root, self.db_name + self.split + "_x.h5")
        path_to_labels = os.path.join(self.root, self.db_name + self.split + "_y.h5")

        if os.path.exists(os.path.join(self.root, self.split, self.split+".pickle")):
            self.samples = pickle.load(open(os.path.join(self.root, self.split, self.split + ".pickle"), "rb"))
        elif os.path.exists(path_to_images):
            self.create_splits(path_to_images, path_to_labels)
            self.samples = pickle.load(open(os.path.join(self.root, self.split, self.split + ".pickle"), "rb"))
        else:
            self.unzip_files(path_to_images, path_to_labels)
            self.create_splits(path_to_images, path_to_labels)
            self.samples = pickle.load(open(os.path.join(self.root, self.split, self.split + ".pickle"), "rb"))

        self.samples = [(os.path.join(self.root, sample[0]), sample[1]) for sample in self.samples]

    def __getitem__(self, idx) -> [Any, torch.Tensor]:

        sample, label = self.samples[idx]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, torch.tensor(label)

    def __len__(self) -> int:
        return len(self.samples)

    def create_splits(self, path_to_images, path_to_labels):
        # create splits given path to images and path to image labels
        # these paths lead to the unzipped archive files
        samples = []

        with h5py.File(path_to_images, "r") as img_archive:
            images = img_archive["x"]

            with h5py.File(path_to_labels, "r") as label_archive:
                labels = label_archive["y"]

                for i in range(images.shape[0]):
                    save_path = Path(self.root, self.split, f"{self.split}-{i}.png")
                    samples.append((str(save_path.relative_to(self.root)), labels[i][0][0].item()))
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    io.imsave(str(save_path), images[i])

        with open(os.path.join(self.root, self.split, self.split + ".pickle"), "wb") as f:
            pickle.dump(samples, f)

    @staticmethod
    def unzip_files(path_to_images, path_to_labels):
        if not os.path.exists(path_to_images+".gz"):
            raise FileNotFoundError(
                f"Need image archives from https://github.com/basveeling/pcam: {path_to_images+'.gz'}")
        with gzip.open(path_to_images+".gz", 'rb') as f_in:
            with open(path_to_images, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        if not os.path.exists(path_to_labels+".gz"):
            raise FileNotFoundError(
                f"Need image archives from https://github.com/basveeling/pcam: {path_to_labels+'.gz'}")
        with gzip.open(path_to_labels+".gz", 'rb') as f_in:
            with open(path_to_labels, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
