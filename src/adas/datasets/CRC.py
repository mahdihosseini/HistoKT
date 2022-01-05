"""
CRC Dataset, retrieved from:https://zenodo.org/record/1214456#.YLTgV6hKhhE

Kather, Jakob Nikolas, Halama, Niels, & Marx, Alexander. (2018). 
100,000 histological images of human colorectal cancer and healthy tissue (v0.1) [Data set]. Zenodo.
https://doi.org/10.5281/zenodo.1214456
"""
import os

from torchvision.datasets.folder import ImageFolder
from torchvision.datasets.utils import verify_str_arg


class CRC(ImageFolder):
    """`CRC
    Args:
        root (string): Root directory of the CRC Dataset.
        split (string, optional): The dataset split, supports ``train``, or
            ``test``.
        transform (callable, optional): A function/transform that  takes in an
            PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes
            in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its
            path.

     Attributes:
        classes (list): List of the class name tuples.
        class_to_idx (dict): Dict with items (class_name, class_index).
        targets (list): The class_index value for each image in the dataset
        split_folder (os.path): a path to the split folders
    """

    train_folder = "NCT-CRC-HE-100K-NONORM"
    test_folder = "CRC-VAL-HE-7K"
    classes_ = ["ADI", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
    # classes_ = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]

    def __init__(self, root, split="train", **kwargs):
        root = self.root = os.path.expanduser(root)
        self.split = verify_str_arg(split, "split", ("train", "test"))
        if self.split == "train":
            self.root = os.path.join(root, self.train_folder)
        elif self.split == "test":
            self.root = os.path.join(root, self.test_folder)

        super(CRC, self).__init__(self.root, **kwargs)

        self.classes = self.classes_
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        samples = self.make_dataset(self.root, self.class_to_idx, self.extensions)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            raise RuntimeError(msg)

        self.samples = samples
        self.targets = [s[1] for s in samples]
