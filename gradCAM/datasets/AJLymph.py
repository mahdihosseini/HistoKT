"""
AJ-Lymph dataset
Retrieved from: http://www.andrewjanowczyk.com/use-case-7-lymphoma-sub-type-classification/

CITATION:
Janowczyk A, Madabhushi A. Deep learning for digital pathology image analysis:
A comprehensive tutorial with selected use cases. J Pathol Inform. 2016 Jul 26;7:29.
doi: 10.4103/2153-3539.186902. PMID: 27563488; PMCID: PMC4977982.
"""

from os import path
from pathlib import Path
import traceback

import pandas as pd
import os

from torchvision.datasets.folder import ImageFolder
from torchvision.datasets.utils import verify_str_arg


class AJLymph(ImageFolder):
    """`AJLymph
    Args:
        root (string): Root directory of the AJ-Lymph Dataset.
            all files downloaded should be placed in a folder
            called "AJ-Lymph" and placed in the data folder
        split (string, optional): The dataset split, supports ``train``
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

    train_folder = "AJ-Lymph"

    def __init__(self, root, split="train", **kwargs):
        root = self.root = os.path.expanduser(root)
        self.split = verify_str_arg(split, "split", ("train",))

        super(AJLymph, self).__init__(str(os.path.join(root, self.train_folder)), **kwargs)
