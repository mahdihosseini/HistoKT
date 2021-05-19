"""
BACH dataset from https://iciar2018-challenge.grand-challenge.org/home/
retrieved from https://zenodo.org/record/3632035#.YKPhB6hKhhF
CITATION:
Guilherme Aresta, Teresa Araújo, Scotty Kwok, Sai Saketh Chennamsetty, Mohammed Safwan, Varghese Alex, Bahram Marami,
Marcel Prastawa, Monica Chan, Michael Donovan, Gerardo Fernandez, Jack Zeineh, Matthias Kohl, Christoph Walz,
Florian Ludwig, Stefan Braunewell, Maximilian Baust, Quoc Dang Vu, Minh Nguyen Nhat To, Eal Kim, Jin Tae Kwak,
Sameh Galal, Veronica Sanchez-Freire, Nadia Brancati, Maria Frucci, Daniel Riccio, Yaqi Wang, Lingling Sun,
Kaiqiang Ma, Jiannan Fang, Ismael Kone, Lahsen Boulmane, Aurélio Campilho, Catarina Eloy, António Polónia, Paulo Aguiar,
BACH: Grand challenge on breast cancer histology images,
Medical Image Analysis,
Volume 56,
2019,
Pages 122-139,
ISSN 1361-8415,
https://doi.org/10.1016/j.media.2019.05.010.
(https://www.sciencedirect.com/science/article/pii/S1361841518307941)
Abstract: Breast cancer is the most common invasive cancer in women, affecting more than 10% of women worldwide.
Microscopic analysis of a biopsy remains one of the most important methods to diagnose the type of breast cancer.
This requires specialized analysis by pathologists, in a task that i) is highly time- and cost-consuming and
ii) often leads to nonconsensual results.
The relevance and potential of automatic classification algorithms using hematoxylin-eosin stained histopathological
images has already been demonstrated, but the reported results are still sub-optimal for clinical use.
With the goal of advancing the state-of-the-art in automatic classification, the Grand Challenge on
BreAst Cancer Histology images (BACH) was organized in conjunction with the 15th International Conference on
Image Analysis and Recognition (ICIAR 2018). BACH aimed at the classification and localization of clinically relevant
histopathological classes in microscopy and whole-slide images from a large annotated dataset, specifically compiled and
made publicly available for the challenge. Following a positive response from the scientific community,
a total of 64 submissions, out of 677 registrations, effectively entered the competition.
The submitted algorithms improved the state-of-the-art in automatic classification of breast cancer with microscopy
images to an accuracy of 87%. Convolutional neuronal networks were the most successful methodology in the BACH
challenge. Detailed analysis of the collective results allowed the identification of remaining challenges in the field
and recommendations for future developments. The BACH dataset remains publicly available as to promote further i
mprovements to the field of automatic classification in digital pathology.
Keywords: Breast cancer; Histology; Digital pathology; Challenge; Comparative study; Deep learning
"""
from os import path
from pathlib import Path
import traceback

import pandas as pd
import os

from torchvision.datasets.folder import ImageFolder
from torchvision.datasets.utils import verify_str_arg


class BACH(ImageFolder):
    """`BACH
    Args:
        root (string): Root directory of the BACH Dataset.
            all files downloaded: "ICIAR2018_BACH_Challenge" and
            "ICIAR2018_BACH_Challenge_TestDataset"
            should be placed in the data folder
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

    train_folder = "ICIAR2018_BACH_Challenge"
    test_folder = "ICIAR2018_BACH_Challenge_TestDataset"

    def __init__(self, root, split="train", **kwargs):
        root = self.root = os.path.expanduser(root)
        self.split = verify_str_arg(split, "split", ("train", "test"))
        if split == "train":
            self.root = os.path.join(root, self.train_folder, "Photos")
        if split == "test":
            self.root = os.path.join(root, self.test_folder, "Photos")
        root = Path(self.root)

        super(BACH, self).__init__(str(root), **kwargs)