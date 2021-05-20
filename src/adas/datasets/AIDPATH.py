"""
Dataset of segmented glomeruli
Retrieved from:
Bueno, Gloria; Gonzalez-Lopez, Lucia; García-Rojo, Marcial ; Laurinavicius, Arvydas (2020), “Data for glomeruli characterization in histopathological images”, Mendeley Data, V3, doi: 10.17632/k7nvtgn2x6.3
"""

import os

from torchvision.datasets.folder import ImageFolder
from torchvision.datasets.utils import verify_str_arg


class AIDPATH(ImageFolder):
    """`AIDPATH
    Args:
        root (string): Root directory of the AJ-Lymph Dataset.
            all files downloaded should be placed in a folder
            called "AJ-Lymph" and placed in the data folder
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

    db_folder = "AIDPATH_kidney/DATASET_B_DIB"

    def __init__(self, root, split="train", **kwargs):
        root = self.root = os.path.expanduser(root)
        self.split = verify_str_arg(split, "split", ("train", "test"))

        super(AIDPATH, self).__init__(str(os.path.join(root, self.db_folder)), **kwargs)
