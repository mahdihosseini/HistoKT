from pathlib import Path

import pandas as pd
import os

from torchvision.datasets.folder import ImageFolder
from torchvision.datasets.utils import verify_str_arg

class MHIST(ImageFolder):
    """`TinyImageNet
    Args:
        root (string): Root directory of the ImageNet Dataset.
        split (string, optional): The dataset split, supports ``train``, or
            ``val``.
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
        wnids (list): List of the WordNet IDs.
        wnid_to_idx (dict): Dict with items (wordnet_id, class_index).
        imgs (list): List of (image path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, split='train', **kwargs):
        root = self.root = os.path.expanduser(root)
        self.split = verify_str_arg(split, "split", ("train", "test"))
        self.root = root
        root = Path(root)
        self.classes = ['SSA', 'HP']
        self.class_to_idx = {cls: idx
                             for idx, cls in enumerate(self.classes)}

        data = pd.read_csv(str(root / 'annotations.csv'))
        _len = len(data['Image Name'])
        self.images = list()
        self.targets = list()
        # self.transform = transform
        if not (root / split).exists():
            Path(root / split).mkdir(parents=True)
            Path(root / split / 'SSA').mkdir(parents=True)
            Path(root / split / 'HP').mkdir(parents=True)
            for i in range(_len):
                if data['Partition'][i] == split:
                    (root / 'images' / data['Image Name'][i]).rename(
                        root / split / data['Majority Vote Label'][i] / data['Image Name'][i])
        super(MHIST, self).__init__(self.split_folder, **kwargs)

    @ property
    def split_folder(self):
        return os.path.join(self.root, self.split)