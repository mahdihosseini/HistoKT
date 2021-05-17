
import torch
from torchvision.datasets.utils import verify_str_arg
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import sys
import os

mod_name = vars(sys.modules[__name__])['__name__']

if 'adas.' in mod_name:
    from ..ADP_utils.classesADP import classesADP
else:
    from ADP_utils.classesADP import classesADP

class ADP_dataset(Dataset):
    db_name = 'ADP V1.0 Release'
    ROI = 'img_res_1um_bicubic'
    csv_file = 'ADP_EncodedLabels_Release1_Flat.csv'
    
    def __init__(self, 
                level, 
                transform, 
                root, 
                split = 'train', 
                loader = default_loader): 
        '''
        Args:
            level (str): a string corresponding to a dict
                defined in "ADP_scripts\classes\classesADP.py"
                defines the hierarchy to be trained on
            transform (callable, optional): A function/transform that  takes in an
                PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            root (string): Root directory of the ImageNet Dataset.
            split (string, optional): The dataset split, supports ``train``, 
                ``valid``, or ``test``.
            loader (callable, optional): A function to load an image given its
                path. Defaults to default_loader defined in torchvision

        Attributes:
            self.full_image_paths (list) : a list of image paths
            self.class_labels (np.ndarray) : a numpy array of class labels 
                (num_samples, num_classes)
        '''
        
        self.root = root
        self.split = verify_str_arg(split, "split", ("train", "valid", "test"))
        self.transform = transform
        self.loader = loader

        # getting paths:
        csv_file_path = os.path.join(self.root, self.db_name, self.csv_file)

        ADP_data = pd.read_csv(filepath_or_buffer=csv_file_path, header=0) # reads data and returns a pd.dataframe
        # rows are integers starting from 0, columns are strings: e.g. "Patch Names", "E", ...

        split_folder = os.path.join(self.root, self.db_name, 'splits')

        if self.split == "train":
            train_inds = np.load(os.path.join(split_folder, 'train.npy'))
            out_df = ADP_data.loc[train_inds, :]

        elif self.split == "valid":
            valid_inds = np.load(os.path.join(split_folder, 'valid.npy'))
            out_df = ADP_data.loc[valid_inds, :]

        elif self.split == "test":
            test_inds = np.load(os.path.join(split_folder, 'test.npy'))
            out_df = ADP_data.loc[test_inds, :]

        self.full_image_paths = [os.path.join(self.root, self.db_name, self.ROI, image_name) for image_name in out_df['Patch Names']]
        self.class_labels = out_df[classesADP[level]['classesNames']].to_numpy(dtype=np.float32)

    def __getitem__(self, idx) -> torch.Tensor:
        
        path = self.full_image_paths[idx]
        label = self.class_labels[idx]

        sample = self.loader(path) # Loading image
        if self.transform is not None: # PyTorch implementation
            sample = self.transform(sample)

        return sample, torch.tensor(label)

    def __len__(self) -> int:
        return(len(self.full_image_paths))