"""
Osteosarcoma data from UT Southwestern/UT Dallas for Viable and Necrotic Tumor Assessment
Retrieved from: https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52756935#52756935bcab02c187174a288dbcbf95d26179e8

Osteosarcoma is the most common type of bone cancer that occurs in adolescents in the age of 10 to 14 years.
The dataset is composed of Hematoxylin and eosin (H&E) stained osteosarcoma histology images.
The data was collected by a team of clinical scientists at University of Texas Southwestern Medical Center, Dallas.
Archival samples for 50 patients treated at Children’ s Medical Center, Dallas, between 1995 and 2015, were used to
create this dataset. Four patients (out of 50) were selected by pathologists based on diversity of tumor specimens
after surgical resection. The images are labelled as Non-Tumor, Viable Tumor and Necrosis according to the predominant
cancer type in each image. The annotation was performed by two medical experts. All images were divided between two
pathologists for the annotation activity. Each image had a single annotation as any given image was annotated by only
one pathologist. The dataset consists of 1144 images of size 1024 X 1024 at 10X resolution with the following
distribution: 536 (47%) non-tumor images, 263 (23%) necrotic tumor images and 345 (30%) viable tumor tiles.

CITATION:
Leavey, P., Sengupta, A., Rakheja, D., Daescu, O., Arunachalam, H. B., & Mishra, R. (2019). Osteosarcoma data
from UT Southwestern/UT Dallas for Viable and Necrotic Tumor Assessment [Data set].
The Cancer Imaging Archive. https://doi.org/10.7937/tcia.2019.bvhjhdas

1) Mishra, Rashika, et al. "Histopathological diagnosis for viable and non-viable tumor prediction for osteosarcoma
using convolutional neural network." International Symposium on Bioinformatics Research and Applications.
Springer, Cham, 2017.

2) Arunachalam, Harish Babu, et al. "Computer aided image segmentation and classification for viable and
non-viable tumor identification in osteosarcoma." PACIFIC SYMPOSIUM ON BIOCOMPUTING 2017. 2017.

3) Mishra, Rashika, et al. "Convolutional Neural Network for Histopathological Analysis of Osteosarcoma."
Journal of Computational Biology 25.3 (2018): 313-325.

4) Leavey, Patrick, et al. "Implementation of Computer-Based Image Pattern Recognition Algorithms to Interpret
Tumor Necrosis; a First Step in Development of a Novel Biomarker in Osteosarcoma." PEDIATRIC BLOOD & CANCER.
Vol. 64. 111 RIVER ST, HOBOKEN 07030-5774, NJ USA: WILEY, 2017.

Clark K, Vendt B, Smith K, Freymann J, Kirby J, Koppel P, Moore S, Phillips S, Maffitt D, Pringle M, Tarbox L, Prior F.
The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository,
Journal of Digital Imaging, Volume 26, Number 6, December, 2013, pp 1045-1057. DOI: 10.1007/s10278-013-9622-7
"""

import torch
from torchvision.datasets.utils import verify_str_arg
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
import pandas as pd
import os
from typing import Any
import pickle


class OSDataset(Dataset):
    """
    Dataset definition for Osteosarcoma
    """

    db_name = "Osteosarcoma-UT"
    csv_file = "ML_Features_1144.csv"

    name2path_path = "name2path.p"
    classes = ['Non-Tumor', 'Viable', 'Non-Viable-Tumor', 'viable: non-viable']

    def __init__(self,
                 root,
                 transform=None,
                 split="train",
                 loader=default_loader) -> None:
        """
        Args:
            transform (callable, optional): A function/transform that  takes in an
                PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            root (string): Root directory of the ImageNet Dataset.
            split (string, optional): The dataset split, supports ``train``.
            loader (callable, optional): A function to load an image given its
                path. Defaults to default_loader defined in torchvision

        Attributes:
            self.class_labels (np.ndarray) : a numpy array of class labels
                (num_samples, num_classes)
            self.samples (list): a list of (image_path, label)
        """

        self.root = root
        self.split = verify_str_arg(split, "split", ("train",))
        self.transform = transform
        self.loader = loader

        # getting csv file path
        csv_file_path = os.path.join(self.root, self.db_name, self.csv_file)

        # reading csv into pandas DataFrame
        self.OSData = pd.read_csv(csv_file_path)
        # rows are integers starting from 0, columns are strings

        # getting a name 2 path dictionary
        self.name2path = self.create_name2path()

        # creating ImageFolder dataset compatibility attributes
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}
        # replacing spaces in filename with hyphens since they did it in the filenames
        # but not the csv for some unfathomable reason
        self.samples = [(self.name2path[file_name.replace(" ", "-").replace("---", "-") + ".jpg"],
                         self.class_to_idx[class_name])
                        for file_name, class_name in zip(self.OSData["image.name"], self.OSData["classification"])]

    def __getitem__(self, idx) -> [Any, torch.Tensor]:
        # following convention from PyTorch datasets
        sample_path, label = self.samples[idx]

        sample = self.loader(sample_path)  # Loading image
        if self.transform is not None:  # PyTorch implementation
            sample = self.transform(sample)

        return sample, torch.tensor(label)

    def __len__(self) -> int:
        return len(self.samples)

    def create_name2path(self) -> dict:
        # function to get a filename to relative
        # path dictionary
        try:
            dict_path = os.path.join(self.root, self.db_name, self.name2path_path)
            out_dict = pickle.load(open(dict_path, "rb"))
        except FileNotFoundError:
            print("creating name to path dictionary for Osteosarcoma")
            out_dict = {file_name: os.path.join(self.root, dir_path, file_name)
                        for dir_path, _, files in os.walk(self.root)
                        for file_name in files}

            with open(os.path.join(self.root, self.db_name, self.name2path_path), "wb") as handle:
                pickle.dump(out_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return out_dict
