from torchvision.datasets.folder import default_loader
import numpy as np
import pandas as pd
import os
import csv
from transforms import ProcessImages


class BCSSLoader:
    """loader to return tuples of pillow images and metadata, as well as a class to idx dictionary"""
    folder = "0_Public-data-Amgad2019_0.25MPP"
    MPP = 0.25
    image_dir = "rgbs_colorNormalized"
    masks_dir = "masks"
    g_truths = "meta/gtruth_codes.tsv"

    def __init__(self, root):
        self.classes_to_idx = {}
        with open(os.path.join(root, self.folder, self.g_truths)) as classes_tsv:
            classes = csv.DictReader(classes_tsv, delimeter="\t", quotechar='"')
            self.classes_to_idx = {row["label"]: row["GT_code"] for row in classes}

        self.file_names = []
        for root_dir, _, files in os.walk(os.path.join(root, self.folder, self.image_dir)):
            for file in files:
                self.file_names.append(file)

        self.samples = [(os.path.join(root, self.folder, self.image_dir, image_name),
                         os.path.join(root, self.folder, self.masks_dir, image_name))
                        for image_name in self.file_names]

    def __getitem__(self, item):
        image, mask = self.samples[item]
        return default_loader(image), default_loader(mask)

    def __len__(self):
        return len(self.samples)


def process_BCSS(root):
    image_dataset = BCSSLoader(root)
    for i in range(len(image_dataset)):
