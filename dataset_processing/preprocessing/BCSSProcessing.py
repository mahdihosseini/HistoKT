from pathlib import Path

from torchvision.datasets.folder import default_loader
import numpy as np
import pandas as pd
import os
import csv
from transforms import ProcessImages
from skimage import io
import pickle
import random


class BCSSLoader:
    """loader to return tuples of pillow images and metadata, as well as a class to idx dictionary"""
    folder = "0_Public-data-Amgad2019_0.25MPP"
    MPP = 0.25
    image_dir = "rgbs_colorNormalized"
    masks_dir = "masks"
    g_truths = "meta/gtruth_codes.tsv"

    def __init__(self, root):
        self.root = root
        self.class_to_value = {}
        with open(os.path.join(root, self.folder, self.g_truths)) as classes_tsv:
            classes = csv.DictReader(classes_tsv, dialect='excel-tab')
            self.class_to_value = {row["label"]: int(row["GT_code"]) for row in classes}

        self.file_names = []
        for root_dir, _, files in os.walk(os.path.join(root, self.folder, self.image_dir)):
            for file in files:
                self.file_names.append(file)

        self.samples = [(os.path.join(root, self.folder, self.image_dir, image_name),
                         os.path.join(root, self.folder, self.masks_dir, image_name))
                        for image_name in self.file_names]

    def __getitem__(self, item):
        return self.samples[item]

    def __len__(self):
        return len(self.samples)


def process_BCSS(root,
                 class_list,
                 target_folder,
                 scale=0.25,
                 target_dim=(272, 272),
                 percent_overlap=0.5,
                 intensity_threshold=0.8,
                 occurrence_threshold=0.975,
                 show_imgs=False,
                 low_contrast=True,
                 lower_percentile=5,
                 upper_percentile=99,
                 label_threshold=0.5,
                 split_dict={"train": 100}):
    new_samples = []
    image_dataset = BCSSLoader(root)

    class_to_idx = {cls: i for i, cls in enumerate(class_list)}
    for i in range(len(image_dataset)):
        image_path, mask_path = image_dataset[i]
        image = np.array(default_loader(image_path))
        mask = np.array(default_loader(mask_path))

        rescaled_img = ProcessImages.scale_img(image, scale)
        rescaled_mask = ProcessImages.scale_img(mask, scale)

        if rescaled_img.shape[0] > target_dim[0] and \
                rescaled_img.shape[1] > target_dim[1]:
            # handle larger image cropping here
            out_images = ProcessImages.crop(rescaled_img, target_dim, percent_overlap)
            out_masks = ProcessImages.crop(rescaled_mask, target_dim, percent_overlap)

        else:

            out_img = ProcessImages.reflection_wrap(rescaled_img,
                                                    dim=target_dim)
            out_mask = ProcessImages.reflection_wrap(rescaled_mask,
                                                     dim=target_dim)

            out_images = ProcessImages.crop(out_img, target_dim, percent_overlap)
            out_masks = ProcessImages.crop(out_mask, target_dim, percent_overlap)

        # remove images containing too much background
        filtered_images = ProcessImages.remove_background(out_images,
                                                          intensity_threshold,
                                                          occurrence_threshold,
                                                          low_contrast,
                                                          lower_percentile,
                                                          upper_percentile)

        # save out_img with the same file structure as the dataset in
        # different folder

        image_path = Path(image_path)
        image_path = image_path.relative_to(root)
        image_path = Path(os.path.join(target_folder, image_path))

        for j, (image2save, mask2save) in enumerate(zip(filtered_images, out_masks)):
            if type(image2save) is tuple:
                print(image_path.stem + f"-{j}" + ".png" + " is a background image because", image2save[1])
                if show_imgs:
                    io.imshow(image2save[0])
                    io.show()
                continue

            labels = mask_to_vec(mask2save,
                                 class_to_value=image_dataset.class_to_value,
                                 class_list=class_list,
                                 norm=True)

            if sum(labels) < label_threshold:
                print(image_path.stem + f"-{j}" + ".png" + " is a background image because not enough label")
                if show_imgs:
                    io.imshow(image2save)
                    io.show()
                continue

            filename = image_path.stem + f"-{j}" + ".png"
            save_path = str(image_path).replace(image_path.name, filename)
            save_path = Path(save_path)
            new_samples.append((str(save_path.relative_to(target_folder)), labels))
            save_path.parent.mkdir(parents=True, exist_ok=True)
            io.imsave(str(save_path), image2save)
    split_and_save_samples(new_samples, target_folder, split_dict=split_dict)
    save_class_to_idx({cls: i for i, cls in enumerate(class_list)}, target_folder)
    return


def mask_to_vec(mask, class_to_value, class_list, norm=True, RGB_mask=True):
    """takes in a numpy array for the mask image, and returns a normalized class distribution"""
    if RGB_mask:
        mask = mask[:, :, 0]
    unique, counts = np.unique(mask, return_counts=True)

    value_to_num = {value: num for value, num in zip(unique, counts)}

    output = [value_to_num[class_to_value[cls]] if class_to_value[cls] in value_to_num else 0.0 for cls in class_list ]

    if norm:
        output /= sum(counts)

    return output


def split_and_save_samples(samples, target_folder, split_dict) -> None:
    """
    splits samples into folds specified by self.split_dict
    saves splits in the target folder as {split_name}.pickle
    Args:
        samples: list
            list of (path, label) to be saved

    Returns:
        None
    """
    random.shuffle(samples)
    dataset_length = len(samples)

    cumulative_percentage = 0
    for split_name, percentage in split_dict.items():
        start_index = int(cumulative_percentage / 100 * dataset_length)
        end_index = int((cumulative_percentage + percentage) / 100 * dataset_length)
        end_index = end_index if end_index < dataset_length else dataset_length
        split_samples = samples[start_index:end_index]

        # saving split samples:
        with open(os.path.join(target_folder, split_name + ".pickle"), "wb") as f:
            pickle.dump(split_samples, f)
        cumulative_percentage += percentage


def save_class_to_idx(class_to_index, target_folder):
    with open(os.path.join(target_folder, "class_to_idx.pickle"), "wb") as f:
        pickle.dump(class_to_index, f)


if __name__ == "__main__":
    root = "C:\\Users\\ryanr\\Desktop\\Summer_Research\\HistoKT\\.adas-data"
    target_folder = "BCSS_transformed"
    target_folder = os.path.join(root, target_folder)

    class_list = ['tumor',
                  'stroma',
                  'lymphocytic_infiltrate',
                  'necrosis_or_debris',
                  'glandular_secretions',
                  'blood',
                  'metaplasia_NOS',
                  'fat',
                  'plasma_cells',
                  'blood_vessel']
    process_BCSS(root,
                 class_list,
                 target_folder,
                 scale=0.25,
                 target_dim=(272, 272),
                 percent_overlap=0.75,
                 intensity_threshold=0.8,
                 occurrence_threshold=0.975,
                 show_imgs=False,
                 low_contrast=True,
                 lower_percentile=5,
                 upper_percentile=99,
                 label_threshold=0.2,
                 split_dict={"train": 80, "valid": 10, "test": 10})
