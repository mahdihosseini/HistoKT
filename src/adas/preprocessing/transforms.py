from skimage.transform import resize, rescale
from skimage import io
from skimage.exposure import is_low_contrast

import matplotlib.pyplot as plt
import numpy as np
from torch.utils import data
from torch.utils.data import dataset
from torchvision.datasets.folder import default_loader
import torch
import os
from pathlib import Path
import random
import pickle


def print_imgs(dataset):
    fig, axes = plt.subplots(nrows=2, ncols=2)
    ax = axes.ravel()
    for i in range(4):
        image, _ = dataset[i]
        image = np.array(image)
        ax[i].imshow(image)
        ax[i].set_title("Original Images")

    plt.tight_layout()
    plt.show()


class ProcessImages:

    def __init__(self, dataset, target_folder, split_dict={"train": 100},) -> None:
        """
        A class to process a dataset and save the resultant images 
        to a given folder

        The dataset MUST HAVE ATTRIBUTES OF
            root: the root folder of the dataset
            class_to_idx: a dictionary converting class
                to index
            samples: a list of tuples (image_path, class)
        """

        self.dataset = dataset
        self.dataset.transforms = None
        self.target_folder = target_folder
        self.split_dict = split_dict
        # checking if dataset has all necessary attributes
        try:
            if self.dataset.root and \
                    self.dataset.class_to_idx and \
                    self.dataset.samples:
                pass
        except AttributeError as err:
            print("The dataset MUST HAVE ATTRIBUTES OF")
            print("root: the root folder of the dataset")
            print("class_to_idx: a dictionary converting class")
            print("    to index")
            print("samples: a list of tuples (image_path, class)")
            raise err

    def process(self,
                scale=1,
                target_dim=(272, 272),
                percent_overlap=0.5,
                intensity_threshold=0.8,
                occurrence_threshold=0.975,
                show_imgs=False,
                low_contrast=True,
                lower_percentile=5,
                upper_percentile=99) -> None:
        # processes images and saves
        # them to the target folder

        new_samples = []

        for i in range(len(self.dataset)):
            image_path, label = self.dataset.samples[i]
            image = default_loader(image_path)
            image = np.array(image)
            rescaled_img = self.scale_img(image, scale)

            if rescaled_img.shape[0] > target_dim[0] and \
                    rescaled_img.shape[1] > target_dim[1]:
                # handle larger image cropping here
                out_images = self.crop(rescaled_img, target_dim, percent_overlap)

            else:

                out_img = self.reflection_wrap(rescaled_img,
                                               dim=target_dim)

                out_images = self.crop(out_img, target_dim, percent_overlap)

            # remove images containing too much background
            filtered_images = self.remove_background(out_images,
                                                     intensity_threshold,
                                                     occurrence_threshold,
                                                     low_contrast,
                                                     lower_percentile,
                                                     upper_percentile)

            # save out_img with the same file structure as the dataset in 
            # different folder

            image_path = Path(image_path)
            image_path = image_path.relative_to(self.dataset.root)
            image_path = Path(os.path.join(self.target_folder, image_path))

            for j, image2save in enumerate(filtered_images):
                if type(image2save) is tuple:
                    print(image_path.stem + f"-{j}" + ".png" + " is a background image because", image2save[1])
                    if show_imgs:
                        io.imshow(image2save[0])
                        io.show()
                    continue
                filename = image_path.stem + f"-{j}" + ".png"
                save_path = str(image_path).replace(image_path.name, filename)
                save_path = Path(save_path)
                new_samples.append((str(save_path.relative_to(self.target_folder)), label))
                save_path.parent.mkdir(parents=True, exist_ok=True)
                io.imsave(str(save_path), image2save)
        self.split_and_save_samples(new_samples)
        self.save_class_to_idx()
        return

    @staticmethod
    def scale_img(image, scale=1) -> np.array:
        # scales up or down an image
        return rescale(image, scale, clip=True, multichannel=True, preserve_range=True)

    @staticmethod
    def reflection_wrap(image, dim=(272, 272)) -> np.array:
        image = np.array(image)

        resized_img = np.zeros(dim + (3,))
        height, width, channel = image.shape
        height_factor = dim[0] // (height - 1)
        width_factor = dim[1] // (width - 1)

        height_resized_img = np.zeros((dim[0], width, 3))
        height_resized_img[:height, :width, :] = image

        # multiple flips

        for i in range(height_factor):
            # wraps in the height direction
            direction = 1 if i % 2 else -1
            # start with -1, then 1, alternating
            height_start = height + i * (height - 1)
            # subtract one since we want to keep the center
            # pixel and not duplicate it, so previous flip and paste
            # is one pixel less
            height_end = height_start + height - 1
            # subtract one since we want to keep the center
            # pixel and not duplicate it
            height_end = height_end if height_end < dim[0] else dim[0]
            # logic for flipping full image or part of image
            cropped_length = height_end - height_start + 1
            # accounting for python array indexing
            height_resized_img[height_start:height_end, :, :] = \
                image[:(cropped_length * direction + (i % 2 - 1)):direction, :, :][1:, :, :]
            # logic of i%2 accounts for the difference between positive indices
            # and negative indices, e.g. -6 = 5

        resized_img[:, :width, :] = height_resized_img

        for i in range(width_factor):
            # wraps in the width direction
            direction = 1 if i % 2 else -1
            # start with -1, then 1, alternating
            width_start = width + i * (width - 1)
            # subtract one since we want to keep the center
            # pixel and not duplicate it, so previous flip and paste
            # is one pixel less
            width_end = width_start + width - 1
            # subtract one since we want to keep the center
            # pixel and not duplicate it
            width_end = width_end if width_end < dim[1] else dim[1]
            # logic for flipping full image or part of image
            cropped_length = width_end - width_start + 1
            # accounting for python array indexing
            resized_img[:, width_start:width_end, :] = \
                height_resized_img[:, :(cropped_length * direction + (i % 2 - 1)):direction, :][:, 1:, :]
            # logic of i%2 accounts for the difference between positive indices
            # and negative indices, e.g. -6 = 5

        return resized_img.astype(np.uint8)

    @staticmethod
    def crop(image, dim=(272, 272), percent_overlap=.25) -> [np.array]:
        image = np.array(image)

        image_as_tensor = torch.as_tensor(image)

        stride = (int(dim[0] * (1 - percent_overlap)), int(dim[1] * (1 - percent_overlap)))  # patch stride
        patches = image_as_tensor.unfold(0, dim[0], stride[0]).unfold(1, dim[1], stride[1])

        patches = patches.reshape(-1, image.shape[-1], *dim).permute(0, 2, 3, 1).numpy().astype(np.uint8)

        return [patches[i] for i in range(patches.shape[0])]

    @staticmethod
    def remove_background(img_list,
                          intensity_threshold=0.5,
                          occurrence_threshold=0.975,
                          low_contrast=False,
                          lower_percentile=5,
                          upper_percentile=99) -> [np.array]:
        out_list = []
        rgb_threshold = intensity_threshold * 255 * 3
        for img in img_list:
            image_size = img.shape[0] * img.shape[1]
            squeezed_img = np.sum(img, axis=-1)
            masked_img = np.where(squeezed_img > rgb_threshold, 1, 0)
            if low_contrast:
                if is_low_contrast(img, lower_percentile, upper_percentile):
                    out_list.append((img, "low_c"))
                    continue
            else:
                if np.sum(masked_img) >= occurrence_threshold * image_size:
                    out_list.append((img, "bright"))
                    continue
            out_list.append(img)

        return out_list

    def split_and_save_samples(self, samples) -> None:
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
        for split_name, percentage in self.split_dict.items():
            start_index = int(cumulative_percentage * dataset_length)
            end_index = int((cumulative_percentage + percentage) * dataset_length)
            end_index = end_index if end_index < dataset_length else dataset_length
            split_samples = samples[start_index:end_index]

            # saving split samples:
            with open(os.path.join(self.target_folder, split_name + ".pickle"), "wb") as f:
                pickle.dump(split_samples, f)
            cumulative_percentage += percentage

    def save_class_to_idx(self):
        with open(os.path.join(self.target_folder, "class_to_idx.pickle"), "wb") as f:
            pickle.dump(self.dataset.class_to_idx, f)


class ProcessDatasets:

    def __init__(self, dataset_list):
        """

        Args:
            dataset_list: [(torch.utils.data.Dataset, "target_folder")]
                list of constructed datasets
        """
        self.dataset_list = dataset_list

    def process(self):
        for dataset, target_folder in self.dataset_list:
            image_processor = ProcessImages(dataset, target_folder=target_folder)

            image_processor.process()

