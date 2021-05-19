from skimage.transform import resize, rescale

import matplotlib.pyplot as plt
import numpy as np
from torch.utils import data
from torch.utils.data import dataset
from torchvision.datasets.folder import default_loader


def scale_up(image, scale=1):
    # 
    return rescale(image, scale, clip=True, multichannel=True)

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
    
    def __init__(self, dataset, target_folder) -> None:
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
        try:
            self.dataset.transforms = None
        except:
            raise
        self.target_folder = target_folder

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

    def process(self, target_dim = (272, 272)):
        # processes images and saves
        # them to the target folder
        
        for i in range(len(self.dataset)):
            image_path, _ = dataset[i]
            image = default_loader(image_path)
            rescaled_img = self.scale_img(image)

            if rescaled_img.shape[0] > target_dim[0] and \
                rescaled_img.shape[1] > target_dim[1]:
                pass
                # handle larger image cropping here

            else:
                out_img = self.reflection_wrap(rescaled_img,
                                dim=target_dim)
            
            # save out_img with the same file structure as the dataset in 
            # different folder

    @staticmethod
    def scale_img(image, scale=1) -> np.array:
        # scales up or down an image
        return rescale(image, scale, clip=True, multichannel=True)
    
    @staticmethod
    def reflection_wrap(image, dim = (272, 272)) -> np.array:
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
            # logic for fipping full image or part of image
            cropped_length = height_end - height_start + 1
            # accounting for python array indexing
            height_resized_img[height_start:height_end, :, :] = \
                image[:(cropped_length * direction + (i % 2 - 1)):direction, :, :][1:, :, :]
            # logic of i%2 accounts for the difference between positive indicies
            # and negative indicies, e.g. -6 = 5 

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
            # logic for fipping full image or part of image
            cropped_length = width_end - width_start + 1
            # accounting for python array indexing
            resized_img[:, width_start:width_end, :] = \
                height_resized_img[:, :(cropped_length * direction + (i % 2 - 1)):direction, :][:, 1:, :]
            # logic of i%2 accounts for the difference between positive indicies
            # and negative indicies, e.g. -6 = 5 

        return resized_img