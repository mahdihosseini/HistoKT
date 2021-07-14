import math
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from scipy.stats import circmean
from scipy.stats import circstd
from skimage.color import hsv2rgb, rgb2hsv


class YCbCr:
    def __init__(self, distortion):
        self.distortion = distortion

    def __call__(self, image):
        image_array = np.asarray(image)
        image_dtype = image_array.dtype

        A = np.array([[65.481, 128.553, 24.966], [-37.797, -74.203, 112.0], [112.0, -93.786, -18.214]])
        A = A / 255.
        b = np.array([16, 128, 128])

        x = image_array.reshape((image_array.shape[0] * image_array.shape[1], image_array.shape[2]))
        image_ycbcr = x @ A.T + b
        image_ycbcr = image_ycbcr.reshape((image_array.shape[0], image_array.shape[1], image_array.shape[2]))

        Y = image_ycbcr[:, :, 0]
        Cb = image_ycbcr[:, :, 1]
        Cr = image_ycbcr[:, :, 2]

        mean_Cb = np.mean(Cb, axis=(0, 1))
        mean_Cr = np.mean(Cr, axis=(0, 1))
        std_Cb = np.std(Cb, axis=(0, 1))
        std_Cr = np.std(Cr, axis=(0, 1))

        Cb_centered = Cb - mean_Cb
        Cr_centered = Cr - mean_Cr

        Cb_centered_augmented = Cb_centered + np.random.normal(loc=0, scale=(self.distortion * std_Cb))
        Cr_centered_augmented = Cr_centered + np.random.normal(loc=0, scale=(self.distortion * std_Cr))

        Cb_augmented = Cb_centered_augmented + mean_Cb
        Cr_augmented = Cr_centered_augmented + mean_Cr

        image_perturbed_ycbcr = np.empty(image_array.shape)
        image_perturbed_ycbcr[:, :, 0] = Y
        image_perturbed_ycbcr[:, :, 1] = Cb_augmented
        image_perturbed_ycbcr[:, :, 2] = Cr_augmented

        inv_A = np.linalg.inv(A)
        image_perturbed = (image_perturbed_ycbcr - b) @ inv_A.T
        image_perturbed = np.rint(np.clip(image_perturbed, 0, 255)).astype('uint8')
        # image_perturbed = (image_perturbed - 193.09203) / (56.450138 + 1e-7)
        image = image_perturbed.astype(image_dtype)
        return Image.fromarray(image)


class HSV:
    def __init__(self, distortion):
        self.distortion = distortion

    def __call__(self, image):
        image_array = np.asarray(image)
        image_dtype = image_array.dtype

        HSV_image = rgb2hsv(image_array)
        H = HSV_image[:, :, 0]
        H_rad = H * [2 * math.pi] - math.pi
        S = HSV_image[:, :, 1]
        V = HSV_image[:, :, 2]
        mean_H_rad = circmean(H_rad)
        std_H_rad = circstd(H_rad)
        mean_S = np.mean(S, axis=(0, 1))
        std_S = np.std(S, axis=(0, 1))

        H_rad_centered = np.angle(np.exp(1j * (H_rad - mean_H_rad)))
        H_rad_centered_augmented = H_rad_centered + np.random.normal(loc=0, scale=(self.distortion * std_H_rad))
        H_rad_augmented = np.angle(np.exp(1j * (H_rad_centered_augmented + mean_H_rad)))
        H_augmented = np.divide(H_rad_augmented + math.pi, 2 * math.pi)

        S_centered = S - mean_S
        S_centered_augmented = S_centered + np.random.normal(loc=0, scale=(self.distortion * std_S))
        S_augmented = S_centered_augmented + mean_S

        image_perturbed_HSV = np.empty(image_array.shape)
        image_perturbed_HSV[:, :, 0] = H_augmented
        image_perturbed_HSV[:, :, 1] = S_augmented
        image_perturbed_HSV[:, :, 2] = V

        image_rgb = hsv2rgb(image_perturbed_HSV)
        image_rgb = image_rgb * 255.0
        image_perturbed = np.rint(np.clip(image_rgb, 0, 255)).astype('uint8')
        # image_perturbed = (image_perturbed - 193.09203) / (56.450138 + 1e-7) 
        image = image_perturbed.astype(image_dtype)
        return Image.fromarray(image)


class ColorDistortion:
    def __init__(self, distortion):
        self.distortion = distortion

    def __call__(self, image):
        color_jitter = transforms.ColorJitter(0.8 * self.distortion, 0.8 * self.distortion,
                                              0.8 * self.distortion, 0.2 * self.distortion)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=1.0)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([
            rnd_color_jitter,
            # rnd_gray
        ])
        transformed_image = color_distort(image)
        return transformed_image


class RGBJitter:
    def __init__(self, distortion):
        self.distortion = distortion

    def __call__(self, image):
        image_array = np.asarray(image)
        image_dtype = image_array.dtype
        image_shifted = self.pca_augmentation(image_array)
        image = image_shifted.astype(image_dtype)
        return Image.fromarray(image)

    def pca_augmentation(self, image):
        # Normalization
        img_array = image / 255.0
        mean = np.mean(img_array, axis=(0, 1))
        img_norm = (img_array - mean)
        # Covariance matrix
        img_rs = img_norm.reshape(img_norm.shape[0] * img_norm.shape[1], img_norm.shape[2])
        cov_matrix = np.cov(img_rs, rowvar=False)
        # Principal Components
        eig_values, eig_vectors = np.linalg.eig(cov_matrix)
        # Sorting the eigen_vectors in the order of their eigen_values (highest to lowest)
        indices = np.flipud(eig_values.argsort())  # indices of the sorted eig_values in decreasing order
        eig_values = sorted(eig_values, reverse=True)  # eig_values sorted in descending order
        eig_vectors = eig_vectors[:, indices]
        alphas = np.random.normal(0, self.distortion, 3)
        delta = np.dot(eig_vectors, (alphas * eig_values))

        image_distorted = img_norm + delta
        image_distorted = (image_distorted + mean) * 255.0
        image_distorted = np.rint(np.clip(image_distorted, 0, 255)).astype('uint8')
        # image_distorted = (image_distorted - 193.09203) / (56.450138 + 1e-7)
        return image_distorted


# class Normalize:  
#     def __call__(self, image): 
#         image_array = np.asarray(image)
#         image_dtype = image_array.dtype
#         image_perturbed = (image_array - 193.09203) / (56.450138 + 1e-7) 
#         image = image_perturbed.astype(image_dtype) 
#         return Image.fromarray(image)


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    @author: uoguelph-mlrg
      (https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py)
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        if n_holes < 0 or length < 0:
            raise ValueError("Must set n_holes or length args for cutout")
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of
            it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


transformed_norm_weights = {
    'AIDPATH_transformed': {'mean': [0.6032, 0.3963, 0.5897], 'std': [0.1956, 0.2365, 0.1906]},
    'AJ-Lymph_transformed': {'mean': [0.4598, 0.3748, 0.4612], 'std': [0.1406, 0.1464, 0.1176]},
    'BACH_transformed': {'mean': [0.6880, 0.5881, 0.8209], 'std': [0.1632, 0.1841, 0.1175]},
    'CRC_transformed': {'mean': [0.6976, 0.5340, 0.6687], 'std': [0.2272, 0.2697, 0.2247]},
    'GlaS_transformed': {'mean': [0.7790, 0.5002, 0.7765], 'std': [0.1638, 0.2418, 0.1281]},
    'MHIST_transformed': {'mean': [0.7361, 0.6469, 0.7735], 'std': [0.1812, 0.2303, 0.1530]},
    'OSDataset_transformed': {'mean': [0.8414, 0.6492, 0.7377], 'std': [0.1379, 0.2508, 0.1979]},
    'PCam_transformed': {'mean': [0.6970, 0.5330, 0.6878], 'std': [0.2168, 0.2603, 0.1933]},
    'ADP': {'mean': [0.81233799, 0.64032477, 0.81902153], 'std': [0.18129702, 0.25731668, 0.16800649]}}

