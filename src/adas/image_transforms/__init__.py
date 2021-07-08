import PIL

import torchvision.transforms as transforms
from .custom_augmentations import YCbCr, HSV, ColorDistortion, \
    RGBJitter, Cutout, transformed_norm_weights


def get_transforms(
        dataset: str,
        degrees: int,
        gaussian_blur: bool,
        kernel_size: int,
        variance: float,
        vertical_flipping: float,
        horizontal_flipping: float,
        horizontal_shift: float,
        vertical_shift: float,
        color_kwargs: dict,
        cutout: bool,
        n_holes: int,
        length: int):
    transform_train = None
    transform_test = None

    color_processed_kwargs = {
        k: v for k, v in color_kwargs.items() if v is not None}

    if dataset == 'ADP-Release1':
        if 'augmentation' not in color_processed_kwargs.keys() or \
                'distortion' not in color_processed_kwargs.keys():
            raise ValueError(
                "'augmentation' and 'distortion' need to be specified for"
                " color augmentation in config.yaml::**kwargs")

        if color_processed_kwargs['augmentation'] == 'Color-Distortion':
            ColorAugmentation = ColorDistortion(color_processed_kwargs['distortion'])
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(p=horizontal_flipping),
                transforms.RandomVerticalFlip(p=vertical_flipping),
                transforms.RandomAffine(degrees=degrees, translate=(horizontal_shift, vertical_shift)),
                ColorAugmentation,
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.81233799, 0.64032477, 0.81902153],
                    std=[0.18129702, 0.25731668, 0.16800649])
            ])

            if gaussian_blur:  # insert gaussian blur before normalization
                transform_train.transforms.insert(-1,
                                                  transforms.GaussianBlur(kernel_size=kernel_size, sigma=variance))

            if cutout:
                transform_train.transforms.append(
                    Cutout(n_holes=n_holes, length=length))

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.81233799, 0.64032477, 0.81902153],
                    std=[0.18129702, 0.25731668, 0.16800649]),
            ])
        else:
            if color_processed_kwargs['augmentation'] == 'YCbCr':
                ColorAugmentation = YCbCr(distortion=color_processed_kwargs['distortion'])
            elif color_processed_kwargs['augmentation'] == 'RGB-Jitter':
                ColorAugmentation = RGBJitter(distortion=color_processed_kwargs['distortion'])
            elif color_processed_kwargs['augmentation'] == 'HSV':
                ColorAugmentation = HSV(distortion=color_processed_kwargs['distortion'])
            else:
                ColorAugmentation = None

            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(p=horizontal_flipping),
                transforms.RandomVerticalFlip(p=vertical_flipping),
                transforms.RandomAffine(degrees=degrees, translate=(horizontal_shift, vertical_shift)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.81233799, 0.64032477, 0.81902153],
                    std=[0.18129702, 0.25731668, 0.16800649])
            ])

            if ColorAugmentation:
                transform_train.transforms.insert(-2, ColorAugmentation)

            if gaussian_blur:  # TODO this should ideally be before normalization but after colour aug
                transform_train.transforms.insert(-3,
                                                  transforms.GaussianBlur(kernel_size=kernel_size, sigma=variance))

            if cutout:
                transform_train.transforms.append(
                    Cutout(n_holes=n_holes, length=length))

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.81233799, 0.64032477, 0.81902153],
                    std=[0.18129702, 0.25731668, 0.16800649]),
            ])
    elif dataset == 'MHIST':
        transform_train = transforms.Compose([
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[x / 255.0 for x in [188.14, 165.39, 192.69]], std=[
                    x / 255.0 for x in [50.30, 62.13, 43.42]]),
        ])

        if gaussian_blur:  # insert before norm
            transform_train.transforms.insert(-1,
                                              transforms.GaussianBlur(kernel_size=kernel_size, sigma=variance))

        if cutout:
            transform_train.transforms.append(
                Cutout(n_holes=n_holes, length=length))

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[x / 255.0 for x in [188.14, 165.39, 192.69]], std=[
                    x / 255.0 for x in [50.30, 62.13, 43.42]]),
        ])
    elif dataset == 'CIFAR100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=horizontal_flipping),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], std=[
                    x / 255.0 for x in [63.0, 62.1, 66.7]]),
        ])

        if gaussian_blur:  # insert before norm
            transform_train.transforms.insert(-1,
                                              transforms.GaussianBlur(kernel_size=kernel_size, sigma=variance))

        if cutout:
            transform_train.transforms.append(
                Cutout(n_holes=n_holes, length=length))

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], std=[
                    x / 255.0 for x in [63.0, 62.1, 66.7]]),
        ])

    elif dataset == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=horizontal_flipping),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        if gaussian_blur:
            transform_train.transforms.insert(-1,
                                              transforms.GaussianBlur(kernel_size=kernel_size, sigma=variance))

        if cutout:
            transform_train.transforms.append(
                Cutout(n_holes=n_holes, length=length))

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

    elif dataset == 'ImageNet':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=horizontal_flipping),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225]),
        ])

        if gaussian_blur:
            transform_train.transforms.insert(-1,
                                              transforms.GaussianBlur(kernel_size=kernel_size, sigma=variance))

        if cutout:
            transform_train.transforms.append(
                Cutout(n_holes=n_holes, length=length))

        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225]),
        ])

    elif dataset == 'TinyImageNet':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(p=horizontal_flipping),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225]),
        ])

        if gaussian_blur:
            transform_train.transforms.insert(-1,
                                              transforms.GaussianBlur(kernel_size=kernel_size, sigma=variance))

        if cutout:
            transform_train.transforms.append(
                Cutout(n_holes=n_holes, length=length))

        transform_test = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225]),
        ])
    elif dataset in ["AIDPATH_transformed",
                     "AJ-Lymph_transformed",
                     "BACH_transformed",
                     "CRC_transformed",
                     "GlaS_transformed",
                     "MHIST_transformed",
                     "OSDataset_transformed",
                     "PCam_transformed",
                     "BCSS_transformed"]:

        if 'augmentation' not in color_processed_kwargs.keys() or \
                'distortion' not in color_processed_kwargs.keys():
            raise ValueError(
                "'augmentation' and 'distortion' need to be specified for"
                " color augmentation in config.yaml::**kwargs")

        if color_processed_kwargs['augmentation'] == 'Color-Distortion':
            ColorAugmentation = ColorDistortion(color_processed_kwargs['distortion'])
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(p=horizontal_flipping),
                transforms.RandomVerticalFlip(p=vertical_flipping),
                transforms.RandomAffine(degrees=degrees, translate=(horizontal_shift, vertical_shift)),
                ColorAugmentation,
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=transformed_norm_weights[dataset]["mean"],
                    std=transformed_norm_weights[dataset]["std"])
            ])

            if gaussian_blur:  # insert gaussian blur before normalization
                transform_train.transforms.insert(-1,
                                                  transforms.GaussianBlur(kernel_size=kernel_size, sigma=variance))

            if cutout:
                transform_train.transforms.append(
                    Cutout(n_holes=n_holes, length=length))

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=transformed_norm_weights[dataset]["mean"],
                    std=transformed_norm_weights[dataset]["std"])
            ])
        else:
            if color_processed_kwargs['augmentation'] == 'YCbCr':
                ColorAugmentation = YCbCr(distortion=color_processed_kwargs['distortion'])
            elif color_processed_kwargs['augmentation'] == 'RGB-Jitter':
                ColorAugmentation = RGBJitter(distortion=color_processed_kwargs['distortion'])
            elif color_processed_kwargs['augmentation'] == 'HSV':
                ColorAugmentation = HSV(distortion=color_processed_kwargs['distortion'])
            else:
                ColorAugmentation = None

            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(p=horizontal_flipping),
                transforms.RandomVerticalFlip(p=vertical_flipping),
                transforms.RandomAffine(degrees=degrees, translate=(horizontal_shift, vertical_shift)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=transformed_norm_weights[dataset]["mean"],
                    std=transformed_norm_weights[dataset]["std"])
            ])

            if ColorAugmentation:
                transform_train.transforms.insert(-2, ColorAugmentation)

            if gaussian_blur:
                transform_train.transforms.insert(-3,
                                                  transforms.GaussianBlur(kernel_size=kernel_size, sigma=variance))

            if cutout:
                transform_train.transforms.append(
                    Cutout(n_holes=n_holes, length=length))

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=transformed_norm_weights[dataset]["mean"],
                    std=transformed_norm_weights[dataset]["std"])
            ])
    return transform_train, transform_test
