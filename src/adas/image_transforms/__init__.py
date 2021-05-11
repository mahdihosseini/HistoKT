import PIL

import torchvision.transforms as transforms
from image_transforms.custom_augmentations import YCbCr, HSV, ColorDistortion,\
                                                  Normalize, RGBJitter, Cutout

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
            color_kwargs: dict(),
            cutout: bool,
            n_holes: int,
            length: int):
    train_transform = None
    test_transform = None
    
    color_processed_kwargs = {
        k: v for k, v in color_kwargs.items() if v is not None}
    
    if 'augmentation' not in color_processed_kwargs.keys() or \
            'distortion' not in color_processed_kwargs.keys():
        raise ValueError(
            "'augmentation' and 'distortion' need to be specified for"
            " color augmentation in config.yaml::**kwargs")
            
    if dataset == 'ADP-Release1':
        if color_processed_kwargs['augmentation'] == 'Color-Distortion':
            ColorAugmentation = ColorDistortion(color_processed_kwargs['distortion'])     
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p = horizontal_flipping),
                transforms.RandomVerticalFlip(p = vertical_flipping),
                transforms.RandomAffine(degrees = degrees, translate = (horizontal_shift, vertical_shift)),
                ColorAugmentation,
                transforms.ToTensor(),
                transforms.Normalize(
                    mean = [0.75740063, 0.75740063, 0.75740063],
                    std = [0.21195677, 0.21195677, 0.21195677])
                ])

            if gaussian_blur: # insert gaussian blur before normalization
                train_transform.transforms.insert(-1, 
                    transforms.GaussianBlur(kernel_size=kernel_size, sigma=variance))

            if cutout:
                train_transform.transforms.append(
                    Cutout(n_holes=n_holes, length=length))
    
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean = [0.75740063, 0.75740063, 0.75740063],
                    std = [0.21195677, 0.21195677, 0.21195677]),
                ])
        else:
            if color_processed_kwargs['augmentation'] == 'YCbCr':
                ColorAugmentation = YCbCr(distortion = color_processed_kwargs['distortion'])
            elif color_processed_kwargs['augmentation'] == 'RGB-Jitter':
                ColorAugmentation = RGBJitter(distortion = color_processed_kwargs['distortion'])
            elif color_processed_kwargs['augmentation'] == 'HSV':
                ColorAugmentation = HSV(distortion = color_processed_kwargs['distortion'])
            else:
                ColorAugmentation = None

            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p = horizontal_flipping),
                transforms.RandomVerticalFlip(p = vertical_flipping),
                transforms.RandomAffine(degrees = degrees, translate = (horizontal_shift, vertical_shift)),
                transforms.ToTensor(),
                ])
            
            if ColorAugmentation:
                train_transform.transforms.insert(-1, ColorAugmentation)
            else:
                train_transform.transforms.insert(-1, Normalize()) # since the colour aug normalizes
            
            if gaussian_blur: # TODO this should ideally be before normalization but after colour aug
                train_transform.transforms.append(
                    transforms.GaussianBlur(kernel_size=kernel_size, sigma=variance))

            if cutout:
                train_transform.transforms.append(
                    Cutout(n_holes=n_holes, length=length))
            
            test_transform = transforms.Compose([
                Normalize(),
                transforms.ToTensor(),
                ])
     
    elif dataset == 'CIFAR100':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p = horizontal_flipping),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], std=[
                    x / 255.0 for x in [63.0, 62.1, 66.7]]),
        ])

        if gaussian_blur: #insert before norm
                train_transform.transforms.insert(-1,
                    transforms.GaussianBlur(kernel_size=kernel_size, sigma=variance))

        if cutout:
            train_transform.transforms.append(
                Cutout(n_holes=n_holes, length=length))

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], std=[
                    x / 255.0 for x in [63.0, 62.1, 66.7]]),
        ])
        
    elif dataset == 'CIFAR10':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p = horizontal_flipping),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
            
        if gaussian_blur:
            train_transform.transforms.insert(-1, 
                transforms.GaussianBlur(kernel_size=kernel_size, sigma=variance))

        if cutout:
            train_transform.transforms.append(
                Cutout(n_holes=n_holes, length=length))

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        
    elif dataset == 'ImageNet':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p = horizontal_flipping),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])

        if gaussian_blur:
            train_transform.transforms.insert(-1, 
                transforms.GaussianBlur(kernel_size=kernel_size, sigma=variance))

        if cutout:
            train_transform.transforms.append(
                Cutout(n_holes=n_holes, length=length))

        test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])
        
    elif dataset == 'TinyImageNet':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(p = horizontal_flipping),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])

        if gaussian_blur:
            train_transform.transforms.insert(-1, 
                transforms.GaussianBlur(kernel_size=kernel_size, sigma=variance))

        if cutout:
            train_transform.transforms.append(
                Cutout(n_holes=n_holes, length=length))

        test_transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])            
    return (train_transform, test_transform) 
