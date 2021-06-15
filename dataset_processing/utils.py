import torch
from collections import defaultdict
import pickle
import os
import tarfile
import random
import tempfile


def get_mean_and_std(dataset: torch.utils.data.dataset,
                     batch_size: int = 32,
                     num_workers: int = 0,
                     shuffle: bool = False):
    """
    Slower direct implementation of mean and std calculations

    TODO Could use Welford’s method for computing variance, but not
    used here
    Args:
        dataset: torch.utils.data.dataset ->
            dataset of images, should return
            (sample, labels)
            images are of shape (n, c, h, w)
        batch_size: int ->
        num_workers: int ->
        shuffle: bool ->

    """
    mean = torch.zeros(size=(3,))
    std = torch.zeros(size=(3,))
    n_samples = 0.0
    n_pixels = dataset[0][0].size(-1) * dataset[0][0].size(-2)

    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         shuffle=shuffle)

    # pass 1 to calculate the mean:
    for images, _ in loader:
        # images are of shape (n, c, h, w)
        # hopefully (they need to be passed through ToTensor)
        # images are also normalized between 0 and 1
        mean += images.mean((-2, -1)).sum(0)
        n_samples += images.size(0)

    mean /= n_samples

    for images, _ in loader:
        std += ((images - mean.view(1, 3, 1, 1)) ** 2).sum(dim=(0, 2, 3)) / n_pixels

    std /= n_samples - 1
    std = std ** 0.5

    return mean, std


def archive_subdataset(dataset: torch.utils.data.dataset,
                       tar_name: str,
                       split_by_classes: bool = True,
                       num_per_class: int = 100):
    # creating dictionary with class index as a key, and [filenames, ] as the value

    class_to_files = defaultdict(list)
    samples = pickle.load(open(os.path.join(dataset.root, dataset.split + ".pickle"), "rb"))
    random.seed(42)
    random.shuffle(samples)

    for filename, label in samples:
        class_to_files[label].append(filename)

    try:
        new_samples = [(file_list[i], label)
                       for label, file_list in class_to_files.items()
                       for i in range(num_per_class)]
    except IndexError as err:
        print(f"not {num_per_class} enough of a class: {[(key, len(value)) for key, value in class_to_files.items()]}")
        raise err
    # getting list of files to tar
    files_to_tar = [path for path, label in new_samples]

    # tarring files
    with tarfile.open(tar_name+".tar", "a") as tar:
        archive_name = os.path.basename(dataset.root)
        for fn in files_to_tar:
            tar.add(os.path.join(dataset.root, fn), arcname=os.path.join(archive_name, fn))

        # adding pickles
        with open(os.path.join(tempfile.gettempdir(), f"{dataset.split}.pickle"), "wb") as file:
            pickle.dump(new_samples, file)
        tar.add(os.path.join(tempfile.gettempdir(), f"{dataset.split}.pickle"), arcname=os.path.join(archive_name, f"{dataset.split}.pickle"))

        # adding class_to_idx pickle
        tar.add(os.path.join(dataset.root, "class_to_idx.pickle"),
                arcname=os.path.join(archive_name, "class_to_idx.pickle"))

