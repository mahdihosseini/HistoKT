import os
import sys
import openslide
import numpy as np
import scipy.misc
import cv2
import glob
import re
import xml.etree.ElementTree as ET
from tqdm import tqdm
from skimage import io


if __name__ == "__main__":
    dataset_root = "/home/zhujiada/scratch/PAIP/PAIP_example_data"
    svs_load_dir = os.path.join(dataset_root, "svs_folder/")
    svs_fns = sorted(glob.glob(svs_load_dir + "*.svs") + glob.glob(svs_load_dir + "*.SVS"))

    tif_save_dir = os.path.join(dataset_root, "tif_folder/")
    os.makedirs(tif_save_dir, exist_ok=True)

    wsi_uid_pattern = "[a-zA-Z]*_PNI2021chall_train_[0-9]{4}"
    wsi_regex = re.compile(wsi_uid_pattern)

    for svs_fn in tqdm(svs_fns):
        wsi_uid = wsi_regex.findall(svs_fn)[0]

        slide = openslide.OpenSlide(svs_load_dir + wsi_uid + ".svs")
        img = np.array(slide.read_region((0, 0), 0, slide.dimensions))
        scipy.misc.imsave(tif_save_dir + wsi_uid + ".svs", img)
        slide.close()