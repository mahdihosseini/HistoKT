import os
import sys
import cv2
import glob
import numpy as np
import pandas as pd
import openslide
import re
import xml.etree.ElementTree as ET
from tqdm import tqdm
from skimage import io
from skimage.transform import rescale
from skimage.exposure import is_low_contrast

def crop_helper(wsi_uid, crop_size, overlap, label, label_name, pts, img_name, info_dict):
    temp = np.array(pts)
    x = np.min(temp[:, 0])
    y = np.min(temp[:, 1])
    while x < np.max(temp[:, 0]):
        while y < np.max(temp[:, 1]):
            info_dict["img_name"].append(str(img_name))
            info_dict["label_Id"].append(label)
            info_dict["label_Name"].append(label_name)
            info_dict["top_left_pixel"].append((x, y))
            info_dict["svs_name"].append(str(wsi_uid))

            img_name += 1
            
            y = round(y + crop_size * overlap)

        y = np.min(temp[:, 1])
        x = round(x + crop_size * overlap)
    return img_name, info_dict


def crop(xml_fn, slide, level, crop_save_dir, wsi_uid, downscale_factor, crop_size, overlap, selection):
    """
    <XML Tree>
    Annotations (root)
    > Annotation
      > Regions
        > Region
          > Vertices
            > Vertex
              > X, Y
    <Label>
    nerve_without_tumor (contour): 1
    perineural_invasion_junction (line): 2
    nerve_without_tumor (bounding box): 11
    tumor_without_nerve (bounding box): 13
    nontumor_without_nerve (bounding box): 14
    """

    etree = ET.parse(xml_fn)

    # Height and Width Ratio
    src_w, src_h  = slide.level_dimensions[0]
    dest_w, dest_h = slide.level_dimensions[level]
    w_ratio = src_w / dest_w
    h_ratio = src_h / dest_h

    info_dict = {
        "img_name": [],
        "label_Id": [],
        "label_Name": [],
        "top_left_pixel": [],
        "svs_name": []
    }
    img_name = 0

    annotations = etree.getroot()
    for annotation in annotations:
        label = int(annotation.get("Id"))
        label_name = str(annotation.get("Name"))

        cntr_pts = list()
        bbox_pts = list()

        regions = annotation.findall("Regions")[0]
        for region in regions.findall("Region"):
            pts = list()

            vertices = region.findall("Vertices")[0]
            for vertex in vertices.findall("Vertex"):
                x = round(float(vertex.get("X")))
                y = round(float(vertex.get("Y")))

                # Match target level coordinates
                x = np.clip(round(x / w_ratio), 0, dest_w)
                y = np.clip(round(y / h_ratio), 0, dest_h)

                pts.append((x, y))

            if len(pts) == 4:
                bbox_pts += [pts]
            else:
                cntr_pts += [pts]

        # Bounding box
        for pts in bbox_pts:
            img_name, info_dict = crop_helper(wsi_uid, crop_size, overlap, label, label_name, pts, img_name, info_dict)

        for pts in cntr_pts:
            # Curved line
            if label == 2:
                img_name, info_dict = crop_helper(wsi_uid, crop_size, overlap, label, label_name, pts, img_name, info_dict)

            # Contour
            else:
                img_name, info_dict = crop_helper(wsi_uid, crop_size, overlap, label, label_name, pts, img_name, info_dict)

    # select 150 images to crop
    selected_info_dict = {
        "img_name": [],
        "label_Id": [],
        "label_Name": [],
        "top_left_pixel": [],
        "svs_name": []
    }
    count = 0
    while count < selection:
        i = int(np.random.choice(info_dict["img_name"]))
        x, y = np.array(info_dict["top_left_pixel"][i])[0], np.array(info_dict["top_left_pixel"][i])[1]
        img = np.array(slide.read_region((x, y), level, (crop_size, crop_size)))  # produce RGBA images with float64
        img = img[:, :, :3]  # drop A channel
        img = rescale(img, downscale_factor, clip=True, multichannel=True, preserve_range=True)
        if not is_low_contrast(img.astype(np.uint8), lower_percentile=5, upper_percentile=99):
            image_name = str(count)
            save_name = f"{wsi_uid}_{image_name}.png"
            io.imsave(crop_save_dir + save_name, img.astype(np.uint8))

            selected_info_dict["img_name"].append(image_name)
            selected_info_dict["label_Id"].append(info_dict["label_Id"][i])
            selected_info_dict["label_Name"].append(info_dict["label_Name"][i])
            selected_info_dict["top_left_pixel"].append(info_dict["top_left_pixel"][i])
            selected_info_dict["svs_name"].append(info_dict["svs_name"][i])

            count += 1

        info_dict["img_name"].remove(str(i))

    df = pd.DataFrame(data=selected_info_dict)
    save_name = f"{wsi_uid}.xlsx"
    df.to_excel(crop_save_dir + save_name)

    return True


if __name__ == "__main__":
    dataset_root = sys.argv[1]  #"/home/zhujiada/scratch/PAIP/colon"
    svs_load_dir = os.path.join(dataset_root, "svs_folder/")
    xml_load_dir = os.path.join(dataset_root, "xml_folder/")
    xml_fns = sorted(glob.glob(xml_load_dir + "*.xml") + glob.glob(xml_load_dir + "*.XML"))
    level = 0
    crop_size = 544  # 272*1/0.5
    downscale_factor = 0.5  # 0.5
    overlap = 0.5
    selection = 150

    crop_save_dir = os.path.join(dataset_root, f"crop_img/")
    os.makedirs(crop_save_dir, exist_ok=True)

    wsi_uid_pattern = "[a-zA-Z]*_PNI2021chall_train_[0-9]{4}"
    wsi_regex = re.compile(wsi_uid_pattern)

    for xml_fn in tqdm(xml_fns):
        wsi_uid = wsi_regex.findall(xml_fn)[0]
        slide = openslide.OpenSlide(svs_load_dir + wsi_uid + ".svs")

        print("Start: ", wsi_uid)
        crop(xml_fn, slide, level, crop_save_dir, wsi_uid, downscale_factor, crop_size, overlap, selection)
        print("End: ", wsi_uid)
