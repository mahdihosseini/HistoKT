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
from skimage.transform import downscale_local_mean


def crop_helper(slide, level, crop_save_dir, wsi_uid, downscale_factor, crop_size, overlap, label, label_name, pts, mask, img_name, info_dict):
    temp = np.array(pts)
    x = np.min(temp[:, 0])
    y = np.min(temp[:, 1])
    while x < np.max(temp[:, 0]):
        while y < np.max(temp[:, 1]):
            if img_name % 50 == 0:
                print((np.min(temp[:, 0]), np.min(temp[:, 1])))
            img = np.array(slide.read_region((x, y), level, (crop_size, crop_size)))
            mask_crop = mask[x:x+crop_size, y:y+crop_size]

            info_dict["img_name"].append(str(img_name))
            info_dict["mask_name"].append(str(img_name)+"_mask")
            info_dict["label_Id"].append(label)
            info_dict["label_Name"].append(label_name)
            info_dict["top_left_pixel"].append((x, y))
            info_dict["svs_name"].append(str(wsi_uid))

            image_name = info_dict["img_name"][img_name]
            save_name = f"{wsi_uid}_{image_name}.png"
            img = downscale_local_mean(img, (downscale_factor, downscale_factor))
            io.imsave(crop_save_dir + save_name, img)
            mask_name = info_dict["mask_name"][img_name]
            save_name = f"{wsi_uid}_{mask_name}.png"
            mask_crop = downscale_local_mean(mask_crop, (downscale_factor, downscale_factor))
            io.imsave(crop_save_dir + save_name, mask_crop)

            img_name += 1
            y = round(y + crop_size * overlap)

        y = np.min(temp[:, 1])
        x = round(x + crop_size * overlap)
    return img_name, info_dict


def crop(xml_fn, slide, level, crop_save_dir, wsi_uid, downscale_factor, crop_size, overlap, edge):
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

    mask = np.zeros((dest_h, dest_w))
    info_dict = {
        "img_name": [],
        "label_Id": [],
        "label_Name": [],
        "top_left_pixel": [],
        "mask_name": [],
        "svs_name": []
    }
    img_list = list()
    mask_list = list()
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
            pts = [np.array(pts, dtype=np.int32)]
            mask = cv2.drawContours(mask, pts, -1, label + 10, -1)

            # cropping
            img_name, info_dict = crop_helper(slide, level, crop_save_dir, wsi_uid, downscale_factor, crop_size, overlap, label, label_name, pts, mask, img_name, info_dict)

        for pts in cntr_pts:
            pts = [np.array(pts, dtype=np.int32)]
            # Curved line
            if label == 2:
                mask = cv2.polylines(mask, pts, isClosed=False, color=label, thickness=1)
                # cropping
                img_name, info_dict = crop_helper(slide, level, crop_save_dir, wsi_uid, downscale_factor, crop_size, overlap, label, label_name, pts, mask, img_name, info_dict)

            # Contour
            else:
                mask = cv2.drawContours(mask, pts, -1, label, -1)
                # cropping
                img_name, info_dict = crop_helper(slide, level, crop_save_dir, wsi_uid, downscale_factor, crop_size, overlap, label, label_name, pts, mask, img_name, info_dict)

    df = pd.DataFrame(data=info_dict, index=[0])
    save_name = f"{wsi_uid}.xlsx"
    df.to_excel(crop_save_dir + save_name)

    return True


if __name__ == "__main__":
    dataset_root = "/home/zhujiada/scratch/PAIP/PAIP_example_data"
    svs_load_dir = os.path.join(dataset_root, "svs_folder/")
    xml_load_dir = os.path.join(dataset_root, "xml_folder/")
    xml_fns = sorted(glob.glob(xml_load_dir + "*.xml") + glob.glob(xml_load_dir + "*.XML"))
    level = 0
    crop_size = 544  # 272*1/0.5
    downscale_factor = 2  # 1/0.5
    overlap = 0.5
    edge = 0.1

    crop_save_dir = os.path.join(dataset_root, f"crop_img/")
    os.makedirs(crop_save_dir, exist_ok=True)

    wsi_uid_pattern = "[a-zA-Z]*_PNI2021chall_train_[0-9]{4}"
    wsi_regex = re.compile(wsi_uid_pattern)

    for xml_fn in tqdm(xml_fns):
        wsi_uid = wsi_regex.findall(xml_fn)[0]
        slide = openslide.OpenSlide(svs_load_dir + wsi_uid + ".svs")

        print("Start: ", wsi_uid)
        crop(xml_fn, slide, level, crop_save_dir, wsi_uid, downscale_factor, crop_size, overlap, edge)
        print("End: ", wsi_uid)
