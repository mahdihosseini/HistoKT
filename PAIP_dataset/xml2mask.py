import os
import sys
import cv2
import glob
import numpy as np
import openslide
import re
import xml.etree.ElementTree as ET
from tqdm import tqdm
from skimage import io

def xml2mask(xml_fn, slide, level):
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

    annotations = etree.getroot()
    for annotation in annotations:
        label = int(annotation.get("Id"))

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
        for pts in cntr_pts:
            pts = [np.array(pts, dtype=np.int32)]
            # Curved line
            if label == 2:
                mask = cv2.polylines(mask, pts, isClosed=False, color=label, thickness=1)
            # Contour
            else:
                mask = cv2.drawContours(mask, pts, -1, label, -1)
    return mask


if __name__ == "__main__":
    dataset_root = "/home/zhujiada/scratch/PAIP/PAIP_example_data"
    svs_load_dir = os.path.join(dataset_root, "svs_folder/")
    xml_load_dir = os.path.join(dataset_root, "xml_folder/")
    xml_fns = sorted(glob.glob(xml_load_dir + "*.xml") + glob.glob(xml_load_dir + "*.XML"))
    level = 2

    mask_save_dir = os.path.join(dataset_root, f"mask_img_l{level}/")
    os.makedirs(mask_save_dir, exist_ok=True)

    wsi_uid_pattern = "[a-zA-Z]*_PNI2021chall_train_[0-9]{4}"
    wsi_regex = re.compile(wsi_uid_pattern)

    for xml_fn in tqdm(xml_fns):
        wsi_uid = wsi_regex.findall(xml_fn)[0]

        slide = openslide.OpenSlide(svs_load_dir + wsi_uid + ".svs")

        mask = xml2mask(xml_fn, slide, level)

        save_name = f"{wsi_uid}_l{level}_mask.tif"
        io.imsave(mask_save_dir + save_name, mask.astype(np.uint8), check_contrast=False)