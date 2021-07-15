import argparse
import yaml
import torch
import os
import cv2
import numpy as np

from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image
import torchvision.transforms as transforms

# import model
from resnet18 import resnet18

# import dataset
from datasets import TransformedDataset
from datasets import ADPDataset

root_path = os.path.dirname(os.path.realpath(__file__))

# defined in image_transformed/custom_augmentations.py
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

dataset_classes = {
    "ADP": 9,
    "GlaS_transformed": 2,
    "AJ-Lymph_transformed": 3,
    "BACH_transformed": 4,
    "OSDataset_transformed": 4,
    "MHIST_transformed": 2,
    "CRC_transformed": 7,
    "PCam_transformed": 2
}

# image size
size = 272

def img_to_tensor(img_path,dataset):
    img = cv2.imread(img_path, 1)[:, :, ::-1]
    img = (np.float32(img) / 255)
    trans = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(
        #     mean=transformed_norm_weights[dataset]["mean"],
        #     std=transformed_norm_weights[dataset]["std"]
        #     )
    ])
    tensor = trans(img).unsqueeze(0)
    return img, tensor

def generate_gradCAM(args):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if(args.dataset_name in dataset_classes.keys()):
        num_classes = dataset_classes[args.dataset_name]
    else:
        print("Error: Dataset not in dataset_classes dict")

    print("loading model...")
    # print(args.model_path)
    # cp = torch.load(args.model_path, map_location=device)
    # print(cp)
    model = resnet18(args.model_path,pretrained=True,device=device,num_classes=num_classes)
    target_layer = model.layer4[-1]
    print("success")

    print("initializing gradCAM")
    cam = GradCAM(model=model, target_layer=target_layer, use_cuda=args.use_cuda)
    print("success")

    img_name = os.path.splitext(os.path.basename(args.image_path))[0]
    # print(img_name)
    img, input_tensor = img_to_tensor(args.image_path,args.dataset_name)

    print("generating gradCAM output")
    grayscale_cam = cam(input_tensor=input_tensor,
                        aug_smooth=args.aug_smooth,
                        eigen_smooth=args.eigen_smooth)

    grayscale_cam = grayscale_cam[0, :]

    cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    print("success")

    print("saving gradCAM output")
    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    gb = gb_model(input_tensor)

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    cv2.imwrite(os.path.join(args.output_path,img_name+"_gradcam.jpg"),cam_image)
    cv2.imwrite(os.path.join(args.output_path,img_name+"_gradcam_gb.jpg"),gb)
    cv2.imwrite(os.path.join(args.output_path,img_name+"_gradcam_gbcam.jpg"),cam_gb)
    print("success")

if(__name__=="__main__"):
    mini_batch_size = 32
    num_workers = 4

    # dataset_name_list = ["ADP", "GlaS_transformed", "AJ-Lymph_transformed", "BACH_transformed", "OSDataset_transformed", "MHIST_transformed","AIDPATH_transformed", "CRC_transformed","PCam_transformed"]
    parser = argparse.ArgumentParser(description="gradCAM")

    parser.add_argument(
        '--image_path', dest='image_path', type=str,
        help="Path to input image location"
    )
    parser.add_argument(
        '--model_path', dest='model_path',
        help="Path to model checkpoint"
    )
    parser.add_argument(
        '--output_path', dest='output_path',
        default = "./output", type=str,
        help="Path to output location: Default = './output'"
    )
    parser.add_argument(
        '--dataset_name',dest='dataset_name', type=str,
        help="Dataset name"
    )
    parser.add_argument(
        '--use_cuda',dest='use_cuda',
        default=True,type=bool,
        help="Run with CUDA GPU acceleration"
    )
    parser.add_argument(
        '--aug_smooth',dest='aug_smooth',
        default=False,type=bool,
        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',dest='eigen_smooth',
        default=False,type=bool,
        help='Reduce noise by taking the first principle component'
            'of cam_weights*activations')

    args = parser.parse_args()
    # if not(os.path.exists(args.image_path) and os.path.isfile(args.image_path)):
    #     print("error: image_path is not defined")
    #     raise SystemExit
    # if not(os.path.exists(args.model_path) and os.path.isfile(args.model_path)):
    #     print("error: model_path is not defined")
    #     raise SystemExit
    if not(os.path.exists(args.output_path)):
        os.makedirs(args.output_path)
    generate_gradCAM(args)
