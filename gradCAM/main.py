from pytorch_grad_cam import GradCAM, ScoreCAM,
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet18

import argparse
import yaml

model = resnet18(pretrained=True)
target_layer = model.layer4[-1]
input_tensor = # Create an input tensor image for your model..
# Note: input_tensor can be a batch tensor with several images!

# Construct the CAM object once, and then re-use it on many images:
cam = GradCAM(model=model, target_layer=target_layer, use_cuda=args.use_cuda)

# If target_category is None, the highest scoring category
# will be used for every image in the batch.
# target_category can also be an integer, or a list of different integers
# for every image in the batch.
target_category = 281

# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

# In this example grayscale_cam has only one image in the batch:
grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(rgb_img, grayscale_cam)

def parse_config(args):
    if(args.config):
        with open(args.config,'r') as params:
            print(yaml.safe_load(params))
    return args

if(__name__=="__main__"):
    parser = argparse.ArgumentParser(description="gradCAM")

    parser.add_argument(
        '--image_path', dest='image_path', type=str,
        help="Path to input image location"
    )
    parser.add_argument(
        '--output_path', dest='output_path',
        default = "./output", type=str,
        help="Path to output location: Default = './output'"
    )
    parser.add_argument(
        '--config', dest='config',
        default = "./gradcam_config.yaml", type=str,
        help="Configuration file file path: Default ='./gradcam_config.yaml'"
    )
    args = parser.parse_args()
    parse_config(args)
