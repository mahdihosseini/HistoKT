# GradCAM Analysis

We utilize GradCAM heatmaps to visualize areas of example images relevant to activations of the network. 

## Generating GradCAM results 

To generate gradCAM heatmaps, run

```
python3 main.py --image_path=<path_to_example> \
 --model_path=<path_to_model> \
 --dataset_name=<name_of_dataset> \
 --output_path=<save_path> \

```

Note that the existing dataset class names are:
* ADP
* GlaS_transformed
* AJ-Lymph_transformed
* BACH_transformed
* OSDataset_transformed
* MHIST_transformed
* CRC_transformed
* PCam_transformed

## GradCAM Options

Additional options exist for generation of GradCAM heatmaps:

```
--use_cuda=True
```
Uses CUDA during generation of heatmaps


```
--aug_smooth
```
Applies test time augmentation to smooth the CAM


```
--eigen_smooth
```
Reduces noise by taking the first principle component of cam_weights * activations
