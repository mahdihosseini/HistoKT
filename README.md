# HistoKT: Cross Knowledge Transfer in Computational Pathology

> [**HistoKT: Cross Knowledge Transfer in Computational Pathology**](link),            
> Ryan Zhang, Annie Zhu, Stephen Yang, Mahdi S. Hosseini, Konstantinos N. Plataniotis        

![](figures/workflow.jpg)

## Overview

In computational pathology, the lack of well-annotated datasets obstructs the application of deep learning techniques. Since pathologist time is expensive, dataset curation is intrinsically difficult. Thus, many CPath workflows involve transferring learned knowledge between various image domains through transfer learning. Currently, most transfer learning research follows a model-centric approach, tuning network parameters to improve transfer results over few datasets. In this paper, we take a data-centric approach to the transfer learning problem and examine the existence of generalizable knowledge between histopathological datasets. First, we create a standardization workflow for aggregating existing histopathological data. We then measure inter-domain knowledge by training ResNet18 models across multiple histopathological datasets, and cross-transferring between them to determine the quantity and quality of innate shared knowledge. Additionally, we use weight distillation to share knowledge between models without additional training. We find that hard to learn, multi-class datasets benefit most from pretraining, and a two stage learning framework incorporating a large source domain such as ImageNet allows for better utilization of smaller datasets. Furthermore, we find that weight distillation enables models trained on purely histopathological features to outperform models using external natural image data.

## Results

## Table of Contents

## Getting Started 

### Dependencies
* Requirements are specified in requirements.txt

```text
argon2-cffi==20.1.0
async-generator==1.10
attrs==21.2.0
backcall==0.2.0
bleach==3.3.0
cffi==1.14.5
colorama==0.4.4
cycler==0.10.0
decorator==4.4.2
defusedxml==0.7.1
entrypoints==0.3
et-xmlfile==1.1.0
h5py==3.2.1
imageio==2.9.0
ipykernel==5.5.4
ipython==7.23.1
ipython-genutils==0.2.0
ipywidgets==7.6.3
jedi==0.18.0
Jinja2==3.0.0
joblib==1.0.1
jsonschema==3.2.0
jupyter==1.0.0
jupyter-client==6.1.12
jupyter-console==6.4.0
jupyter-core==4.7.1
jupyterlab-pygments==0.1.2
jupyterlab-widgets==1.0.0
kiwisolver==1.3.1
MarkupSafe==2.0.0
matplotlib==3.4.2
matplotlib-inline==0.1.2
mistune==0.8.4
nbclient==0.5.3
nbconvert==6.0.7
nbformat==5.1.3
nest-asyncio==1.5.1
networkx==2.5.1
notebook==6.3.0
numpy==1.20.3
openpyxl==3.0.7
packaging==20.9
pandas==1.2.4
pandocfilters==1.4.3
parso==0.8.2
pickleshare==0.7.5
Pillow==8.2.0
prometheus-client==0.10.1
prompt-toolkit==3.0.18
pyaml==20.4.0
pycparser==2.20
Pygments==2.9.0
pyparsing==2.4.7
pyrsistent==0.17.3
python-dateutil==2.8.1
pytz==2021.1
PyWavelets==1.1.1
pywin32==300
pywinpty==0.5.7
PyYAML==5.4.1
pyzmq==22.0.3
qtconsole==5.1.0
QtPy==1.9.0
scikit-image==0.18.1
scikit-learn==0.24.2
scipy==1.6.3
Send2Trash==1.5.0
six==1.16.0
sklearn==0.0
terminado==0.9.5
testpath==0.4.4
threadpoolctl==2.1.0
tifffile==2021.4.8
torch==1.8.1+cu102
torchaudio==0.8.1
torchvision==0.9.1+cu102
tornado==6.1
traitlets==5.0.5
typing-extensions==3.10.0.0
wcwidth==0.2.5
webencodings==0.5.1
widgetsnbextension==3.5.1

```
### Running the Code

#### Training Parameters

#### Training Output

## Code Organization

### Configs

### Dependency Extraction

### Metrics

### Version History
* 0.1 
    * Initial Release
