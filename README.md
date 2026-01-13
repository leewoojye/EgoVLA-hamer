# EgoVLA-hamer(HaMeR: Hand Mesh Recovery)
## Overview
This repository is a combination of the data preprocessing pipeline in the EgoVLA paper with the hamer repository. <br>
paper : [EgoVLA: Learning Vision-Language-Action Models from Egocentric Human Videos](https://arxiv.org/abs/2507.12440)
hamer repository : [HaMeR: Hand Mesh Recovery](https://github.com/geopavlakos/hamer)

## Data Preprocessing Pipeline
### Frame Sampling
- Sampling of image frames with 3 FPS and resizing to 384 * 384 standards <br>
About 27 frames are extracted when 3 FPS is applied based on 10 seconds
	
### Hand Pose Estimation (HaMeR)
- **Setup Detector (ViTDet) & Keypoint Detector (ViTPose)** : Run the hand, keypoint detection module required to extract MANO parameters
- **Run HaMeR** : Perform HaMeR on hand information extracted by ViT to extract MANO parameters
- **Extract MANO PCA components** : The extracted 45-dimensional MANO parameter is post-processed into hands_components, hands_means defined as constants in the MANO model (.pkl) and converted into 15-dimensional MANO parameters
### Visualization
- Get mesh files created by HaMeR for each frame to visually check that the hand is recognized well

## Installation
First you need to clone the repo:
```
git clone --recursive https://github.com/geopavlakos/hamer.git
cd hamer
```

We recommend creating a virtual environment for HaMeR. You can use venv:
```bash
python3.10 -m venv .hamer
source .hamer/bin/activate
```

or alternatively conda:
```bash
conda create --name hamer python=3.10
conda activate hamer
```

Then, you can install the rest of the dependencies. This is for CUDA 11.7, but you can adapt accordingly:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
pip install -e .[all]
pip install -v -e third-party/ViTPose
```

You also need to download the trained models:
```bash
bash fetch_demo_data.sh
```

Besides these files, you also need to download the MANO model. Please visit the [MANO website](https://mano.is.tue.mpg.de) and register to get access to the downloads section.  We only require the right hand model. You need to put `MANO_RIGHT.pkl` under the `_DATA/data/mano` folder.

## Data Preparation
You should set the input data path(`VIDEO_PATH`), and natural language instructions(`INSTRUCTION`) in the `run_pipeline.sh` file as below. (setting `OUTPUT_ROOT` is optional) And you have to place the input video data in the path you set. For example, you can put your videos under the `data_in/egovla_demo` folder. Data preprocessed by the pipeline is newly created in a folder located in root directory named `data_out_${current time}`. 
``` bash
# 기본값 설정
VIDEO_PATH=${1:-"data_in/egovla_demo/video_name.MOV"}
OUTPUT_ROOT=${2:-"data_out"}
INSTRUCTION=${3:-"Get the ball out"}
```
The output folder consists of an npz file containing the mano parameter extracted for each frame and a mesh image generated for each frame.

## Running Pipeline
You can simply run `run_pipeline.sh` file to run all steps of the pipeline, where OUTPUT_ROOT is the name of the root directory folder where the output will be stored. 
``` bash
bash run_pipeline.sh
```

## For more information about HaMeR ...
You can find out more about the hammer in the original repository. [HaMeR: Hand Mesh Recovery](https://github.com/geopavlakos/hamer)
