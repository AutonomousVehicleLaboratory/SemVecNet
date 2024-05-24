#!/bin/bash

# Base Image : nvcr.io/nvidia/pytorch:21.10-py3
MapTR_path="/workspace/SemVecNet"
python -m pip install --upgrade pip
python -m pip install wheel
python -m pip install numpy==1.21.3
python -m pip install mmcv-full==1.4.0
python -m pip install mmdet==2.14.0
python -m pip install mmsegmentation==0.14.1
python -m pip install timm
cd "$MapTR_path/mmdetection3d/"
python setup.py develop
python -m pip install scikit-image
python -m pip install numba==0.48.0
python -m pip install plyfile
python -m pip install nuscenes-devkit
python -m pip install networkx==2.2
cd ../projects/mmdet3d_plugin/maptr/modules/ops/geometric_kernel_attn/
python setup.py build install
cd "$MapTR_path"
python -m pip install -r requirement.txt
python -m pip install google-auth==2.22.0
python -m pip install google-auth-oauthlib==1.0.0
python -m pip install absl-py
python -m pip install grpcio
python -m pip install markdown
python -m pip install protobuf==4.23.4
python -m pip install tensorboard-data-server==0.7.1
python -m pip install Werkzeug==2.3.6
mkdir ckpts
cd ckpts
wget https://download.pytorch.org/models/resnet50-19c8e357.pth 
wget https://download.pytorch.org/models/resnet18-f37072fd.pth
python -m pip install numpy==1.21.3
python -m pip install lyft-dataset-sdk
python -m pip install tqdm ninja openmim similaritymeasures
python -m pip install ortools==9.2.9972 iso3166 chardet
python -m pip install setuptools==59.5.0
python -m pip install seaborn
cd ..
python -m pip install opencv-python==4.5.5.64
apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
# Numba error https://github.com/openstreams/wflow/commit/a7c5b8442722832915d8165d3afa2bc082c68c54
python -m pip install yapf==0.40.1
python -m pip install protobuf==3.20.*











