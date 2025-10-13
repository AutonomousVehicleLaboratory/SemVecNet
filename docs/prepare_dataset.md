## Prepare Dataset

There are 3 steps to get everything installed
- Creating the TensorRT version of [HRNet](https://github.com/NVIDIA/semantic-segmentation) 
- Passing the datasets through the Semantic Map Generation Pipeline
- Generate the custom annotation files as in MapTRv2

**a. Creating the TensorRT version of [HRNet](https://github.com/NVIDIA/semantic-segmentation)**

Download the assets (weights and config) here [link](https://drive.google.com/drive/folders/1bV7mnjSOYg2M-LBbM59C_eFDIvqvZZ7i?usp=drive_link), put it in a folder, this will be your ASSETS_PATH.

We use the docker image [naruarjun/tensorrt-nvidia-semseg](https://hub.docker.com/r/naruarjun/tensorrt-nvidia-semseg)
```shell
# Pull docker image
docker pull naruarjun/tensorrt-nvidia-semseg

# Create a docker container
docker run --name hrnet-1 -t -d --gpus all --shm-size=16gb --net=host --volume="/media:/media:rw" --volume="/home:/home:rw" naruarjun/tensorrt-nvidia-semseg

# Enter the container
docker exec -it hrnet-1 /bin/bash
```


```shell
# Clone forked and modified Nvidia Semantic Segmentation Monorepo
git clone https://github.com/naruarjun/TensorRT-tutorial.git
cd TensorRT-tutorial

# Change ASSETS_PATH(Where the segmentation weights and config are stored) in config.py Line 52

# Convert to ONNX
# dataset-name mapillary
# checkpoint name mapillary_ocrnet.HRNet_Mscale_fast-rattlesnake.pth
python convert-onnx.py --dataset {dataset-name} --cv 0 --bs_val 1 --eval folder --eval_folder ./imgs/test_imgs --dump_assets --dump_all_images --n_scales 0.5,1.0 --snapshot ASSETS_PATH/seg_weights/path-to-pth.pth --arch ocrnet.HRNet_Mscale --result_dir logs/onnx-export/log-name

# Create TensorRT engine to be used
trtexec --onnx=hrnet.onnx --fp16 --workspace=64 --buildOnly --saveEngine=hrnet.engine
```

**b. Passing the datasets through the Semantic Map Generation Pipeline**

We use the docker image [naruarjun/tensorrt-nvidia-semseg](https://hub.docker.com/r/naruarjun/tensorrt-nvidia-semseg)

```shell
# For AV2
# Change paths in semantic_mapping/src/node_config/
cd SemVecNet
python python tools/av2_semantic_mapping.py
```
**c. Generate the custom annotation files as in MapTRv2**

We use the docker image [naruarjun/semvecnet:v2](https://hub.docker.com/r/naruarjun/semvecnet)

This is exactly as it is in [MapTRv2](https://github.com/hustvl/MapTR/blob/maptrv2/docs/prepare_dataset.md)