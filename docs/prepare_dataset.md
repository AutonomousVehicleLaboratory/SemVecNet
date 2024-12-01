## Prepare Dataset

There are 3 steps to get everything installed
- Creating the TensorRT version of [HRNet](https://github.com/NVIDIA/semantic-segmentation) 
- Passing the datasets through the Semantic Map Generation Pipeline
- Generate the custom annotation files as in MapTRv2

**a. Creating the TensorRT version of [HRNet](https://github.com/NVIDIA/semantic-segmentation)**

We use the docker image ```nvcr.io/nvidia/pytorch:23.04-py3```
```shell
# Clone forked and modified Nvidia Semantic Segmentation Monorepo
git clone https://github.com/naruarjun/TensorRT-tutorial.git
cd TensorRT-tutorial

# Change ASSETS_PATH(Where the segmentation weights are stored) in config.py Line 52

# Convert to ONNX
python convert-onnx.py --dataset {dataset-name} --cv 0 --bs_val 1 --eval folder --eval_folder ./imgs/test_imgs --dump_assets --dump_all_images --n_scales 0.5,1.0 --snapshot ASSETS_PATH/seg_weights/path-to-pth.pth --arch ocrnet.HRNet_Mscale --result_dir logs/onnx-export/log-name

# Create TensorRT engine to be used
trtexec --onnx=hrnet.onnx --fp16 --workspace=64 --buildOnly --saveEngine=hrnet.engine
```

**b. Passing the datasets through the Semantic Map Generation Pipeline**

We use the docker image ```naruarjun/semvecnet:v2```

```shell
# For AV2
# Change paths in semantic_mapping/src/node_config/
cd SemVecNet
python python tools/av2_semantic_mapping.py
```
**c. Generate the custom annotation files as in MapTRv2**

This is exactly as it is in [MapTRv2](https://github.com/hustvl/MapTR/blob/maptrv2/docs/prepare_dataset.md)