# Train and Test

The training and testing is exactly the same as [MapTRv2](https://github.com/hustvl/MapTR/blob/maptrv2/docs/train_eval.md) with different configs

```shell
cd SemVecNet/SemVecNet

# For AV2
./tools/dist_train.sh projects/configs/semvecnet/semvecnet_av2_centerline.py gpu_num

# For nuScenes
./tools/dist_train.sh projects/configs/semvecnet/semvecnet_nusc_centerline.py gpu_num
```