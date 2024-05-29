""" Generate Semantic Map for nuScenes Dataset
"""
import numpy as np
import os
import os.path as osp
import sys
import cv2
from pathlib import Path
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.data_classes import RadarPointCloud, LidarPointCloud
from pyquaternion import Quaternion

# Add src directory into the path
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "../")))

import src.network.deeplab_v3_plus.data.utils.mapillary_visualization as mapillary_visl
from src.hrnet.hrnet_semantic_segmentation_tensorrt import HRNetSemanticSegmentationTensorRT, get_custom_hrnet_args
from src.dynamic_map import DynamicMap
from src.node_config.argo_cfg import get_cfg_defaults

camera_types = [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_FRONT_LEFT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT',
]

def load_nuscenes(version, dataroot):
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
    return nusc


def get_sample_pcd(nusc, sample):
    lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    file_rel_path = lidar_data['filename']
    file_abs_path = osp.join(nusc.dataroot, file_rel_path)
    pcd = LidarPointCloud.from_file(file_abs_path).points
    return pcd


def get_sample_image(nusc, sample):
    images = {}
    file_rel_paths = {}
    for camera_name in camera_types:
        camera_data = nusc.get('sample_data', sample['data'][camera_name])
        file_rel_path = camera_data['filename']
        file_abs_path = osp.join(nusc.dataroot, file_rel_path)
        image = cv2.imread(file_abs_path, cv2.IMREAD_COLOR)
        if image is None:
            print('Error: Could not read image.', file_abs_path)
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images[camera_name] = image
        file_rel_paths[camera_name] = file_rel_path
    return images, file_rel_paths


def get_transformations(nusc, sample):
    intrinsics_ = {}
    T_lidar_to_cameras = {}
    """ Get the intrinsics of camera and extrinsic from lidar to camera."""
    lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    lidar_calib = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
    T_lidar_in_ego = transform_matrix(lidar_calib['translation'], Quaternion(lidar_calib['rotation']), inverse=False)

    for camera_name in camera_types:
        camera_data = nusc.get('sample_data', sample['data'][camera_name])
        camera_calib = nusc.get('calibrated_sensor', camera_data['calibrated_sensor_token'])
        T_ego_to_camera = transform_matrix(camera_calib['translation'], Quaternion(camera_calib['rotation']), inverse=True)

        intrinsics = np.array(camera_calib['camera_intrinsic']).reshape(3,3)

        T_lidar_to_camera = np.matmul(T_ego_to_camera, T_lidar_in_ego)

        intrinsics_[camera_name] = intrinsics
        T_lidar_to_cameras[camera_name] = T_lidar_to_camera
    
    ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    T_ego_to_map = transform_matrix(ego_pose['translation'], Quaternion(ego_pose['rotation']), inverse=False)

    T_lidar_to_map = np.matmul(T_ego_to_map, T_lidar_in_ego)
    
    return intrinsics_, T_lidar_to_cameras, T_lidar_to_map, T_ego_to_map


def get_semantic_image(segmentation, image_in, seg_color_ref, image_scale=0.5):
    # Resize the image to reduce the memory overhead
    network_image_shape = (1920 // 2, 1440 // 2)
    if image_scale < 1:
        # width = int(image_in.shape[1] * image_scale)
        # height = int(image_in.shape[0] * image_scale)
        # dim = (width, height)
        dim = network_image_shape
        image_in_resized = cv2.resize(image_in, dim, interpolation=cv2.INTER_AREA)
    else:
        image_in_resized = image_in

    ## ========== semantic segmentation
    image_in_resized = segmentation.preprocess(image_in_resized)
    image_out_resized = segmentation.segmentation(image_in_resized)
    image_out_resized = np.reshape(image_out_resized, (1, 1440 // 2, 1920 // 2))
    image_out = image_out_resized.astype(np.uint8).squeeze()

    ## ========== Visualize semantic images
    # Convert network label to color
    colored_output = mapillary_visl.apply_color_map(image_out, seg_color_ref)
    colored_output = np.squeeze(colored_output)
    image_colored = colored_output.astype(np.uint8)

    if image_scale != 1.0:
        new_shape = (image_in.shape[1], image_in.shape[0])
        # NOTE: we use INTER_NEAREST because values are discrete labels
        image_colored = cv2.resize(image_colored, new_shape,
                    interpolation=cv2.INTER_NEAREST)

    return image_colored


def get_semantic_image_path(nusc, relative_path, dirname="semantic/image"):
    new_path = os.path.join(nusc.dataroot, dirname, '/'.join(relative_path.split('/')[1::]))[:-4] + '.png'
    return new_path


def save_semantic_image(semantic_image, new_path, RGB2BGR=True):
    """ save semantic image or map"""
    Path('/'.join(new_path.split('/')[:-1])).mkdir(parents=True, exist_ok=True)
    if RGB2BGR == True:
        semantic_image = cv2.cvtColor(semantic_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(new_path, semantic_image)


def get_dynamic_map(nusc, scene, cfg, dm_dict):
    log_token = scene['log_token']
    log = nusc.get('log', log_token)
    location = log['location']
    if not location in dm_dict:
        print("Adding map location:", location)
        dm_dict[location] = DynamicMap(cfg)
    return dm_dict[location]


def generate_semantic_map_on_nuscenes(nusc, segmentation, cfg, seg_color_ref):
    dm_dict = {} # store dynamic map saparately for each location
    import pickle
    with open('dynamic_map.pickle', 'rb') as handle:
        dm_dict = pickle.load(handle)
    from nuscenes.utils import splits
    val_scenes = splits.val
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(15,7))
    
    # Go over scenes
    for scene in tqdm(nusc.scene):
        scene_token = scene['token']
        if scene["name"] not in  val_scenes:
            continue
        sample_token = scene['first_sample_token']

        dm = get_dynamic_map(nusc, scene, cfg, dm_dict)
        
        # Go over samples in each scene
        while sample_token != '':
            sample = nusc.get('sample', sample_token)
            
            pcd = get_sample_pcd(nusc, sample)
            images, relative_paths = get_sample_image(nusc, sample)
            intrinsics_, T_lidar_to_cameras, T_lidar_to_map, T_ego_to_map = get_transformations(nusc, sample)

            semantic_images = {}
            for cam_name in camera_types:
                semantic_image_path = get_semantic_image_path(nusc, relative_paths[cam_name], dirname='semantic/image')
                semantic_map_path = get_semantic_image_path(nusc, relative_paths[cam_name], dirname='semantic/map')
                if not os.path.exists(semantic_image_path):
                    semantic_image = get_semantic_image(segmentation, images[cam_name], seg_color_ref, image_scale=0.5)
                    save_semantic_image(semantic_image, semantic_image_path, RGB2BGR=False)
                else:
                    semantic_image = cv2.imread(semantic_image_path) # the semantic image is in BGR
                semantic_images[cam_name] = semantic_image

            map, map_rendered, pcd_in_range, pcd_label = dm.mapping_nuscenes(
                pcd,
                semantic_images,
                camera_types,
                intrinsics_,
                T_lidar_to_cameras,
                T_lidar_to_map,
                T_ego_to_map
            )

            save_semantic_image(map_rendered, semantic_map_path)
            sample_token = sample['next']

            # axes[0].cla()
            # axes[1].cla()
            
            # axes[0].imshow(images['CAM_FRONT'])
            # axes[1].imshow(map_rendered)
            # plt.pause(0.05)


def main():
    cfg = get_cfg_defaults()

    network_cfg = cfg.VISION_SEM_SEG.SEM_SEG_NETWORK
    seg_color_ref = mapillary_visl.get_labels(network_cfg.DATASET_CONFIG)
    
    seg = HRNetSemanticSegmentationTensorRT(get_custom_hrnet_args())

    nusc = load_nuscenes(version='v1.0-trainval', dataroot='/home/semantic_mapping/nuscenes/nuScenes')
    full_map = generate_semantic_map_on_nuscenes(nusc, seg, cfg, seg_color_ref)


if __name__ == '__main__':
    main()
