""" Generate Semantic Segmentation for Argoverse Dataset
"""
import numpy as np
import os
import os.path as osp
import sys
import cv2
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt


# Add src directory into the path
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "../")))

import semantic_mapping.src.utils.mapillary_visualization as mapillary_visl
from semantic_mapping.src.hrnet.hrnet_semantic_segmentation_tensorrt import HRNetSemanticSegmentationTensorRT, get_custom_hrnet_args
from semantic_mapping.src.dynamic_map import DynamicMap
from semantic_mapping.src.node_config.argo_cfg import get_cfg_defaults
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.structures.sweep import Sweep

CAM_NAMES = ['ring_front_center', 'ring_front_right', 'ring_front_left',
    'ring_rear_right','ring_rear_left', 'ring_side_right', 'ring_side_left']

FAIL_LOGS = [
    # official
    '75e8adad-50a6-3245-8726-5e612db3d165',
    '54bc6dbc-ebfb-3fba-b5b3-57f88b4b79ca',
    'af170aac-8465-3d7b-82c5-64147e94af7d',
    '6e106cf8-f6dd-38f6-89c8-9be7a71e7275',
    # observed
    '01bb304d-7bd8-35f8-bbef-7086b688e35e',
    '453e5558-6363-38e3-bf9b-42b5ba0a6f1d'
]


def load_argoverse(dataset_dir, split):
    argoverse2_dir = Path(os.path.join(str(dataset_dir), split))
    av_loader = AV2SensorDataLoader(data_dir=Path(argoverse2_dir), labels_dir=Path(argoverse2_dir))
    lidar_paths = sorted(argoverse2_dir.glob(f'**/sensors/lidar/*.feather'), key=lambda x: int(x.stem))
    return av_loader, lidar_paths

def get_image_attr(file_abs_path):
    split = str(file_abs_path).split("/")
    log_name = split[-5]
    camera_name = split[-2]
    timestamp = int(split[-1].split(".")[0])
    return log_name, camera_name, timestamp

def get_lidar_attr(file_abs_path):
    split = str(file_abs_path).split("/")
    log_name = split[-4]
    timestamp = int(split[-1].split(".")[0])
    return log_name, timestamp

def get_timestamp(file_abs_path):
    split = str(file_abs_path).split("/")
    timestamp = int(split[-1].split(".")[0])
    return timestamp

def get_timestamp_dict(dict_):
    timestamp_dict = {}
    for name, file_abs_path in dict_.items():
        split = str(file_abs_path).split("/")
        timestamp = int(split[-1].split(".")[0])
        timestamp_dict[name] = timestamp
    return timestamp_dict

def get_sample_image(file_abs_path):
    image = cv2.imread(str(file_abs_path), cv2.IMREAD_COLOR)
    if image is None:
        print('Error: Could not read image.', file_abs_path)
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def get_sample_lidar(file_abs_path):
    sweep = Sweep.from_feather(file_abs_path)
    return sweep

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

    # breakpoint()
    ## ========== semantic segmentation
    image_in_resized = segmentation.preprocess(image_in_resized)
    image_out_resized = segmentation.segmentation(image_in_resized)
    # breakpoint()
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

def get_semantic_image_path(image_path):
    if "train" in str(image_path):
        new_path = str(image_path).replace("/cogrob-avl-dataset/argoverse2/sensor/train", "/cogrob-avl-dataset/argoverse2/sensor/train_segmented")
        new_path = new_path.replace(".jpg", ".png")
        return new_path
    elif "val" in str(image_path):
        new_path = str(image_path).replace("/cogrob-avl-dataset/argoverse2/sensor/val", "/cogrob-avl-dataset/argoverse2/sensor/val_segmented")
        new_path = new_path.replace(".jpg", ".png")
        return new_path
    elif "test" in str(image_path):
        new_path = str(image_path).replace("/cogrob-avl-dataset/argoverse2/sensor/test", "/cogrob-avl-dataset/argoverse2/sensor/test_segmented")
        new_path = new_path.replace(".jpg", ".png")
        return new_path

def get_semantic_map_path(image_path):
    if "train" in str(image_path):
        new_path = str(image_path).replace("/cogrob-avl-dataset/argoverse2/sensor/train", "/cogrob-avl-dataset/argoverse2/sensor/train_segmented_map")
        new_path = new_path.replace(".jpg", ".png")
        return new_path
    elif "val" in str(image_path):
        new_path = str(image_path).replace("/cogrob-avl-dataset/argoverse2/sensor/val", "/cogrob-avl-dataset/argoverse2/sensor/val_segmented_map")
        new_path = new_path.replace(".jpg", ".png")
        return new_path
    elif "test" in str(image_path):
        new_path = str(image_path).replace("/cogrob-avl-dataset/argoverse2/sensor/test", "/cogrob-avl-dataset/argoverse2/sensor/test_segmented_map")
        new_path = new_path.replace(".jpg", ".png")
        return new_path


def save_semantic_image(semantic_image, new_path, RGB2BGR=True):
    """ save semantic image or map"""
    Path('/'.join(new_path.split('/')[:-1])).mkdir(parents=True, exist_ok=True)
    if RGB2BGR == True:
        semantic_image = cv2.cvtColor(semantic_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(new_path, semantic_image)


def get_dynamic_map(av_loader, log_id, cfg, dm_dict):
    location = av_loader.get_city_name(log_id)
    if not location in dm_dict:
        print("Adding map location:", location)
        dm_dict[location] = DynamicMap(cfg)
    return dm_dict[location]


def generate_semantic_image_on_argoverse(av_loader, segmentation, cfg, seg_color_ref):
    # Go over scenes
    dm_dict = {} # store dynamic map saparately for each location
    vertical_center_cam_crop = (750, 1650) # (750, 1750)
    cam_crop_filter = {}
    for cam_name in CAM_NAMES:
        cam_crop_filter[cam_name] = {}
        if "center" in cam_name:
            cam_crop_filter[cam_name]['segmentation_input'] = [750, 1650]
            cam_crop_filter[cam_name]['lidar_filter'] = [750, 1450]
        elif 'ring_front_left' in cam_name:
            cam_crop_filter[cam_name]['segmentation_input'] = [350, 1450]
            cam_crop_filter[cam_name]['lidar_filter'] = [350, 1350]
        else:
            cam_crop_filter[cam_name]['segmentation_input'] = [350, 1550]
            cam_crop_filter[cam_name]['lidar_filter'] = [350, 1350]
            

    center_crop_min, center_crop_max = vertical_center_cam_crop
    fig, axes = plt.subplots(1, 3, figsize=(15,10))
    all_log_ids = av_loader.get_log_ids()
    # all_log_ids = ['3b68c074-1680-3a93-92e5-5b711406f2fe']
    # all_log_ids = ['5bd6bd4d-3c89-3794-9935-2d044ce6ef37']
    # all_log_ids = ['7e3d8631-3b7d-38c1-b833-ee7cfa7235ca']
    for log_id in tqdm(all_log_ids):
        if log_id in FAIL_LOGS:
            continue
        print("LOG", log_id)
        dm = get_dynamic_map(av_loader, log_id, cfg, dm_dict)
        lidar_all_paths = av_loader.get_ordered_log_lidar_fpaths(log_id)
        for lidar_path in lidar_all_paths:
            sweep = get_sample_lidar(lidar_path)
            log_name, timestamp = get_lidar_attr(lidar_path)
            # extract all nearby cameras
            image_paths = {}
            for cam_name in CAM_NAMES:
                closest_image = av_loader.get_closest_img_fpath(log_name, cam_name, timestamp)
                if closest_image is None:
                    continue
                image_paths[cam_name] = closest_image
            if len(image_paths.keys()) < len(CAM_NAMES):
                print("YO ME HERE")
                continue
            semantic_images = {}
            images = {}
            for cam_name, image_path in image_paths.items():
                new_path = get_semantic_image_path(image_path)
                image = get_sample_image(image_path)
                center_crop_min, center_crop_max = cam_crop_filter[cam_name]['segmentation_input'][0], cam_crop_filter[cam_name]['segmentation_input'][1]
                image = image[center_crop_min:center_crop_max, :]
                if not os.path.exists(new_path):
                    semantic_image = get_semantic_image(segmentation, image, seg_color_ref, image_scale=0.5)
                    save_semantic_image(semantic_image, new_path, RGB2BGR=False)
                else:
                    semantic_image = cv2.imread(new_path) # the semantic image is in BGR
                semantic_images[cam_name] = semantic_image
                images[cam_name] = image
            try:
                T_pcd_to_map = av_loader.get_city_SE3_ego(log_name, get_timestamp(image_path))
            except:
                print(f'Pose not available for log {log_name} and timestamp {timestamp}')
                continue
            map_path = get_semantic_map_path(image_paths['ring_front_center'])
            if not os.path.exists(map_path):
                map, map_rendered, pcd_in_range, pcd_label = dm.mapping_argoverse(sweep.xyz, sweep.intensity,
                    av_loader,
                    semantic_images,
                    CAM_NAMES,
                    timestamp, # lidar timestamp
                    get_timestamp_dict(image_paths), # camera_timestamp_dict
                    log_name,
                    T_pcd_to_map.transform_matrix,
                    T_pcd_to_map.transform_matrix,
                    cam_crop_filter)
                save_semantic_image(map_rendered, map_path)
            else:
                map_rendered = cv2.imread(map_path) # the semantic image is in BGR


            # axes[0].cla()
            # axes[1].cla()
            # axes[2].cla()
            
            # axes[0].imshow(images["ring_front_center"])
            # axes[1].imshow(semantic_images["ring_front_center"])
            # axes[2].imshow(map_rendered)
            # plt.pause(0.00005)
    

def main():
    cfg = get_cfg_defaults()
    # dm = DynamicMap(cfg)

    seg_color_ref = mapillary_visl.get_labels(cfg.VISION_SEM_SEG.DATASET_CONFIG)
    
    seg = HRNetSemanticSegmentationTensorRT(get_custom_hrnet_args(cfg))

    av_loader, _ = load_argoverse(dataset_dir = cfg.AV2_PATH, split = cfg.AV2_SPLIT)
    generate_semantic_image_on_argoverse(av_loader, seg, cfg, seg_color_ref)


if __name__ == '__main__':
    main()
