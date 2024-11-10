"""
Create map tile for dynamic and efficient mapping
Author: Henry Zhang
"""


# Dependencies
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

from semantic_mapping.src.utils.utils import homogenize, dehomogenize
from semantic_mapping.src.utils.renderer import render_bev_map, apply_filter
from semantic_mapping.src.utils.confusion_matrix import ConfusionMatrix, adjust_for_mapping
from semantic_mapping.src.utils.utils import color_pcd_by_distance, project_colored_pcd_on_image

# Classes
class DynamicMap():
    """ Dynamic Map
    Expose interface that are transparent to outside as a big map
    Internal implementation using map tiles
    """
    def __init__(self, cfg, logger=None) -> None:
        self.logger = logger

        self.depth_method = cfg.MAPPING.DEPTH_METHOD
        self.pcd_range_max = cfg.MAPPING.PCD.RANGE_MAX
        self.use_pcd_intensity = cfg.MAPPING.PCD.USE_INTENSITY
        self.use_distance = cfg.MAPPING.PCD.USE_DISTANCE
        self.use_angular_velocity = cfg.MAPPING.PCD.USE_ANGULAR_VELOCITY
        
        self.map = None             # current map in logits
        self.map_rendered = None    # current map rendered
        self.render_ego_centric = cfg.MAPPING.RENDER_EGO_CENTRIC # render ego centric map
        self.resolution = cfg.MAPPING.RESOLUTION
        
        self.label_names = cfg.LABELS_NAMES
        self.label_colors = np.array(cfg.LABEL_COLORS)
        self.color_remap_source = np.array(cfg.COLOR_REMAP_SOURCE)
        self.color_remap_dest = np.array(cfg.COLOR_REMAP_DEST)

        self.map_tile = {}          # Dictionary of smaller map tiles
        self.tile_size_meter = cfg.MAPPING.TILE_SIZE_METER # meter
        self.tile_size_d = int(self.tile_size_meter / self.resolution) # grid size
        self.pad_num = cfg.MAPPING.TILE_PAD_NUMBER   # number of tiles on each direction
        self.pad_total_num = 2 * self.pad_num + 1
        self.map_boundary = None
        self.render_global_map_flag = cfg.MAPPING.RENDER_GLOBAL_MAP_FLAG
        
        self.map_height = int(self.pad_total_num * self.tile_size_d)
        self.map_width = int(self.pad_total_num * self.tile_size_d)
        self.map_depth = len(self.label_names)

        self.ego_x_min, self.ego_x_max, self.ego_y_min, self.ego_y_max = cfg.MAPPING.RANGE

        self.ego_x_max_d  = int(self.ego_x_max / self.resolution)
        self.ego_x_min_d  = int(self.ego_x_min / self.resolution)
        self.ego_y_max_d  = int(self.ego_y_max / self.resolution)
        self.ego_y_min_d  = int(self.ego_y_min / self.resolution)

        # load confusion matrix, we may take log probability instead
        if cfg.MAPPING.CONFUSION_MTX.LOAD_PATH != "":
            confusion_matrix = ConfusionMatrix(load_path=cfg.MAPPING.CONFUSION_MTX.LOAD_PATH)
            confusion_matrix.merge_labels(cfg.SRC_INDICES, cfg.DST_INDICES)
            self.confusion_matrix = confusion_matrix.get_submatrix(cfg.LABELS, to_probability=True, use_log=False)
            self.confusion_matrix = adjust_for_mapping(self.confusion_matrix, factor=cfg.MAPPING.REWEIGHT_FACTOR)
            print('confusion_matrix:', self.confusion_matrix)
            self.confusion_matrix = np.log(self.confusion_matrix)
            # self.confusion_matrix = np.ones_like(self.confusion_matrix)
        else:
            # Use Identity confusion matrix
            self.confusion_matrix = np.eye(len(self.label_names))


    def mapping(self, 
                pcd, 
                semantic_image, 
                intrinsics,
                T_lidar_to_camera, 
                T_lidar_to_map,
                T_ego_to_map):
        """
        Receives the semantic segmentation image, the pose of the vehicle, and the calibration of the camera,
        we will build a semantic point cloud, then project it into the 2D bird's eye view coordinates.

        Args:
            pcd: np.ndarray with shape (4, N), point cloud of type (x, y, z, intensity)
            semantic_image: np.ndarray with shape (H, W, C), 2D semantic image
            intrinsics: np.ndarray with shape (3, 3), camera intrinsic matrix
            T_lidar_to_camera: np.ndarray with shape (4, 4), the transformation from lidar frame to camera frame.
            T_lidar_to_map: np.ndarray with shape (4,4), the transformation matrix from lidar frame to map frame,
            T_ego_to_map: np.ndarray with shape (4,4), the transformation matrix from ego vehicle frame to map frame.
        """
        # Initialize the map
        self.check_map_origin(T_lidar_to_map)

        if self.depth_method == 'points_map' or self.depth_method == 'points_raw':
            # print("pcd:", pcd.shape)
            # pcd_colored = color_pcd_by_distance(pcd, pcd_range_x_max=30, pcd_range_x_min=1)
            # print("pcd_colored", pcd_colored.shape)
            # image_with_pcd = project_colored_pcd_on_image(np.array(semantic_image), pcd_colored, intrinsics, T_lidar_to_camera, k=3, pcd_range_max=self.pcd_range_max)

            pcd_in_range, pcd_label = self.project_pcd(pcd, semantic_image,
                                                       intrinsics, T_lidar_to_camera)
            pcd_label = self.merge_color(pcd_label)
            
            self.map = self.update_map(self.map, pcd_in_range, pcd_label, T_lidar_to_map)

            local_map_rendered = self.render_local_map(self.map, T_ego_to_map, self.render_ego_centric)

            global_map_rendered = self.render_global_map(self.map_tile) if self.render_global_map_flag else None

        return self.map, local_map_rendered, global_map_rendered, pcd_in_range, pcd_label, None #, image_with_pcd
    

    def mapping_nuscenes(self, 
                pcd, 
                semantic_images,
                cam_names,
                intrinsics_,
                T_lidar_to_cameras, 
                T_lidar_to_map,
                T_ego_to_map):
        """
        Receives the semantic segmentation image, the pose of the vehicle, and the calibration of the camera,
        we will build a semantic point cloud, then project it into the 2D bird's eye view coordinates.

        Args:
            pcd: np.ndarray with shape (4, N), point cloud of type (x, y, z, intensity)
            semantic_image: np.ndarray with shape (H, W, C), 2D semantic image
            intrinsics: np.ndarray with shape (3, 3), camera intrinsic matrix
            T_lidar_to_camera: np.ndarray with shape (4, 4), the transformation from lidar frame to camera frame.
            T_lidar_to_map: np.ndarray with shape (4,4), the transformation matrix from lidar frame to map frame,
            T_ego_to_map: np.ndarray with shape (4,4), the transformation matrix from ego vehicle frame to map frame.
        """
        # Initialize the map
        self.check_map_origin(T_lidar_to_map)

        pcd_in_range_ = []
        pcd_label_ = []
        if self.depth_method == 'points_map' or self.depth_method == 'points_raw':
            for cam_name in cam_names:
                pcd_in_range, pcd_label = self.project_pcd(pcd, semantic_images[cam_name],
                                                        intrinsics_[cam_name], T_lidar_to_cameras[cam_name])
                pcd_label = self.merge_color(pcd_label)

                pcd_in_range_.append(pcd_in_range)
                pcd_label_.append(pcd_label)
                
            pcd_in_range = np.concatenate(pcd_in_range_, axis = 1)
            pcd_label = np.concatenate(pcd_label_, axis = 1)
            
            self.map = self.update_map(self.map, pcd_in_range, pcd_label, T_lidar_to_map)

            map_rendered = self.render_local_map(self.map, T_ego_to_map, self.render_ego_centric, filter=False)

        return self.map, map_rendered, pcd_in_range, pcd_label
    

    def mapping_argoverse(self, 
                pcd,
                intensity,
                av_loader,
                semantic_images,
                cam_names,
                pcd_timestamp,
                cam_timestamps,
                log_id, 
                T_lidar_to_map,
                T_ego_to_map,
                cam_crop_filter = None):
        """
        Receives the semantic segmentation image, the pose of the vehicle, and the calibration of the camera,
        we will build a semantic point cloud, then project it into the 2D bird's eye view coordinates.

        Args:
            pcd: np.ndarray with shape (4, N), point cloud of type (x, y, z, intensity)
            semantic_image: np.ndarray with shape (H, W, C), 2D semantic image
            intrinsics: np.ndarray with shape (3, 3), camera intrinsic matrix
            T_lidar_to_camera: np.ndarray with shape (4, 4), the transformation from lidar frame to camera frame.
            T_lidar_to_map: np.ndarray with shape (4,4), the transformation matrix from lidar frame to map frame,
            T_ego_to_map: np.ndarray with shape (4,4), the transformation matrix from ego vehicle frame to map frame.
        """
        # Initialize the map
        self.check_map_origin(T_lidar_to_map)

        pcd_in_range_ = []
        pcd_label_ = []
        if self.depth_method == 'points_map' or self.depth_method == 'points_raw':
            for cam_name in cam_names:
                pcd_in_range, pcd_label = self.project_pcd_argoverse(av_loader, semantic_images[cam_name], 
                                                                     pcd, intensity, cam_name, cam_timestamps[cam_name], 
                                                                     pcd_timestamp, log_id, cam_crop_filter[cam_name]['lidar_filter'])
                pcd_label = self.merge_color(pcd_label)
                pcd_in_range_.append(pcd_in_range)
                pcd_label_.append(pcd_label)
            
            pcd_in_range_ = np.concatenate(pcd_in_range_, axis = 1)
            pcd_label_ = np.concatenate(pcd_label_, axis = 1)
            
            self.map = self.update_map(self.map, pcd_in_range_, pcd_label_, T_lidar_to_map)

            map_rendered = self.render_local_map(self.map, T_ego_to_map, self.render_ego_centric)

        return self.map, map_rendered, pcd_in_range, pcd_label


    def check_map_origin(self, T_pcd_to_map):
        """ Update map origin based on current position. """
        offset = T_pcd_to_map[0:2, -1]
        new_map_origin = self.get_origin(offset)
        if self.map is None:
            self.init_map(new_map_origin)
        elif self.need_update(new_map_origin):
            self.update_map_origin(new_map_origin)
    

    def get_tile_origin(self, i, j, map_origin=None):
        if map_origin is None:
            map_origin = self.map_origin
        return (map_origin[0] + i * self.tile_size_meter, 
                map_origin[1] + j * self.tile_size_meter)


    def update_map_origin(self, new_map_origin):
        """ Update the local map with map tiles and save old tiles back."""
        # create new maps
        new_map = np.zeros((self.map_height, self.map_width, self.map_depth))
        new_map_rendered = np.zeros((self.map_height, self.map_width, 3))

        # loop over new map and old map to handle update
        for i in range(self.pad_total_num):
            for j in range(self.pad_total_num):

                tile_origin = self.get_tile_origin(i, j, new_map_origin)
                new_boundary = self.get_tile_boundary(i, j)
                # print('new_map_origin {} tile_origin {} new_boundary {}'.format(new_map_origin, tile_origin, new_boundary))

                if not self.tile_exists(tile_origin):
                    # For new tile, create tile but not write anything
                    # print('Tile ({}, {}) not exist'.format(i, j))
                    self.map_tile[tile_origin] = {
                        'boundary': new_boundary
                    }
                else:
                    # For existing tile, get the data
                    new_map_tile = self.get_map_with_boundary(new_map, new_boundary, copy=False)
                    new_map_rendered_tile = self.get_map_with_boundary(new_map_rendered, new_boundary, copy=False)
                    if self.tile_is_active(tile_origin):
                        # for active tile, get it from previous map
                        # print('Tile ({}, {}) active'.format(i, j))
                        old_boundary = self.map_tile[tile_origin]['boundary']
                        new_map_tile[:] = self.get_map(old_boundary)
                        new_map_rendered_tile[:] = self.get_map_rendered(old_boundary)
                    else:
                        # for not active tile, get it from map tile
                        # print('Tile ({}, {}) not active'.format(i, j))
                        new_map_tile[:] = np.array(self.map_tile[tile_origin]['map'], copy=True)
                        new_map_rendered_tile[:] = np.array(self.map_tile[tile_origin]['map_rendered'], copy=True)
                    self.map_tile[tile_origin]['boundary'] = new_boundary
                
                old_tile_origin = (
                    self.map_origin[0] + i * self.tile_size_meter,
                    self.map_origin[1] + j * self.tile_size_meter)

                if not self.tile_is_active(old_tile_origin, new_map_origin):
                    # if an old tile is no logner active, write it into map_tile
                    # print('Tile ({}, {}) deactivate'.format(i, j))
                    old_boundary = self.map_tile[old_tile_origin]['boundary']
                    self.map_tile[old_tile_origin] = {
                        'map': self.get_map(old_boundary),
                        'map_rendered': self.get_map_rendered(old_boundary),
                        'boundary': None
                    }
        
        self.map = new_map
        self.map_rendered = new_map_rendered
        self.map_origin = new_map_origin
        self.map_boundary = self.get_map_boundary()
    

    def get_map_boundary(self, map_origin=None):
        """ Return current map boundary in map frame. """
        if map_origin is None:
            map_origin = self.map_origin
        map_boundary = np.array([
            [map_origin[0], map_origin[0] + self.tile_size_meter * self.pad_total_num],
            [map_origin[1], map_origin[1] + self.tile_size_meter * self.pad_total_num]
        ])
        return map_boundary


    def get_map_with_boundary(self, map, boundary, copy=False):
        """ Helper function to simplify the code."""
        return np.array(map[boundary[0][0]:boundary[0][1], boundary[1][0]:boundary[1][1]], copy=copy)


    def get_map(self, boundary):
        return self.get_map_with_boundary(self.map, boundary, copy=True)


    def get_map_rendered(self, boundary):
        return self.get_map_with_boundary(self.map_rendered, boundary, copy=True)
    

    def get_tile_boundary(self, i, j):
        """ Given the tile index, return the tile boundary index"""
        idx_start_x = i * self.tile_size_d
        idx_end_x = idx_start_x + self.tile_size_d
        idx_start_y = j * self.tile_size_d
        idx_end_y = idx_start_y + self.tile_size_d
        boundary = [[idx_start_x, idx_end_x], [idx_start_y, idx_end_y]]
        return boundary


    def tile_exists(self, tile_origin):
        """ Return if a tile is in current map_tile. """
        return tile_origin in self.map_tile


    def tile_is_active(self, tile_origin, map_origin=None):
        """ Return if an existing map tile is active in the dynamic map. """
        if map_origin is None:
            map_origin = self.map_origin
        if not self.tile_exists(tile_origin):
            print("Error: expect tile exist when using tile is active.")
            exit()
        
        map_boundary = self.get_map_boundary(map_origin)
        is_active = (map_boundary[0][0] < tile_origin[0] + self.tile_size_meter / 2.0 < map_boundary[0][1] ) and \
            (map_boundary[1][0] < tile_origin[1] + self.tile_size_meter / 2.0 < map_boundary[1][1])
        
        return is_active


    def init_map(self, new_map_origin):
        self.map = np.zeros((self.map_height, self.map_width, self.map_depth))
        self.map_rendered = np.zeros((self.map_height, self.map_width, 3))
        self.map_origin = new_map_origin
        self.map_boundary = self.get_map_boundary()
        for i in range(self.pad_total_num):
            for j in range(self.pad_total_num):
                tile_origin = self.get_tile_origin(i, j)
                tile_boundary = self.get_tile_boundary(i, j)
                self.map_tile[tile_origin] = {
                        'boundary': tile_boundary
                }


    def get_origin(self, offset):
        map_origin = (offset // self.tile_size_meter - self.pad_num) * self.tile_size_meter
        return tuple(map_origin)


    def need_update(self, new_map_origin):
        if np.allclose(new_map_origin, self.map_origin):
            return False
        else:
            return True


    def project_pcd(self, pcd, image, intrinsics, T_lidar_to_camera, T_base_to_origin=None):
        """
        Extract labels of each point in the pcd from image
        Args:
            pcd: np.ndarray with shape (M, N), point cloud of size N with dimension M starts with [x, y, z]
            image: np.ndarray with shape (H, W, C), semantic image
            intrinsics: np.ndarray with shape (3, 3), camera intrinsic matrix
            T_lidar_to_camera: np.ndarray with shape (4, 4), the transformation from lidar to camera.

        Returns: Point cloud that are visible in the image, and their associated labels

        """
        if pcd is None:
            return
        
        pcd_homo = homogenize(pcd[0:3, :])
        pcd_camera = np.matmul(T_lidar_to_camera, pcd_homo)
        # visualize the projection to see 

        # Only use the points in the front.
        mask_positive = np.logical_and(0 < pcd_camera[2, :], pcd_camera[2, :] < self.pcd_range_max)
        # print("in front of camera:", np.sum(mask_positive))

        IXY = dehomogenize(np.matmul(intrinsics, dehomogenize(pcd_camera))).astype(np.int32)

        # Only select the points that project to the image
        mask = np.logical_and(np.logical_and(0 <= IXY[0, :], IXY[0, :] < image.shape[1]),
                              np.logical_and(0 <= IXY[1, :], IXY[1, :] < image.shape[0]))
        mask = np.logical_and(mask, mask_positive)
        # print("mean:", np.mean(IXY, axis=1), "image shape:", image.shape)

        masked_pcd = pcd[:, mask]
        image_idx = IXY[:, mask]
        label = image[image_idx[1, :], image_idx[0, :]].T

        return masked_pcd, label


    def project_pcd_argoverse(self, av_loader, segmented_image, pcd, intensity, cam_name, cam_timestamp, pcd_timestamp, log_id, cam_crop = None):
        IXY, points_cam, mask = av_loader.project_ego_to_img_motion_compensated(pcd,
                                                                                cam_name,
                                                                                cam_timestamp,
                                                                                pcd_timestamp,
                                                                                log_id)
        pcd = np.concatenate((pcd, intensity[..., np.newaxis]), axis = 1)
        masked_pcd = pcd[mask].T
        image_idx = IXY[mask].T.astype(np.int64)
        crop_valid_mask = np.logical_and(image_idx[1, :] > cam_crop[0], image_idx[1, :] < cam_crop[1])
        image_idx = image_idx[:, crop_valid_mask]
        masked_pcd = masked_pcd[:, crop_valid_mask]
        image_idx[1, :] = image_idx[1, :] - cam_crop[0]
        label = segmented_image[image_idx[1, :], image_idx[0, :]].T

        return masked_pcd, label
    

    def merge_color(self, pcd_label):
        # print(pcd_label.shape)
        for color_src, color_dest in zip(self.color_remap_source, self.color_remap_dest):
            pcd_mask = np.all(pcd_label == color_src.reshape(3,1), axis=0)
            pcd_label[:,pcd_mask] = color_dest.reshape(3,1)
        return pcd_label
    
    
    def update_map(self, map, pcd, label, T_pcd_to_map=None):
        """
        Project the semantic point cloud on the BEV map

        Args:
            map: np.ndarray with shape (H, W, C). H is the height, W is the width, and C is the semantic class.
            pcd: np.ndarray with shape (4, N). N is the number of points. The point cloud
            label: np.ndarray with shape (3, N). N is the number of points. The RGB label of each point cloud.

        Returns:
            Updated map
        """
        normal = np.array([[0.0, 0.0, 1.0]]).T  # The normal of the z axis
        pcd_origin_offset = np.array([[self.map_boundary[0][0]], [self.map_boundary[1][0]], [0.0]]) # pcd origin with respect to map origin
        
        # compute point cloud distance to sensor, only valid in sensor frame
        distance = np.linalg.norm(pcd[0:3], axis=0) 

        # If pcd not in map frame, transform to map
        if T_pcd_to_map is None:
            pcd_map = pcd[0:3]
        else:
            pcd_map = dehomogenize(np.matmul(T_pcd_to_map, homogenize(pcd[0:3])))
        
        pcd_local = pcd_map - pcd_origin_offset
        pcd_flatten = pcd_local - np.matmul(normal, np.matmul(normal.T, pcd_local))
        # Discretize point cloud into grid, Note that here we are basically doing the nearest neighbor search
        # pcd_pixel = ((pcd_on_map[0:2, :] - np.array([[self.map_boundary[0][0]], [self.map_boundary[1][0]]]))
        #             / self.resolution).astype(np.int32)
        pcd_pixel = (pcd_flatten[0:2, :] / self.resolution).astype(np.int32)
        on_grid_mask = np.logical_and(np.logical_and(0 <= pcd_pixel[0, :], pcd_pixel[0, :] < self.map_height),
                                      np.logical_and(0 <= pcd_pixel[1, :], pcd_pixel[1, :] < self.map_width))
        
        # Update corresponding labels
        for i, label_name in enumerate(self.label_names):
            # Code explanation:
            # We first do an elementwise comparison
            # a = (label == self.label_colors[i].reshape(3, 1))
            # Then we do a logical AND among the rows of a, represented by *a.
            idx = np.logical_and(*(label == self.label_colors[i].reshape(3, 1)))
            idx_mask = np.logical_and(idx, on_grid_mask)
            
            # Update the local map with Bayes update rule
            # map[pcd_pixel[0, idx_mask], pcd_pixel[1, idx_mask], :] has shape (n, num_classes)
            if self.use_distance:
                # print('pcd:', pcd[0:3, idx_mask][:,:5])
                weights =  np.log(1 / distance[idx_mask])
                # print('weights:', weights[:5])
                map[pcd_pixel[0, idx_mask], pcd_pixel[1, idx_mask], :] += \
                    self.confusion_matrix[:, i].reshape(1, -1) - weights.reshape(-1,1)
            else:
                map[pcd_pixel[0, idx_mask], pcd_pixel[1, idx_mask], :] += self.confusion_matrix[:, i].reshape(1, -1)

            # print(f'mean:{np.mean(map)}, max:{np.max(map)}, min:{np.min(map)}')
            # LiDAR intensity augmentation
            if not self.use_pcd_intensity: continue

            # For all the points that have been classified as land, we augment its count by looking at its intensity
            # print(label_name)
            if label_name == "lane":
                intensity_mask = np.logical_or(pcd[3] < 2, pcd[3] > 14)  # These thresholds are found by experiment.
                intensity_mask = np.logical_and(intensity_mask, idx_mask)

                # 2 is an experimental number which we think is good enough to connect the lane on the side.
                # Too large the lane will be widen, too small the lane will be fragmented.
                map[pcd_pixel[0, intensity_mask], pcd_pixel[1, intensity_mask], i] += 2
                # map[pcd_pixel[0, idx_mask], pcd_pixel[1, idx_mask], i] += 100

                # For the region where there is no intensity by our network detected as lane, we will degrade its
                # threshold
                # non_intensity_mask = np.logical_and(~intensity_mask, idx_mask)
                # map[pcd_pixel[1, non_intensity_mask], pcd_pixel[0, non_intensity_mask], i] -= 0.5

        return map


    def render_local_map(self, map, T_ego_to_map=None, ego_centric=False, filter=True):
        """ Render the local map with a small number of map tiles"""
        if filter:
            map_filtered = apply_filter(map)
        else:
            map_filtered = np.array(map)
        self.map_rendered = render_bev_map(map_filtered, self.label_colors)
        
        if T_ego_to_map is not None and ego_centric:
            # crop an ego centric view from map
            r = R.from_matrix(T_ego_to_map[0:3,0:3])
            euler_angles = r.as_euler('zxy', degrees=True)
            
            t = T_ego_to_map[0:2, -1]
            t_xy = np.round((t - np.array(self.map_origin)) / self.resolution).astype(np.int32)
            
            #rotate around ego position
            center = (t_xy[1].item(), t_xy[0].item())
            (h, w) = self.map_rendered.shape[:2]
            M = cv2.getRotationMatrix2D(center, -euler_angles[0], 1.0)
            map_rendered_warpped = cv2.warpAffine(self.map_rendered, M, (w, h), cv2.INTER_NEAREST)
            
            map_rendered_ego_centric = cv2.flip(map_rendered_warpped[
                center[1] + self.ego_x_min_d: center[1] + self.ego_x_max_d,
                center[0] + self.ego_y_min_d: center[0] + self.ego_y_max_d], -1)

            return map_rendered_ego_centric
        return self.map_rendered


    def render_global_map(self, global_map):
        origins = np.array([key for key in global_map.keys()])
        x_min, y_min = np.min(origins, axis=0)
        x_max, y_max = np.max(origins, axis=0)
        x_tile_num = int((x_max - x_min) / self.tile_size_meter) + 1
        y_tile_num = int((y_max - y_min) / self.tile_size_meter) + 1
        x_map_size = x_tile_num * self.tile_size_d
        y_map_size = y_tile_num * self.tile_size_d
        global_map_rendered = np.zeros((x_map_size, y_map_size, 3), dtype=np.uint8)
        
        # active map tiles
        x_min_local, y_min_local = self.map_origin
        x_origin_d = int((x_min_local - x_min) / self.tile_size_meter) * self.tile_size_d
        y_origin_d = int((y_min_local - y_min) / self.tile_size_meter) * self.tile_size_d
        global_map_rendered[
            x_origin_d:x_origin_d+self.tile_size_d*self.pad_total_num, 
            y_origin_d:y_origin_d+self.tile_size_d*self.pad_total_num] = self.map_rendered
        
        # inactive map tiles
        for key in global_map.keys():
            if 'map_rendered' not in global_map[key]:
                continue
            map_tile_rendered = global_map[key]['map_rendered']
            x_origin, y_origin = key
            x_origin_d = int((x_origin - x_min) / self.tile_size_meter) * self.tile_size_d
            y_origin_d = int((y_origin - y_min) / self.tile_size_meter) * self.tile_size_d
            global_map_rendered[x_origin_d:x_origin_d+self.tile_size_d, y_origin_d:y_origin_d+self.tile_size_d] = map_tile_rendered
        return global_map_rendered

# Test
def main():
    pass


# main
if __name__ == '__main__':
    main()