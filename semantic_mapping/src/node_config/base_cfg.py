# The basic configuration system
import os.path as osp
import sys

# Add src directory into the path
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "../")))

from yacs.config import CfgNode as CN

_C = CN()


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()


# Usually I will use UPPER CASE for non-parametric variables, and lower case for parametric variables because it can be
# directly pass into the function as key-value pairs.

# --------------------------------------------------------------------------- #
# General Configuration
# --------------------------------------------------------------------------- #
# We will create a sub-folder with this name in the output directory
# _C.TASK_NAME = "vanilla_confusion_matrix"
_C.TASK_NAME = "semvecnet"

# '@' here means the root directory of the project
_C.OUTPUT_DIR = "@/outputs"
_C.MAP_DIR = ""

# If the time stamp reaches this threshold, we will stop generating the map.
# Usually, our start time frame is 390. If you want a shorter test time, you can set it to 1581541270, which is about
# 20 seconds.
#_C.TEST_END_TIME = 1604445190 
#_C.TEST_END_TIME = 1604445200 
# _C.TEST_END_TIME = 1604445459 
# _C.TEST_END_TIME = 1581540870 # 30 second testing
# _C.TEST_END_TIME = 1581541250 # half way testing
_C.TEST_END_TIME = 1581541450
# _C.TEST_END_TIME = 1581541631
#_C.TEST_END_TIME = 1602190420
_C.TEST_END_TIME = None # online mapping

# Ground truth semantic map label directory
_C.GROUND_TRUTH_DIR = ""

# Set the random seed of the network for reproducibility
_C.RNG_SEED = -1

# The associate index of each label in the semantic segmentation network
# _C.LABELS = [13, 8, 24, 30, 15 ] # with vegetation labels
_C.LABELS = [13, 8, 24, 15 ] # take out vegetation labels
_C.SRC_INDICES = [23, 7]
_C.DST_INDICES = [8, 13]
# The name of the label
# _C.LABELS_NAMES = ["road", "crosswalk", "lane", "vegetation", "sidewalk"]
_C.LABELS_NAMES = ["road", "crosswalk", "lane", "sidewalk"]
# The RGB color of each label. We will use this to identify the label of each RGB pixel
_C.LABEL_COLORS = [
    [128, 64, 128],  # road
    [140, 140, 200],  # crosswalk
    [255, 255, 255],  # lane
    # [107, 142, 35],  # vegetation
    [244, 35, 232],  # sidewalk
]
# The color from source will be coverted to color in dest to merge labels of similar kind.
_C.COLOR_REMAP_SOURCE = [
    [200, 128, 128], # zebra line
    # [196, 196, 196], # curb
    [128, 64, 255],  # bike lane
    # [170, 170, 170], # curb cut
    # [96, 96, 96],    # pedestrian walk
]
_C.COLOR_REMAP_DEST = [
    [140, 140, 200], # crosswalk
    # [244, 35, 232],  # sidewalk
    [128, 64, 128], # road
    # [244, 35, 232],  # sidewalk
    # [244, 35, 232],  # sidewalk
]   

# --------------------------------------------------------------------------- #
# Mapping Configuration
# --------------------------------------------------------------------------- #
_C.MAPPING = CN()

# The resolution of the occupancy grid in meters
_C.MAPPING.RESOLUTION = 0.2
# The boundary of the occupancy grid, in meters. The format of the boundary is [[xmin, xmax], [ymin, ymax]]
# _C.MAPPING.BOUNDARY = [[100, 300], [800, 1000]]
#_C.MAPPING.BOUNDARY = [[0, 1000], [0, 1400]]
# mail-route map
_C.MAPPING.BOUNDARY = [[-1369, 149], [-563, 874]]
# summer2020-map1
# _C.MAPPING.BOUNDARY = [[-637.05267334, 837.194641113], [-1365.04785156, 117.317863464]]
# summer2020-map2
#_C.MAPPING.BOUNDARY = [[-267.616485596, 242.025421143], [-696.055175781, 126.109397888]]
# summer2020-map3
#_C.MAPPING.BOUNDARY = [[-118.229263306, 680.575927734], [-81.1667251587, 392.865081787]]
#_C.MAPPING.BOUNDARY = [[-119, 681], [-82, 393]]

# Extend previous map (raw map in .npy format)
_C.MAPPING.PREV_MAP = ""
# _C.MAPPING.PREV_MAP = "/home/dfpazr/Documents/CogRob/avl/TritonNet/iros_psm_ws/src/vision_semantic_segmentation/outputs/cfn_mtx_with_intensity/version_23/raw_map.npy"

#_C.MAPPING.BOUNDARY = [[0, 1400], [0, 1400]]
# This variable defines the way how we estimate the depth from the image. If use "points_map", then we are using the
# offline point cloud map. If use the points_raw", then we are using the the online point cloud map, i.e. the output
# from the LiDAR per frame.
_C.MAPPING.DEPTH_METHOD = "points_raw"
# _C.MAPPING.DEPTH_METHOD = "points_map"

# Point cloud setting
_C.MAPPING.PCD = CN()
# If True, use the point cloud intensity data to augment our semantic BEV estimation
_C.MAPPING.PCD.USE_INTENSITY = True
# If True, weight probability by distance
_C.MAPPING.PCD.USE_DISTANCE = True
# If True, weight probability by angular velocity
_C.MAPPING.PCD.USE_ANGULAR_VELOCITY = False
_C.MAPPING.PCD.RANGE_MAX = 30.0
# _C.MAPPING.PCD.RANGE_MAX = 100.0
# _C.MAPPING.PCD.RANGE_MAX = 15.0

_C.MAPPING.CONFUSION_MTX = CN()
# The load path of the confusion matrix
_C.MAPPING.CONFUSION_MTX.LOAD_PATH = ""
# _C.MAPPING.CONFUSION_MTX.LOAD_PATH = ""
_C.MAPPING.REWEIGHT_FACTOR = 1.3
# The store and load path of deterministic input to the mapping process
_C.MAPPING.INPUT_DIR = ""
# If round to close or round down
_C.MAPPING.ROUND_CLOSE = True

# 
_C.MAPPING.RENDER_EGO_CENTRIC = True
# ego map range [x_min, x_max, y_min, y_max] 
_C.MAPPING.RANGE = [-30.0, 30.0, -30.0, 30.0] 
#
_C.MAPPING.TILE_SIZE_METER = 35.5
# _C.MAPPING.TILE_SIZE_METER = 60.0
# number of tile padding in each direction
_C.MAPPING.TILE_PAD_NUMBER = 1

# --------------------------------------------------------------------------- #
# Vision Semantic Segmentation Configuration
# --------------------------------------------------------------------------- #
_C.VISION_SEM_SEG = CN()

# Determine the scale of the input image, from 0 to 1.
_C.VISION_SEM_SEG.IMAGE_SCALE = 0.5

# --------------------------------------------------------------------------- #
# Vision Semantic Segmentation Configuration
# --------------------------------------------------------------------------- #
_C.VISION_SEM_SEG = CN()

# Determine the scale of the input image, from 0 to 1.
_C.VISION_SEM_SEG.IMAGE_SCALE = 0.5
_C.VISION_SEM_SEG.TRAIN_DATASET = "Mapillary"
_C.VISION_SEM_SEG.DATASET_CONFIG = ""
