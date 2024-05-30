# The basic configuration system
import os.path as osp
import sys

# Add src directory into the path
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "../")))

from semantic_mapping.src.node_config.base_cfg import _C

from yacs.config import CfgNode as CN



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
_C.TASK_NAME = "semvecnet_dev"

# _C.MAPPING.PCD.RANGE_MAX = 30.0
_C.MAPPING.PCD.RANGE_MAX = 100.0

# The load path of the confusion matrix
_C.MAPPING.CONFUSION_MTX.LOAD_PATH = "/home/hzhang/data/hrnet/hrnet_cfn_1999.npy"

# ego map range [x_min, x_max, y_min, y_max] 
# _C.MAPPING.RANGE = [-30.0, 30.0, -30.0, 30.0] 
_C.MAPPING.RANGE = [-30.0, 30.0, -15.0, 15.0] 

_C.VISION_SEM_SEG.DATASET_CONFIG = "/home/hzhang/Documents/projects/noeticws/src/SemVecNet/semantic_mapping/config/config_65.json"
_C.VISION_SEM_SEG.ENGINE_FILE = "/home/hzhang/Documents/projects/noeticws/src/online_semantic_mapping/src/hrnet/assets/seg_weights/hrnet-avl-map.engine"
_C.VISION_SEM_SEG.DUMMY_IMAGE_PATH = "/home/hzhang/Documents/projects/noeticws/src/online_semantic_mapping/src/hrnet/assets/seg_weights/1682354962.572154522.jpg"

_C.NUSCENES_VERSION = "v1.0-trainval"
_C.NUSCENES_PATH = "/home/hzhang/data/nuScenes"