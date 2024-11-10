import json
import numpy as np
import os.path as osp 

# from deeplab_v3_plus.data.dataset.mapillary import MapillaryVistas
from matplotlib import pyplot as plt
# from src.utils.utils import profile


def get_labels(data_dir):
    """
    Get the label of the data
    Args:
        data_dir (str): fetch the label from the data_dir
    """
    config_file = data_dir
    with open(config_file) as f:
        config = json.load(f)
    return config["labels"]

labels_np = np.array([label["color"] for label_id, label in enumerate(get_labels("./semantic_mapping/config/config_65.json"))])


def display_labels(data_dir, blocking=False):
    """
    Display the label

    Modified from: https://matplotlib.org/2.0.2/examples/color/named_colors.html

    Args:
        data_dir (str): fetch the label from the data_dir
        blocking (bool): True if the plot blocks the process
    """
    labels = get_labels(data_dir)

    num_col = 3
    num_row = len(labels) // num_col
    fig, ax = plt.subplots(figsize=(8, 8))

    # Get figure's height and width
    X, Y = fig.get_dpi() * fig.get_size_inches()
    height = Y / (num_row + 1)
    width = X / num_col
    for i, label in enumerate(labels):
        name = label['readable']
        color = np.array(label["color"])
        # Normalize color to 0-1 range because ax.hlines() requires it
        color = color / 255

        col = i % num_col
        row = i // num_col
        y = Y - (row * height) - height

        xi_line = width * (col + 0.05)
        xf_line = width * (col + 0.25)
        xi_text = width * (col + 0.3)
        ax.text(xi_text, y, name, fontsize=(height * 0.3), horizontalalignment='left', verticalalignment='center')

        ax.hlines(y + height * 0.1, xi_line, xf_line, color=color, linewidth=(height * 0.6))

    ax.set_xlim(0, X)
    ax.set_ylim(0, Y)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)

    if blocking:
        plt.show()
    else:
        plt.draw()
        plt.pause(0.001)


def apply_color_map(label_array, labels):
    """Apply color map for Mapillary Vistas"""
    if len(label_array.shape) == 2:
        batch_size = None
        height, width = label_array.shape
    elif len(label_array.shape) == 3:
        batch_size, height, width = label_array.shape
    else:
        raise NotImplementedError

    if batch_size:
        color_array = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
    else:
        color_array = np.zeros((height, width, 3), dtype=np.uint8)
    


    # for label_id, label in enumerate(labels):
    #     # set all pixels with the current label to the color of the current label
    #     color_array[label_array == label_id] = label["color"]
    color_array = labels_np[label_array]

    return color_array