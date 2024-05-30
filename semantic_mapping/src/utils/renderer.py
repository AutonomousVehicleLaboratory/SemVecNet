""" Render color map

Author: Henry Zhang
Date:March 01, 2020
"""

# module
import numpy as np

# The remote is using scipy==0.17.0 so we cannot do from scipy.special.logsumexp
# In local testing, we can use scipy.special.logsumexp
try:
    from scipy.misc import logsumexp
except:
    from scipy.special import logsumexp

# parameters

label_colors = np.array([
    [128, 64, 128],  # road
    [140, 140, 200],  # crosswalk
    [255, 255, 255],  # lane
    [107, 142, 35],  # vegetation
    [244, 35, 232]  # sidewalk
])


# classes


# functions
def render_bev_map(map, label_colors):
    """
    Render the Bird's eye view semantic map, the color of each pixel is picked by the max number of points
    Args:
        map: np.ndarray (W, H, C)
        label_colors: the RGB color of each label

    Returns:

    """
    # Sanity check
    assert len(map.shape) == 3
    for c in label_colors:
        if len(c) != 3:
            raise ValueError("Color should be an RGB value.")

    width, height, num_channels = map.shape
    if num_channels != len(label_colors):
        raise ValueError("Each channel should have a color!")

    colored_map = np.zeros((width, height, 3)).astype(np.uint8)
    map_argmax = np.argmax(map, axis=2)
    for i in range(num_channels):
        colored_map[map_argmax == i] = label_colors[i]

    map_sum = np.sum(map, axis=2)  # get all zero mask
    colored_map[map_sum == 0] = [0, 0, 0]  # recover all zero positions
    return colored_map


def fill_black(img):
    """ Fill the black area according to the labels in its 3*3 neighbor.
    the approach is based on a priority list
    this approach will expand the prioritized labels
    """
    priority_list = [0, 3, 4, 2, 1]  # from low to high priority
    xmax, ymax = img.shape[0], img.shape[1]

    # constructing 3*3 area for faster option
    img_stacked = np.zeros(img.shape)
    img_stacked[1:-1, 1:-1] = np.vstack([img[1:xmax - 1, 1:ymax - 1, 0].reshape([-1, xmax - 2, ymax - 2]),
                             img[0:xmax - 2, 1:ymax - 1, 0].reshape([-1, xmax - 2, ymax - 2]),
                             img[2:xmax, 1:ymax - 1, 0].reshape([-1, xmax - 2, ymax - 2]),
                             img[1:xmax - 1, 0:ymax - 2, 0].reshape([-1, xmax - 2, ymax - 2]),
                             img[0:xmax - 2, 0:ymax - 2, 0].reshape([-1, xmax - 2, ymax - 2]),
                             img[2:xmax, 0:ymax - 2, 0].reshape([-1, xmax - 2, ymax - 2]),
                             img[1:xmax - 1, 2:ymax, 0].reshape([-1, xmax - 2, ymax - 2]),
                             img[0:xmax - 2, 2:ymax, 0].reshape([-1, xmax - 2, ymax - 2]),
                             img[2:xmax, 2:ymax, 0].reshape([-1, xmax - 2, ymax - 2])])

    mask_dict = {}
    for i in range(len(label_colors)):
        mask_dict[i] = np.any(img_stacked == label_colors[i, 0], axis=0)

    img_out = np.zeros((xmax, ymax), dtype=np.uint8)

    # get colors
    for label in priority_list:
        img_out[mask_dict[label]] = label_colors[label, 0]

    # img_out[img[:,:, 0]!=0] = img[:,:, 0][img[:,:, 0]!=0]
    # expand to three channels
    img_out = np.concatenate([img_out.reshape([xmax, ymax, 1]),
                              img_out.reshape([xmax, ymax, 1]),
                              img_out.reshape([xmax, ymax, 1])], axis=2)
    img_out = resume_color(img_out)

    return img_out


def resume_color(img):
    for i in range(len(label_colors)):
        mask = img[:, :, 0] == label_colors[i, 0]
        img[mask] = label_colors[i]
    return img


def fill_black_for_loop(img):
    """ fill the black area with the most popular label within its 3*3 neighbor """
    from scipy.stats import mode
    img_filled = np.zeros(img.shape, dtype=np.uint8)
    xmax, ymax = img.shape[0], img.shape[1]
    print(xmax, ymax)
    for x in range(1, xmax - 1):
        if x % 100 == 0:
            print(x)
        for y in range(1, ymax - 1):
            color_list = []
            for i in range(x - 1, x + 2):
                for j in range(y - 1, y + 2):
                    if img[i, j, 0] != 0:
                        color_list.append(img[i, j, 0])
            xy_mode, count = mode(color_list)
            img_filled[x, y] = xy_mode if xy_mode.shape[0] != 0 else 0

    img_filled = resume_color(img_filled)

    return img_filled


def render_bev_map_with_thresholds(map, label_colors, priority=None, thresholds=[0.01, 0.01, 0.01, 0.01, 0.01]):
    """
    Render the map by whether the labels have probability higher than the specified threshold. Only when the
    probability is higher than the threshold will be rendered.

    Args:
        map: np.ndarray (W, H, C). The count value of each layer in the map.
        label_colors: corresponding color for each channel
        priority: priority of each label ordered from low to high, higher will overwrite lower colors
        thresholds: specify the threshold for each category, default to minimum requirement
    """
    assert len(map.shape) == 3
    for c in label_colors:
        if len(c) != 3:
            raise ValueError("Color should be an RGB value.")

    width, height, num_channels = map.shape
    if num_channels != len(label_colors):
        raise ValueError("Each channel should have a color.")
    if priority is not None and num_channels != len(priority):
        raise ValueError("Each channel should have a priority.")

    if priority is None:
        priority = np.arange(num_channels)

    # Normalize the map and convert it into probability
    channel_sum = np.sum(map, axis=2, keepdims=True)
    map_normalized = np.divide(map, channel_sum, out=np.zeros_like(map), where=(channel_sum != 0))

    # Reorder the map and label_colors with priority
    map_normalized = map_normalized[:, :, priority]
    label_colors = label_colors[priority]

    # Identify the explored region by checking the sum of the map, we will only update these area.
    known_region = (np.sum(map, axis=2) != 0)

    colored_map = np.zeros((width, height, 3)).astype(np.uint8)
    for i, p in enumerate(priority):
        mask = np.logical_and(map_normalized[:, :, i] >= thresholds[i], known_region)
        colored_map[mask] = label_colors[i]

    return colored_map


def apply_filter(src):
    import cv2
    ddepth = -1

    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    kernel /= (kernel_size * kernel_size)

    # ratio = 1/4.0
    # kernel = (1-ratio) / 8.0 * np.ones((kernel_size, kernel_size), dtype=np.float32)
    # kernel[int(kernel_size/2),int(kernel_size/2)] = ratio

    dst = cv2.filter2D(src, ddepth, kernel)

    return dst


def fill_edge(color_map):
    color_map[[0, -1], :, :] = 250
    color_map[:, [0, -1], :] = 250
    color_map[0:5, 0:5] = 254
    return color_map


def exec_render_portion():
    import cv2
    priority = [3, 4, 0, 2, 1]
    map_local = np.load('/home/henry/Pictures/map_local.npy')

    # map_local = apply_filter(map_local)

    color_map = render_bev_map_with_thresholds(map_local, label_colors, priority,
                                               thresholds=[0.1, 0.1, 0.5, 0.20, 0.05])

    # color_map = fill_black(color_map)

    # color_map = fill_edge(color_map)
    cv2.imwrite('/home/henry/Pictures/global_map_new.png', color_map)


# main
def main():
    # test_filter()
    # test_map_layer()
    # test_separate_map()
    # test_render_portion()
    # exec_render_portion()
    pass


if __name__ == "__main__":
    main()
