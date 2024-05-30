""" Utility functions

Author: Henry Zhang
Date:February 12, 2020
"""

import cv2
import numpy as np
import cProfile
import io
import pstats

from matplotlib import pyplot as plt


def profile(function):
    """ A decorator that uses cProfile to profile a function """

    def inner(*args, **argv):
        pr = cProfile.Profile()
        pr.enable()
        retval = function(*args, **argv)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr).sort_stats(sortby)
        ps.print_stats(0.1)
        print(s.getvalue())
        return retval

    return inner


def homogenize(x):
    # converts points from inhomogeneous to homogeneous coordinates
    return np.vstack((x, np.ones((1, x.shape[1]))))


def dehomogenize(x):
    # converts points from homogeneous to inhomogeneous coordinates
    return x[:-1] / x[-1]


from scipy.linalg import logm, expm


# Note that np.sinc is different than defined in class
def sinc(x):
    # Returns a scalar valued sinc value
    """your code here"""
    if x == 0:
        y = 1
    else:
        y = np.sin(x) / x

    return y


def differentiate_sinc(x):
    if x == 0:
        return 0
    else:
        return np.cos(x) / x - np.sin(x) / (x ** 2)


def skew(w):
    # Returns the skew-symmetrix represenation of a vector
    """your code here"""
    w = w.reshape([3, 1])
    w_skew = np.array([[0., -w[2, 0], w[1, 0]],
                       [w[2, 0], 0., -w[0, 0]],
                       [-w[1, 0], w[0, 0], 0.]])

    return w_skew


def de_skew(w_skew):
    w = np.array([[-w_skew[1, 2], w_skew[0, 2], -w_skew[0, 1]]]).T
    return w


def singularity_normalization(w):
    """ w has a singularity at 2 pi, check every time change w """
    theta = np.linalg.norm(w)
    if theta > np.pi:
        w = (1 - 2 * np.pi / theta * np.ceil((theta - np.pi) / (2 * np.pi))) * w
    return w


def parameterize_rotation(R):
    # Parameterizes rotation matrix into its axis-angle representation
    """your code here"""
    # lecture implementation
    U, D, VT = np.linalg.svd(R - np.eye(R.shape[0]))
    v = VT.T[:, -1::]
    v_hat = np.array([[R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]]).T
    theta_sin = np.matmul(v.T, v_hat) / 2.
    theta_cos = (np.trace(R) - 1.) / 2.
    theta = np.arctan2(theta_sin, theta_cos).item()
    w = theta * v / np.linalg.norm(v)

    # scipy implementation
    # w_skew_2 = logm(R)
    # w_2 = DeSkew(w_skew_2)
    # theta_2 = np.linalg.norm(w_2)
    w = singularity_normalization(w)
    theta = np.linalg.norm(w)

    if theta < 1e-7:
        w = v_hat / 2.0

    theta = np.linalg.norm(w)
    return w, theta


def deparameterize_rotation(w):
    # Deparameterizes to get rotation matrix
    """your code here"""
    w = w.reshape([3, 1])
    w_skew = skew(w)
    theta = np.linalg.norm(w)
    if theta < 1e-7:
        R = np.eye(w.shape[0]) + w_skew
    else:
        R = np.cos(theta) * np.eye(w.shape[0]) + \
            sinc(theta) * w_skew + (1 - np.cos(theta)) / theta ** 2 * np.matmul(w, w.T)

    return R


def jacobian_vector_norm(v):
    assert (v.shape[1] == 1)
    J = 1. / np.linalg.norm(v) * v.T
    return J


def right_null(A):
    U, S, VT = np.linalg.svd(A)
    if S[-1] < 1e-5:
        return VT.T[:, -1::]
    else:
        print("right null space not exists")
        return None


def show_image_list(image_list, delay=0, size=None):
    if len(image_list) == 0:
        return
    elif len(image_list) == 1:
        cv2.imshow("image", image_list[0])
        cv2.waitKey(delay)
    else:
        reshaped_list = []
        if size is None:
            min_shape_y, min_shape_x = image_list[0].shape
            for image in image_list:
                if image.shape[0] < min_shape_y:
                    min_shape_y = image.shape[0]
                if image.shape[1] < min_shape_x:
                    min_shape_x = image.shape[1]
            for image in image_list:
                if image.shape[0] != min_shape_y or image.shape[1] != min_shape_x:
                    reshaped_image = cv2.resize(image, (min_shape_x, min_shape_y), interpolation=cv2.INTER_NEAREST)
                    reshaped_list.append(reshaped_image)
                else:
                    reshaped_list.append(image)
        else:
            for image in image_list:
                if image.shape[0] != size[0] or image.shape[1] != size[1]:
                    reshaped_image = cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
                    reshaped_list.append(reshaped_image)
                else:
                    reshaped_list.append(image)

        channel_fixed = []
        for image in reshaped_list:
            if len(image.shape) == 2:
                fixed_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                channel_fixed.append(fixed_image)
            else:
                channel_fixed.append(image)

        concatenated = np.concatenate(channel_fixed, axis=1)
        cv2.imshow("concatenated", concatenated)
        cv2.waitKey(delay)


def get_rotation_from_angle_2d(angle):
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    return R


def test_arg_max():
    mat = np.array([
        [
            [1, 2, 3],
            [3, 2, 1],
            [2, 3, 1]
        ],
        [
            [3, 3, 1],
            [5, 4, 2],
            [7, 2, 9]
        ]
    ])
    print(mat.shape)
    mat_argmax = np.argmax(mat, axis=2)
    print(mat_argmax)
    mat_new = np.zeros((2, 3))
    print(mat_new)
    mat_new[mat_argmax == 0] = 77
    print(mat_new)

    
def get_color(v, v_max):
    """ Covert a scalar array to RGB array 
    Params:
        v: a numpy array of scalars
        v_max: the maximum value of mapping range
    """
    inc = 6.0 / v_max
    x = v * inc
    rgb = np.zeros((3, v.shape[0]))
    rgb[0,np.logical_or(np.logical_and(0 <= x, x <= 1),np.logical_and(5 <= x, x <= 6))] = 1.0
    rgb[0,np.logical_and(4 <= x, x <= 5)] = x[np.logical_and(4 <= x, x <= 5)] - 4
    rgb[0,np.logical_and(1 <= x, x <= 2)] = 1.0 - (x[np.logical_and(1 <= x, x <= 2)] - 1)
    rgb[1,np.logical_and(1 <= x, x <= 3)] = 1.0
    rgb[1,np.logical_and(0 <= x, x <= 1)] = x[np.logical_and(0 <= x, x <= 1)] - 0
    rgb[1,np.logical_and(3 <= x, x <= 4)] = 1.0 - (x[np.logical_and(3 <= x, x <= 4)] - 3)
    rgb[2,np.logical_and(3 <= x, x <= 5)] = 1.0
    rgb[2,np.logical_and(2 <= x, x <= 3)] = x[np.logical_and(2 <= x, x <= 3)] - 2
    rgb[2,np.logical_and(5 <= x, x <= 6)] = 1.0 - (x[np.logical_and(5 <= x, x <= 6)] - 5)
    rgb = (rgb * 255).astype(np.uint8)
    return rgb

def color_pcd_by_distance(pcd, pcd_range_x_max = 3.0, pcd_range_x_min = 1.0):
    # Only use the points in the front.
    mask_positive = np.logical_and(pcd_range_x_min < pcd[0, :], pcd[0, :] < pcd_range_x_max)
    pcd_in_distance = pcd[:, mask_positive]
    # pcd_color = np.ones((pcd_in_distance.shape[0], 3)) * 255
    distance = pcd_in_distance[0, :]
    distance_normalized = (distance - pcd_range_x_min) / (np.max(distance) - pcd_range_x_min)
    # pcd_color[:,0] = pcd_color[:,0] * distance_normalized
    # pcd_color[:,1] = pcd_color[:,1] * (1 - distance_normalized)
    # pcd_color[:,2] = pcd_color[:,2] * distance_normalized # (1 - distance_normalized)
    pcd_color = get_color(distance_normalized, 1.0)
    pcd_colored = np.concatenate([pcd_in_distance, pcd_color], axis=0)
    return pcd_colored


def project_colored_pcd_on_image(image, pcd_colored, intrinsics, T_lidar_to_camera, k = 2, pcd_range_max=1000):
    # pcd = pcd_colored[0:3,:]
    # pcd_velodyne = homogenize(pcd[0:3, :])
    # IXY = dehomogenize(np.matmul(P, pcd_velodyne)).astype(np.int32)
    
    pcd_homo = homogenize(pcd_colored[0:3, :])
    pcd_camera = np.matmul(T_lidar_to_camera, pcd_homo)
    # visualize the projection to see 

    # Only use the points in the front.
    mask_positive = np.logical_and(0 < pcd_camera[2, :], pcd_camera[2, :] < pcd_range_max)
    # print("in front of camera:", np.sum(mask_positive))

    IXY = dehomogenize(np.matmul(intrinsics, dehomogenize(pcd_camera))).astype(np.int32)

    # Only select the points that project to the image
    mask = np.logical_and(np.logical_and(k <= IXY[0, :], IXY[0, :] < image.shape[1]-k),
                            np.logical_and(k <= IXY[1, :], IXY[1, :] < image.shape[0]-k))
    
    for i in range(-k, k, 1):
        for j in range(-k, k, 1):
            image[IXY[1, mask]+i, IXY[0,mask]+j] = pcd_colored[3:6, mask].T
    return image


def test_queue():
    import queue
    q = queue.Queue()
    q.put(1)
    print(q)
    q.put(3)
    print(q.get())


def test_crop():
    img_path = "/home/henry/Documents/projects/pylidarmot/src/vision_semantic_segmentation/outputs/distance_new/version_3/global_map_input_list_0.png"
    img_path = "/home/henry/Documents/projects/pylidarmot/src/vision_semantic_segmentation/outputs/points_raw/version_1/global_map.png"
    img = cv2.imread(img_path)
    clipped_img = np.flip(img[805:885, 5350:5700], axis=1)
    clipped_img = np.flip(clipped_img, axis=0)
    plt.figure()
    plt.imshow(clipped_img)
    plt.show()
    cv2.imwrite("/home/henry/Pictures/global_map_real_time_scan_clipped.png", clipped_img)

# main
def main():
    pass
    # test_arg_max()
    # test_queue()
    test_crop()


if __name__ == "__main__":
    main()
