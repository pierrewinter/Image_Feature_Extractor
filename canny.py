#!/usr/bin/env python

""" Performs Canny edge detection for a given image."""
__author__ = "Pierre Winter"
# TODO: Implement function for hysteresis thresholding - https://rosettacode.org/wiki/Canny_edge_detector#Python

import numpy as np
import scipy
import scipy.misc
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from scipy import *
from scipy.ndimage import *
import cv2


def edge_threshold(grad_mag, thresh):

    """
    Takes the array of gradient magnitude values and suppresses pixels below the threshold value.

    grad_mag: Gradient magnitude for an image which is an array of the same shape as the original image.
    thresh: Threshold value for which pixels to include in edge detection.

    return: Array of of gradient magnitude values where pixel values below threshold are suppressed.
    """

    grad_mag_thresh = grad_mag.copy()
    grad_mag_thresh[grad_mag_thresh < thresh] = 0


    return grad_mag_thresh


def angle_quant(angle):

    """
    Takes the array of gradient direction angles and returns a quantized array where the angles
    are grouped into either 0, 45, 90, or 135 degrees.

    :param angle: Array of gradient directions angles for an image (in degrees).
    :return: Array of quantized angles (in degrees) so that they can be used for Canny nonmax-suppression.
    """

    thetaQ = []
    for ang in angle.flatten():

        if np.abs(ang) < 22.5:
            ang = 0
        elif 22.5 <= np.abs(ang) < 67.5:
            ang = 45
        elif 67.5 <= np.abs(ang) < 112.5:
            ang = 90
        elif 112.5 <= np.abs(ang) < 157.5:
            ang = 135
        else:
            ang = 0
        thetaQ.append(ang)

    thetaQ = np.array(thetaQ).reshape(angle.shape)

    return thetaQ


def nonmax_suppression_canny(grad_mag_thresh, grad_dir_quant):

    """
    Takes the gradient magnitude and direction from an image, and performs Canny nonmax suppression in the direction
    normal to the edge. This essentially thins the detected edges.

    :param grad_mag_thresh: Gradient magnitude for an image which is an array of the same shape as the original image.
    This should already be threshold filtered.
    :param grad_dir_quant: Gradient direction for an image which is an array of the same shape as the original image.
    This should already be quantized into 0, 45, 90, 135 degree angles.
    :return: Array of Canny edges after nonmax suppression normal to the edge direction.
    """

    grad_mag_thresh = grad_mag_thresh.copy()
    grad_dir_quant = grad_dir_quant.copy()

    for (i, j), k in ndenumerate(grad_mag_thresh):
        if k > 0:  # only look at pixel values that aren't black
            if 1 < i < grad_mag_thresh.shape[0] - 1:  # ignore edge pixels
                if 1 < j < grad_mag_thresh.shape[1] - 1:  # ignore edge pixels

                    if grad_dir_quant[i, j] == 0:
                        if grad_mag_thresh[i, j] >= grad_mag_thresh[i, j-1] and grad_mag_thresh[i, j] >= grad_mag_thresh[i, j+1]:
                            continue
                        else:
                            grad_mag_thresh[i, j] = 0

                    if grad_dir_quant[i, j] == 45:
                        if grad_mag_thresh[i, j] >= grad_mag_thresh[i-1, j-1] and grad_mag_thresh[i, j] >= grad_mag_thresh[i+1, j+1]:
                            continue
                        else:
                            grad_mag_thresh[i, j] = 0

                    if grad_dir_quant[i, j] == 90:
                        if grad_mag_thresh[i, j] >= grad_mag_thresh[i-1, j] and grad_mag_thresh[i, j] >= grad_mag_thresh[i+1, j]:
                            continue
                        else:
                            grad_mag_thresh[i, j] = 0

                    if grad_dir_quant[i, j] == 135:
                        if grad_mag_thresh[i, j] >= grad_mag_thresh[i-1, j+1] and grad_mag_thresh[i, j] >= grad_mag_thresh[i+1, j-1]:
                            continue
                        else:
                            grad_mag_thresh[i, j] = 0


    return grad_mag_thresh


if __name__ == "__main__":


    ###### Set Parameters ######
    sigma_g = 2
    sobel_k = 5
    edge_thresh = 100


    ###### Read Image ######
    # im = cv2.imread('../pics/CircleLineRect.png', 0).astype('float')
    im = cv2.imread('../pics/zurlim.png', 0).astype('float')


    ####### Gaussian Smoothing #######
    img = gaussian_filter(im, sigma=sigma_g)


    ###### Gradients in x and y (Sobel filters) ######
    im_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_k)
    im_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_k)


    ###### Gradient and Direction ########
    gradient_mag = np.sqrt(im_x**2 + im_y**2)
    gradient_dir = np.arctan2(im_y, im_x) * 180 / np.pi


    ###### Edge Threshold ########
    grad_magnitude_thresh = edge_threshold(gradient_mag, thresh=edge_thresh)


    ###### Quantize Angles ######
    grad_direction_quant = angle_quant(gradient_dir)


    ###### Canny Non-Maximum Suppression ########
    canny_edges = nonmax_suppression_canny(grad_magnitude_thresh, grad_direction_quant)




    ###### Plotting ######
    f, ax_arr = plt.subplots(1, 3, figsize=(18, 16))
    ax_arr[0].set_title("Input Image")
    ax_arr[1].set_title("Gradient Magnitude Threshold")
    ax_arr[2].set_title("Canny Edge Detector")
    ax_arr[0].imshow(im, cmap='gray')
    ax_arr[1].imshow(grad_magnitude_thresh, cmap='gray')
    ax_arr[2].imshow(canny_edges, cmap='gray')
    plt.show()

    # print(im_x)
    # print(scipy.ndimage.sobel(img))
    #
    # plt.imshow(im_x, cmap='gray')
    # plt.imshow(scipy.ndimage.sobel(img), cmap='gray')
    # plt.show()
