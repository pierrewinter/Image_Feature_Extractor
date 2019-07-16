#!/usr/bin/env python

""" Performs Harris corner detection for a given image."""
__author__ = "Pierre Winter"

from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import rotate
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter
import numpy as np
import cv2


def harris_response(img, sigma, k_h, k_s):

    # Calculate first and second derivatives of the image
    dx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=k_s)
    dy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=k_s)
    dx2 = gaussian_filter(dx*dx, sigma).flatten()
    dy2 = gaussian_filter(dy*dy, sigma).flatten()
    dxdy = gaussian_filter(dx*dy, sigma).flatten()

    # Initialize arrays
    img_len = img.shape[0]*img.shape[1]
    M = np.zeros((img_len, 2, 2))
    R = np.zeros(img_len)

    # Calculate Harris response R for each pixel in image
    for x in range(img_len):
        M[x, 0, 0] = dx2[x]
        M[x, 0, 1] = dxdy[x]
        M[x, 1, 0] = dxdy[x]
        M[x, 1, 1] = dy2[x]
        det_M = np.linalg.det(M[x])
        trace_M = np.trace(M[x])
        R[x] = det_M - k_h*trace_M**2
    R = R.reshape((img.shape[0], img.shape[1]))

    return R


def harris_corner_thresh(harris, thr):
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if harris[i, j] < thr:
                harris[i, j] = 0

    return harris


def harris_edge_thresh(harris, thr):
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if harris[i, j] > -thr:
                harris[i, j] = 0

    return harris


def nonmax_suppression(harris_resp, halfwidth=2):

    """
    Takes a Harris response from an image, performs nonmax suppression, and outputs the x,y values
     of the corners in the image.

    :param harris_resp: Harris response for an image which is an array of the same shape as the original image.
    :param halfwidth: The size of the padding to use in building the window (matrix) for nonmax suppression.
    The window will have a total shape of (2*halfwidth+1, 2*halfwidth+1).
    :return: Tuple of x and y coordinates for the corners that were found from the Harris response
    after nonmax suppression.
    """

    cornersx = []
    cornersy = []
    h, w = harris_resp.shape[:2]
    boxlength = 2*halfwidth + 1
    for i in range(halfwidth, w-halfwidth-1):
        for j in range(halfwidth, h-halfwidth-1):
            matrix = np.zeros((boxlength, boxlength))
            for k in range(-halfwidth, halfwidth+1):
                for l in range(-halfwidth, halfwidth+1):
                    matrix[k+halfwidth, l+halfwidth] = harris_resp[i+k, j+l]
            if matrix[halfwidth, halfwidth] == 0:
                pass
            elif matrix[halfwidth, halfwidth] < np.amax(matrix):
                matrix[halfwidth, halfwidth] = 0
            else:
                cornersx.append(j)
                cornersy.append(i)

    return cornersx, cornersy


if __name__ == "__main__":
    # Define parameters
    # sigmaA = 2.0
    sigmaD = 2.0
    k_sobel = 5
    kappa_harris = 0.04
    rot_angle = 0
    zoom = 1.0
    thresh = 800

    # Read the image
    im = cv2.imread('./pics/CircleLineRect.png', 0).astype('float')

    # Rotation of the image
    if rot_angle != 0:
        im = rotate(im, rot_angle)

    har = harris_response(im, sigmaD, kappa_harris, k_sobel)
    har_edges = harris_edge_thresh(har.copy(), thresh)
    har_corners = harris_corner_thresh(har.copy(), thresh)
    cornx, corny = nonmax_suppression(har_corners, 2)

    f, ax_arr = plt.subplots(1, 4, figsize=(18, 16))
    ax_arr[0].set_title("Input Image")
    ax_arr[1].set_title("Harris Response")
    ax_arr[2].set_title("Harris Response Edges")
    ax_arr[3].set_title("Harris Response Corners w/ nonmax suppression")

    ax_arr[0].imshow(im, cmap='gray')
    ax_arr[1].imshow(har, cmap='gray')
    ax_arr[2].imshow(har_edges, cmap='gray')
    ax_arr[3].imshow(im, cmap='gray')
    ax_arr[3].scatter(x=cornx, y=corny, c='r', s=10)
    plt.show()
