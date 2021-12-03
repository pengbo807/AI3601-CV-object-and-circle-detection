#!/usr/bin/env python3
import cv2
import numpy as np
import sys
import math
import matplotlib.pyplot as plt

def detect_edges(image):
  """Find edge points in a grayscale image.

  Args:
  - image (2D uint8 array): A grayscale image.

  Return:
  - edge_image (2D float array): A heat map where the intensity at each point
      is proportional to the edge magnitude.
  """
  G_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
  G_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
  H,W = image.shape
  edge_image = np.zeros(image.shape)
  for i in range(0, H - 2):
    for j in range(0, W - 2):
        v = np.sum(G_x * image[i:i+3, j:j+3])
        h = np.sum(G_y * image[i:i+3, j:j+3])
        edge_image[i+1, j+1] = np.sqrt((v ** 2) + (h ** 2))
  return edge_image


def hough_circles(edge_image, edge_thresh, radius_values):
  """Threshold edge image and calculate the Hough transform accumulator array.

  Args:
  - edge_image (2D float array): An H x W heat map where the intensity at each
      point is proportional to the edge magnitude.
  - edge_thresh (float): A threshold on the edge magnitude values.
  - radius_values (1D int array): An array of R possible radius values.

  Return:
  - thresh_edge_image (2D bool array): Thresholded edge image indicating
      whether each pixel is an edge point or not.
  - accum_array (3D int array): Hough transform accumulator array. Should have
      shape R x H x W.
  """
  thresh_edge_image = edge_image >= edge_thresh
  H,W = thresh_edge_image.shape
  accum_array = np.zeros((len(radius_values), H, W), dtype=int)
  theta = np.arange(0, 2*math.pi, 2*math.pi / 100)
  

  directions = []
  for i, r in enumerate(radius_values):
    for t in theta:
        directions.append((i, int(r*math.cos(t)), int(r*math.sin(t))))

  for i in range(H):
    for j in range(W):
        if thresh_edge_image[i,j]==0:
            continue
        else:
            for (r_ind, dx, dy) in directions:
                x = j + dx
                y = (H - 1 - i) + dy
                if x >= 0 and x < W and y >= 0 and y < H: accum_array[r_ind, y, x] += 1
  return thresh_edge_image, accum_array


def find_circles(image, accum_array, radius_values, hough_thresh):
    """Find circles in an image using output from Hough transform.

    Args:
    - image (3D uint8 array): An H x W x 3 BGR color image. Here we use the
        original color image instead of its grayscale version so the circles
        can be drawn in color.
    - accum_array (3D int array): Hough transform accumulator array having shape
        R x H x W.
    - radius_values (1D int array): An array of R radius values.
    - hough_thresh (int): A threshold of votes in the accumulator array.

    Return:
    - circles (list of 3-tuples): A list of circle parameters. Each element
        (r, y, x) represents the radius and the center coordinates of a circle
        found by the program.
    - circle_image (3D uint8 array): A copy of the original image with detected
        circles drawn in color.
    """
    accum = accum_array
    accum[accum < hough_thresh] = 0
    (rs_ind, ys, xs) = np.where(accum > 0)
    ys = len(image) - 1 - ys
    circles = []
    circle_image = image.copy()

    for i in range(len(rs_ind)):
        Flag = False
        for item in circles:
            if item[2]-3 < ys[i] and ys[i]<item[2]+3 and item[1]-3 < xs[i] and xs[i]<item[1]+3:
                Flag = True                
                break
        if not Flag:
            circles.append((radius_values[rs_ind[i]], xs[i], ys[i]))
            cv2.circle(circle_image, (xs[i], ys[i]), radius_values[rs_ind[i]], (0, 0, 255), thickness=2)

    return circles, circle_image

def main(argv):
    img_name = argv[0]
    thresh_val1 = int(argv[1])
    thresh_val2 = float(argv[2])
    img = cv2.imread('data/' + img_name + '.png', cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    image_edges = detect_edges(gray_image)
    thresh_edge_image, accum_array = hough_circles(image_edges, thresh_val1, np.arange(25, 40))
    circles, circle_image = find_circles(img, accum_array, np.arange(25, 40), np.amax(accum_array) * thresh_val2)
    print(circles)
    cv2.imwrite('output/' + img_name + "_gray.png", gray_image)
    cv2.imwrite('output/' + img_name + "_edges.png", thresh_edge_image * 255)
    cv2.imwrite('output/' + img_name + '_circles.png', circle_image)


if __name__ == '__main__':
  main(sys.argv[1:])


