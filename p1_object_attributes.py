#!/usr/bin/env python3
import cv2
import numpy as np
import sys
import math


def binarize(gray_image, thresh_val):
  # TODO: 255 if intensity >= thresh_val else 0
  binary_image = np.zeros_like(gray_image)
  for i in range(len(gray_image)):
    for j in range(len(gray_image[0])):
      if gray_image[i][j]>thresh_val:
        binary_image[i][j] = 255
  return binary_image

def label(binary_image):
    labeled_image = np.zeros_like(binary_image)
    H, W = labeled_image.shape
    eqtable = dict()
    label_num = 1
    non_zeros = np.where(binary_image > 0)
    # first pass
    for i in range(len(non_zeros[0])):
        x, y  = non_zeros[0][i], non_zeros[1][i]
        neighbour = labeled_image[max(0, x - 2):min(x + 2,H), max(0, y - 2):min(y + 2, W)].flatten()
        neighbour = list(neighbour)
        neighbour.sort()
        if neighbour[-1] == 0:
            labeled_image[x][y] = label_num
            label_num += 1
        else:
            if neighbour[-2] == 0:
                labeled_image[x][y] = neighbour[-1]
            else:
                for i in neighbour:
                    if i == 0:
                        continue # background
                    else:
                        m = i
                        labeled_image[x][y] = m
                        for i in neighbour:
                            if i > 0 and i > m: eqtable[i] = m
                        break
    # second pass
    reverse_labels = sorted(eqtable.keys(), reverse=True)
    for i in reverse_labels:
        labeled_image[labeled_image == i] = eqtable[i]
    
    labels = np.unique(labeled_image)
    for i in range(1, len(labels)):
        labeled_image[labeled_image == labels[i]] = int(i / (len(labels)-1) * 255)
    return labeled_image



            

def get_attribute(labeled_image):
  # TODO
  labels, counts = np.unique(labeled_image, return_counts=True)
  labels, counts, attribute_list = labels[1:], counts[1:], []

  for i, label in enumerate(labels):
      # calculate area
      area = counts[i]

      # calculate positions
      ys, xs = np.where(labeled_image == label)
      ys = (labeled_image.shape[0] - 1) - ys
      X, Y = sum(xs) / area, sum(ys) / area # position x,y

      # calculate second moment and other parameters
      a,b,c = 0,0,0
      normed_xs, normed_ys = xs - X, ys - Y
      normed_xys = [((normed_xs[j], normed_ys[j])) for j in range(area)]
      for x, y in normed_xys: 
          a += x ** 2
          b += 2 * x * y
          c += y ** 2

      # calculate orientation (angles)
      theta1 = np.arctan2(b, a - c)/2 # orientation theta1
      theta2 = theta1 + math.pi/2 # theta2
      orientation = theta1

      # calculate roundness
      Emin = a*math.sin(theta1)**2 - b*math.sin(theta1)*math.cos(theta1) + c*math.cos(theta1)**2
      Emax = a*math.sin(theta2)**2 - b*math.sin(theta2)*math.cos(theta2) + c*math.cos(theta2)**2
      roundedness = Emin/Emax

      # attribute list
      attribute_list.append({'label': label,'position':{'x': float(X), 'y': float(Y)}, 
                            'orientation': float(orientation),
                            'roundedness':float(roundedness)})
  return attribute_list

def main(argv):
  img_name = argv[0]
  thresh_val = int(argv[1])

  img = cv2.imread('data/' + img_name + '.png', cv2.IMREAD_COLOR)
  gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  binary_image = binarize(gray_image, thresh_val=thresh_val)
  labeled_image = label(binary_image)
  attribute_list = get_attribute(labeled_image)

  cv2.imwrite('output/' + img_name + "_gray.png", gray_image)
  cv2.imwrite('output/' + img_name + "_binary.png", binary_image)
  cv2.imwrite('output/' + img_name + "_labeled.png", labeled_image)
  
  print(attribute_list)



if __name__ == '__main__':
  main(sys.argv[1:])
