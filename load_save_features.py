"""
Normalize and visualize inputs and model's learned features.
"""

import cv2
import numpy as np

a = np.load('inputs.npy')[0]
for i in range(16):
  img = a[i] * 255
  cv2.imwrite('feature_img/a_' + str(i) + '.jpg', img)

a_f = np.load('input_features.npy')
a_f = a_f - np.min(a_f)
a_f = a_f / np.max(a_f)
for i in range(16):
  img = a_f[i] * 255
  cv2.imwrite('feature_img/b_' + str(i) + '.jpg', img)
