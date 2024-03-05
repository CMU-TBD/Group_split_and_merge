"""
Hacky baseline (supplementary CVPR)
"""

import os
import os.path
import time

import numpy as np
import scipy
from scipy import ndimage
import tensorflow as tf
import cv2
from PIL import Image

import bouncing_balls as b
import layer_def as ld
import BasicConvLSTMCell

import Data_loader_aug as dt

def shortest_dist(blob):
    min_dist = 1000000
    center1 = ndimage.center_of_mass(blob, 1)
    center1 = np.array([center1[0], center1[1]])
    min_loc = (0, 0)
    for i in range(224):
        for j in range(224):
            lb = blob[i][j]
            if lb == 2:
                dist = np.linalg.norm(center1 - np.array([i, j]))
                if dist < min_dist:
                    min_dist = dist
                    min_loc = (i, j)
    min_dist = 1000000
    for i in range(224):
        for j in range(224):
            lb = blob[i][j]
            if lb == 1:
                dist = np.linalg.norm(np.array(min_loc) - np.array([i, j]))
                if dist < min_dist:
                    min_dist = dist
    return min_dist

def bbratio(blob):
    min_x = 500
    min_y = 500
    max_x = 0
    max_y = 0
    for i in range(224):
        for j in range(224):  
            lb = blob[i][j]
            if lb == 1:
                if i < min_x:
                    min_x = i
                if i > max_x:
                    max_x = i
                if j < min_y:
                    min_y = j
                if j > max_y:
                    max_y = j
    x_len = (max_x - min_x) * 1.0
    y_len = (max_y - min_y) * 1.0
    return max(x_len, y_len) / min(x_len, y_len)

def main(argv=None):  # pylint: disable=unused-argument

    dt_loader = dt.DataLoader()
    dat = dt_loader.get_test_samples()

    y_gt = dat['action']
    y_gt_lc = dat['location']
    num_total = np.shape(y_gt)[0]
    label_gt = np.argmax(y_gt, axis=1)
    label_pd = []
    loc_accuracies = []
    for idx in range(num_total):
        print(idx)
        #y_pd, y_lc = sess.run([tf.nn.softmax(y), y_loc], 
        #                      feed_dict={x:np.expand_dims(dat['img_seq'][idx], axis=0),
        #                                 is_train: False})
        img_seq = dat['img_seq'][idx]
        img_f = img_seq[0]
        img_r = img_seq[15]
        blobs_f, num_blobs_f = ndimage.label(img_f)
        blobs_r, num_blobs_r = ndimage.label(img_r)
        if (num_blobs_f == 1) and (num_blobs_r == 2):
            y_pd = np.array([1,0,0])
        elif (num_blobs_f == 2) and (num_blobs_r == 1):
            y_pd = np.array([0,1,0])
        elif (num_blobs_f == 1) and (num_blobs_r == 1):
            #cmass_f = np.sum(blobs_f == 1)
            #cmass_r = np.sum(blobs_r == 1)
            rat_f = bbratio(blobs_f)
            rat_r = bbratio(blobs_r)
            if rat_r > rat_f:
                y_pd = np.array([0,0,1])
            else:
                y_pd = np.array([1,0,0])
        elif (num_blobs_f == 2) and (num_blobs_r == 2):
            #center_f1 = np.array(ndimage.center_of_mass(blobs_f, 1))
            #center_f2 = np.array(ndimage.center_of_mass(blobs_f, 2))
            #center_r1 = np.array(ndimage.center_of_mass(blobs_r, 1))
            #center_r2 = np.array(ndimage.center_of_mass(blobs_r, 2))
            #dist_f = np.linalg.norm(center_f1 - center_f2)
            #dist_r = np.linalg.norm(center_r1 - center_r2)
            dist_f = shortest_dist(blobs_f)
            dist_r = shortest_dist(blobs_r)
            if dist_r < dist_f:
                y_pd = np.array([0,1,0])
            else:
                y_pd = np.array([1,0,0])
        else:
            print('Warning!')
            y_pd = np.array([1,0,0])

        y_lc = np.array([[0,0]])
        label_pd.append(np.argmax(y_pd))
        if label_gt[idx] > 0:
            loc_accuracies.append(np.linalg.norm(y_lc[0] - y_gt_lc[idx]))
    num_correct = np.sum(label_gt == np.array(label_pd))
    t_acc = num_correct * 1.0 / num_total
    t_loc_acc = np.mean(np.array(loc_accuracies))
    confusion_matrix = np.zeros((4, 4))
    for i in range(len(label_gt)):
        confusion_matrix[label_gt[i], label_pd[i]] += 1
        confusion_matrix[label_gt[i], 3] += 1
        confusion_matrix[3, label_pd[i]] += 1
    confusion_matrix[3, 3] = len(label_gt)
    print(confusion_matrix)
    print('Test Accuracy: ' + str(t_acc))
    print('Test Location Accuracy: (' + str(len(loc_accuracies)) + ') ' + str(t_loc_acc))

    return
	
if __name__ == '__main__':
	tf.app.run()

