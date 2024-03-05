from __future__ import print_function

from PIL import Image
import numpy as np
import tensorflow as tf
import os, sys
from os import listdir
import cv2
import pickle

sys.path.append('/home/allanwan/Private/social_grouping_project')
import Social_grouping as sg
class DataLoader(object):

    def __init__(self, sim = False):
        a = sg.SocialGrouping(dataset = 'eth', flag = 0)
        b = sg.SocialGrouping(dataset = 'eth', flag = 1)
        c = sg.SocialGrouping(dataset = 'ucy', flag = 0)
        d = sg.SocialGrouping(dataset = 'ucy', flag = 1)
        e = sg.SocialGrouping(dataset = 'ucy', flag = 2)
        #f = sg.SocialGrouping(dataset = 'ucy', flag = 3)
        #g = sg.SocialGrouping(dataset = 'ucy', flag = 4)

        self.train_class = [b, c, d, e]
        self.test_class = [a]
        self.train_iter = 0
        self.test_iter = 0
        self.test_pos_dict = pickle.load(open('test_data_complete/eth_0_pos.p', 'rb'))
        self.test_neg_dict = pickle.load(open('test_data_complete/eth_0_neg.p', 'rb'))

        self.sim = sim
        print('Start preloading test data...')
        self.test_dic = self._set_test_samples(isPixel = True)
        #self.validation_dic = self._set_validation_samples()
        print('Loading Complete!')
        return

    def _process_img(self, frame):
        im = Image.fromarray(np.uint8(frame))
        im = im.resize((224, 224))
        im = np.array(im) / 255.0
        im = im[:, :, 0]
        im = np.expand_dims(im, axis = 2)
        return im

    def extend_img_seq(self, img_seq):
        seq_length = len(img_seq)
        new_img_seq = []
        for i in range(seq_length):
            new_img_seq.append(img_seq[i])
            new_img_seq.append(img_seq[i])
        return np.array(new_img_seq)

    def get_seq(self, batch_size):
        img_seqs = []
        actions = []
        locations = []
        aug_flag = True
        for i in range(batch_size):
            class_idx = np.random.choice(len(self.train_class))
            cl = self.train_class[class_idx]
            if np.random.rand() < (1/3.0):
                img_seq, action, action_location = \
                    cl.random_positive_data(1, aug_flag, sim = self.sim)
            elif np.random.rand() > (2/3.0):
                img_seq, action, action_location = \
                    cl.random_positive_data(-1, aug_flag, sim = self.sim)
            else:
                img_seq, action, action_location = \
                    cl.random_negative_data(aug_flag, sim = self.sim)
            #tmp_img_seq = []
            #for j in range(len(img_seq)):
            #    tmp_img_seq.append(self._process_img(img_seq[j]))
            #img_seqs.append(tmp_img_seq)
            #img_seq = self.extend_img_seq(img_seq)
            img_seqs.append(img_seq)
            if action == -1:
                action = 2
            tmp_action = [0, 0, 0]
            tmp_action[action] = 1
            actions.append(tmp_action)
            locations.append(action_location)
        x = {'img_seq': np.array(img_seqs), 
             'action': np.array(actions), 
             'location': np.array(locations)}
        return x

    def _set_test_samples(self, isPixel = True):
        img_seqs = self.test_pos_dict['image_sequence'] + self.test_neg_dict['image_sequence']
        #for i, img_seq in enumerate(img_seqs):
        #    img_seqs[i] = self.extend_img_seq(img_seq)
        actions = self.test_pos_dict['action'] + self.test_neg_dict['action']
        reverse_loc_params = self.test_pos_dict['reverse_loc_params'] + \
                             self.test_neg_dict['reverse_loc_params']
        for i in range(len(actions)):
            if actions[i] == -1:
                actions[i] = 2
            tmp_action = [0, 0, 0]
            tmp_action[actions[i]] = 1
            actions[i] = tmp_action
        if isPixel:
            locations = self.test_pos_dict['processed_location'] + self.test_neg_dict['processed_location']
        else:
            locations = self.test_pos_dict['meter_location'] + self.test_neg_dict['meter_location']
        x = {'img_seq': np.array(img_seqs), 
             'action': np.array(actions), 
             'location': np.array(locations),
             'reverse_loc_params': reverse_loc_params}
        return x

    """
    def _set_test_samples(self):
        img_seqs = []
        actions = []
        locations = []
        for i in range(1500):
            #class_idx = np.random.choice(1)
            class_idx = np.random.choice(len(self.test_class))
            cl = self.test_class[class_idx]
            if (self.test_iter % 3) == 0:
                img_seq, action, action_location = \
                    cl.random_positive_data(1, True, sim = self.sim)
            elif (self.test_iter % 3) == 1:
                img_seq, action, action_location = \
                    cl.random_positive_data(-1, True, sim = self.sim)
            else:
                img_seq, action, action_location = \
                    cl.random_negative_data(True, sim = self.sim)
            #tmp_img_seq = []
            #for j in range(len(img_seq)):
            #    tmp_img_seq.append(self._process_img(img_seq[j]))
            #img_seqs.append(tmp_img_seq)
            img_seqs.append(img_seq)
            if action == -1:
                action = 2
            tmp_action = [0, 0, 0]
            tmp_action[action] = 1
            actions.append(tmp_action)
            locations.append(action_location)
            self.test_iter += 1
        x = {'img_seq': np.array(img_seqs), 
             'action': np.array(actions), 
             'location': np.array(locations)}
        return x

    def _set_validation_samples(self):
        img_seqs = []
        actions = []
        locations = []
        for i in range(450):
            class_idx = np.random.choice(len(self.train_class))
            cl = self.train_class[class_idx]
            if (self.train_iter % 3) == 0:
                img_seq, action, action_location = \
                    cl.random_positive_data(1, True, sim = self.sim)
            elif (self.train_iter % 3) == 1:
                img_seq, action, action_location = \
                    cl.random_positive_data(-1, True, sim = self.sim)
            else:
                img_seq, action, action_location = \
                    cl.random_negative_data(True, sim = self.sim)
            #tmp_img_seq = []
            #for j in range(len(img_seq)):
            #    tmp_img_seq.append(self._process_img(img_seq[j]))
            #img_seqs.append(tmp_img_seq)
            img_seqs.append(img_seq)
            if action == -1:
                action = 2
            tmp_action = [0, 0, 0]
            tmp_action[action] = 1
            actions.append(tmp_action)
            locations.append(action_location)
            self.train_iter += 1
        x = {'img_seq': np.array(img_seqs), 
             'action': np.array(actions), 
             'location': np.array(locations)}
        return x
    """

    def get_test_samples(self):
        return self.test_dic

    def get_validation_samples(self):
        return self.validation_dic
