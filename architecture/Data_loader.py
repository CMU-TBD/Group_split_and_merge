from __future__ import print_function

from PIL import Image
import numpy as np
import tensorflow as tf
import os
from os import listdir
import cv2

class DataLoader(object):

    def __init__(self):
        self.sample_number = 470
        np.random.seed(100)

        self.seq_length = 16
        self.train_null_img_seqs, self.train_null_locations = self._load_data_sequence(0, 'train')
        self.train_merge_img_seqs, self.train_merge_locations = self._load_data_sequence(1, 'train')
        self.train_split_img_seqs, self.train_split_locations = self._load_data_sequence(2, 'train')
        self.test_null_img_seqs, self.test_null_locations = self._load_data_sequence(0, 'test')
        self.test_merge_img_seqs, self.test_merge_locations = self._load_data_sequence(1, 'test')
        self.test_split_img_seqs, self.test_split_locations = self._load_data_sequence(2, 'test')

        # Form test set
        self.final_test_img_seqs = np.concatenate((self.test_null_img_seqs,
                                                   self.test_merge_img_seqs,
                                                   self.test_split_img_seqs))
        self.final_test_action_seqs = np.concatenate((np.zeros(len(self.test_null_img_seqs)),
                                                      np.ones(len(self.test_merge_img_seqs)) * 1,
                                                      np.ones(len(self.test_split_img_seqs)) * 2))
        self.final_test_location_seqs = np.concatenate((self.test_null_locations,
                                                        self.test_merge_locations,
                                                        self.test_split_locations))

        self.num_train_data = self.sample_number * 3
        self.num_test_data = np.shape(self.final_test_img_seqs)[0]
        self.train_pointer = 0
        self._resample_data_1()
        self._form_validation_data()

        return

    def _load_data_sequence(self, action, purpose_dir):
        load_path = '../' + purpose_dir + '_data/' + str(action)
        img_paths = os.listdir(load_path)
        img_seqs_array = []
        locations_array = []
        print(len(img_paths))
        for img_dir in img_paths:
            img_info = img_dir.split('_')
            action_location = [0, 0]
            if action <> 0:
                action_location[0] = img_info[0]
                action_location[1] = img_info[1]
            locations_array.append(action_location)
            img_seq = []
            for i in range(self.seq_length):
                img = cv2.imread(load_path + '/' + img_dir + '/' + str(i) + '.jpg')
                img = self._process_img(img)
                img_seq.append(img)
            img_seqs_array.append(img_seq)
        return np.array(img_seqs_array), np.array(locations_array)

    def _process_img(self, frame):
        im = Image.fromarray(np.uint8(frame))
        im = im.resize((224, 224))
        im = np.array(im) / 255.0
        im = im[:, :, 0]
        im = np.expand_dims(im, axis = 2)
        return im

    def _data_augmentation(sellf, img):
        img = Image.fromarray(np.uint8(img))
        if np.random.rand() <= 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.rand() <= 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        rotate_ang = np.random.rand() * 360
        img = img.rotate(rotate_ang)
        return np.array(img)

    def _resample_data_1(self):
        null_pick_idxes = np.random.permutation(len(self.train_null_img_seqs))[:self.sample_number]
        merge_pick_idxes = np.random.permutation(len(self.train_merge_img_seqs))[:self.sample_number]
        split_pick_idxes = np.random.permutation(len(self.train_split_img_seqs))[:self.sample_number]
        self.final_img_seqs = []
        self.final_action_seqs = []
        self.final_location_seqs = []
        for idx in null_pick_idxes:
            self.final_img_seqs.append(self.train_null_img_seqs[idx])
            self.final_action_seqs.append([1, 0, 0])
            self.final_location_seqs.append(self.train_null_locations[idx])
        for idx in merge_pick_idxes:
            self.final_img_seqs.append(self.train_merge_img_seqs[idx])
            self.final_action_seqs.append([0, 1, 0])
            self.final_location_seqs.append(self.train_merge_locations[idx])
        for idx in split_pick_idxes:
            self.final_img_seqs.append(self.train_split_img_seqs[idx])
            self.final_action_seqs.append([0, 0, 1])
            self.final_location_seqs.append(self.train_split_locations[idx])
        
        self.final_img_seqs = np.array(self.final_img_seqs)
        self.final_action_seqs = np.array(self.final_action_seqs)
        self.final_location_seqs = np.array(self.final_location_seqs)
        self.train_idxes = np.random.permutation(self.num_train_data)
        np.random.shuffle(self.train_idxes)
        return

    def _form_validation_data(self):
        # Form validation set
        self.validation_img_seqs = []
        self.validation_action_seqs = []
        self.validation_location_seqs = []
        for i in range(self.num_test_data):
            self.validation_img_seqs.append(self.final_img_seqs[self.train_idxes[i], :, :, :, :])
            self.validation_action_seqs.append(self.final_action_seqs[self.train_idxes[i], :])
            self.validation_location_seqs.append(self.final_location_seqs[self.train_idxes[i], :])
        return

    def _resample_data_2(self):
        np.random.shuffle(self.train_idxes)
        return

    def get_seq(self, batch_size):
        if (self.train_pointer + batch_size) < self.num_train_data:
            idxes = range(self.train_pointer, self.train_pointer + batch_size)
            self.train_pointer += batch_size
        else:
            idxes = range(self.train_pointer, self.num_train_data)
            new_end = batch_size - len(idxes)
            idxes += range(new_end)
            self.train_pointer = new_end
            self._resample_data_1()
        
        img_seqs = []
        actions = []
        locations = []
        for idx in idxes:
            seq = self.final_img_seqs[self.train_idxes[idx]]
            action = self.final_action_seqs[self.train_idxes[idx]]
            location = self.final_location_seqs[self.train_idxes[idx]]
            img_seqs.append(seq)
            actions.append(action)
            locations.append(location)

        x = {'img_seq': np.array(img_seqs), 
             'action': np.array(actions), 
             'location': np.array(locations)}
        return x

    def get_test_samples(self):
        x = {'img_seq': self.final_test_img_seqs, 
             'action': self.final_test_action_seqs, 
             'location': self.final_test_location_seqs}
        return x
		
    def get_validation_samples(self):
        x = {'img_seq': self.validation_img_seqs, 
             'action': self.validation_action_seqs,
             'location': self.validation_location_seqs}
        return x
