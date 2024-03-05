from __future__ import print_function

import math

import os
import cv2
import copy
import pickle
import numpy as np
#from sets import Set
from scipy.spatial import ConvexHull
from scipy.stats import norm
from scipy.stats import truncnorm
from sklearn.cluster import DBSCAN
from PIL import Image

class SocialGrouping(object):

    # This class imports data from eth and ucy datasets
    # The data are stored in two formats:
    # ---If using people centered format: (indices are matched)
    # ------people_start_frame: record the start frame of each person 
    # ------                    when the person first makes its appearance
    # ------                    (people_start_frame[i] means the start frame of ith person)
    # ------people_end_frame: record the end frame of each person 
    # ------                  when the person last makes its appearance
    # ------                  (people_end_frame[i] means the end frame of ith person)
    # ------people_coords_complete: the coordinates of each person throughout its appearance
    # ------                        (people_coords_complete[i][j][0] means the x coordinate 
    # ------                         of person i in frame j+people_start_frame[i])
    # ------people_velocity_complete: the coordinates of each person throughout its appearance
    # ------                        (people_velocity_complete[i][j][0] means the x velocity 
    # ------                         of person i in frame j+people_start_frame[i])
    # ---If using frame centered format: (indices are matched)
    # ------video_position_matrix: A 3D irregular list
    # ------                       1st Dimension indicates frames
        # ------                       2nd Dimension indicates people
        # ------                       3rd Dimension indicates coordinates of each person
    # ------                       (video_position_matrix[i][j][0] means the x coordinate
        # ------                        of preson j in frame i)
    # ------video_velocity_matrix: A 3D irregular list
    # ------                       1st Dimension indicates frames
        # ------                       2nd Dimension indicates people
        # ------                       3rd Dimension indicates velocities of each person
    # ------                       (video_velocity_matrix[i][j][0] means the x velocity
        # ------                        of preson j in frame i)
    #
    # Note: eth data are in meters form

    def __init__(self, dataset = 'eth', flag = 0):
        self._init_transform_params()

        self.dataset = dataset
        self.flag = flag
        self.video_position_matrix = []
        self.video_velocity_matrix = []
        self.video_pedidx_matrix = []
        self.video_labels_matrix = []
        self.video_debug_labels_matrix = []
        self.video_dynamics_matrix = []

        self.people_start_frame = []
        self.people_end_frame = []
        self.people_coords_complete = []
        self.people_velocity_complete = []

        self.frame_id_list = []
        self.person_id_list = []
        self.x_list = []
        self.y_list = []
        self.vx_list = []
        self.vy_list = []
        self.H = []

        self.action_frames = []
        self.num_groups = 0
        self.train_ratio = 0.9

        if dataset == 'eth':
            read_success = self._read_eth_data(flag)
        elif dataset == 'ucy':
            read_success = self._read_ucy_data(flag)
        else:
            print('dataset argument must be \'eth\' or \'ucy\'')
            read_success = False
        
        if read_success:
            self._organize_frame()
            #self._data_normalization()
            self._data_processing()
            self._load_parameters(dataset)
            self._social_grouping()
            #self._refine_split_merge()
            self.num_merge = 0
            self.num_split = 0
            for action_info in self.video_dynamics_matrix:
                self.action_frames.append(action_info[1])
                action = action_info[0]
                if action == 1:
                    self.num_merge += 1
                else:
                    self.num_split += 1
            self.num_merge_train = int(self.num_merge * self.train_ratio)
            self.num_merge_test = self.num_merge - self.num_merge_train
            self.num_split_train = int(self.num_split * self.train_ratio)
            self.num_split_test = self.num_split - self.num_split_train
            self.merge_category_array = np.array([0] * self.num_merge_train + \
                                                 [1] * self.num_merge_test)
            self.split_category_array = np.array([0] * self.num_split_train + \
                                                 [1] * self.num_split_test)
            np.random.shuffle(self.merge_category_array)
            np.random.shuffle(self.split_category_array)
            self.num_groups += 1

        return

    def _init_transform_params(self):
        self.frame_width = 0
        self.frame_height = 0
        self.H = 0
        self.aug_trans = 0
        self.aug_angle = 0
        self.bbox_crop = 0
        self.process_scale = 0
        self.new_size = 0
        return

    def _read_eth_data(self, flag):
        if flag == 0:
            folder = 'seq_eth'
        elif flag == 1:
            folder = 'seq_hotel'
        else:
            print('Flag for \'eth\' should be 0 or 1')
            return False

        # Create a VideoCapture object and get total number of frames
        self.fname = 'ewap_dataset/' + folder + '/' + folder + '.avi'
        cap = cv2.VideoCapture(self.fname)
        if (cap.isOpened()== False): 
          print("Error opening video stream or file")
        self.total_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(cap.get(3))
        self.frame_height = int(cap.get(4))
        cap.release()
        self.has_video = True

        # Read Homography matrix
        fname = 'ewap_dataset/' + folder + '/H.txt'
        with open(fname) as f:
          for line in f:
            line = line.split(' ')
            real_line = []
            for elem in line:
              if len(elem) > 0:
                real_line.append(elem)
            real_line[-1] = real_line[-1][:-1]
            h1, h2, h3 = real_line
            self.H.append([float(h1), float(h2), float(h3)])

        f.close()

        # Read the data from text file 
        fname = 'ewap_dataset/'+ folder + '/obsmat.txt'

        with open(fname) as f:
          for line in f:
            line = line.split(' ')
            real_line = []
            for elem in line:
              if len(elem) > 0:
                real_line.append(elem)
            real_line[-1] = real_line[-1]
            frame_id, person_id, x, z, y, vx, vz, vy = real_line
            self.frame_id_list.append(int(round(float(frame_id))))
            self.person_id_list.append(int(round(float(person_id))))

            x = float(x)
            y = float(y)
            vx = float(vx)
            vy = float(vy)
            # pt = np.matmul(np.linalg.inv(self.H), [[x], [y], [1.0]])
            # x, y = (pt[0] / pt[2]), (pt[1] / pt[2])
            # pt = np.matmul(np.linalg.inv(self.H), [[vx], [vy], [1.0]])
            # vx, vy = (pt[0] / pt[2]), (pt[1] / pt[2])
            self.x_list.append(x)
            self.y_list.append(y)
            self.vx_list.append(vx)
            self.vy_list.append(vy)

        f.close()

        #curr_std = np.std(np.linalg.norm(np.array([self.vx_list, self.vy_list]), axis = 0))
        #for i in range(len(self.vx_list)):
        #    self.vx_list[i] = self.vx_list[i] / curr_std
        #    self.vx_list[i] = self.vx_list[i] / curr_std

        print('File reading done!')
        return True

    def _read_ucy_data(self, flag):
        if flag == 0:
            folder = 'zara'
            source = 'crowds_zara01'
        elif flag == 1:
            folder = 'zara'
            source = 'crowds_zara02'
        elif flag == 2:
            folder = 'university_students'
            source = 'students003'
        elif flag == 3:
            folder = 'zara'
            source = 'crowds_zara03'
        elif flag == 4:
            folder = 'university_students'
            source = 'students001'
        elif flag == 5:
            folder = 'arxiepiskopi'
            source = 'arxiepiskopi1'
            print('Warning: bad data used!')
        else:
            print('Flag for \'ucy\' should be 0 - 5')
            return False

        # Create a VideoCapture object and read from input file
        if (flag == 3) or (flag == 4):
            if flag == 3:
                alt_source = 'crowds_zara01'
            else:
                alt_source = 'students003'
            self.fname = 'ucy_dataset/' + folder + '/' + alt_source + '.avi'
            cap = cv2.VideoCapture(self.fname)
            if (cap.isOpened()== False):
              print("Error opening video stream or file")
            self.total_num_frames = -1
            self.has_video = False
        else:
            self.fname = 'ucy_dataset/' + folder + '/' + source + '.avi'
            cap = cv2.VideoCapture(self.fname)
            if (cap.isOpened()== False):
              print("Error opening video stream or file")
            self.total_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.has_video = True
        self.frame_width = int(cap.get(3))
        self.frame_height = int(cap.get(4))
        cap.release()

        # Approximate H matrix
        offx = 17.5949
        offy = 9.665722
        pts_img = np.array([[476, 117], [562, 117], [562, 311],[476, 311]])
        pts_wrd = np.array([[0, 0], [1.81, 0], [1.81, 4.63],[0, 4.63]])
        pts_wrd[:,0] += offx
        pts_wrd[:,1] += offy
        self.H, status = cv2.findHomography(pts_img, pts_wrd)

        # Read the data from text file 
        fname = 'ucy_dataset/' + folder + '/data_' + folder + '/' + source + '.vsp'

        with open(fname) as f:
          person_id = 0
          mem_x = 0
          mem_y = 0
          mem_frame_id = 0
          record_switch = True

          for idx, line in enumerate(f):
            line = line.split()
            if len(line) == 6:
              person_id += 1
              record_switch = False
              if idx > 0:
                vx = self.vx_list[-1]
                vy = self.vy_list[-1]
                self.vx_list.append(vx)
                self.vy_list.append(vy)
            if (len(line) == 8) or (len(line) == 4):
              x = float(line[0])
              y = float(line[1])
              frame_id = int(line[2])
              pt = np.matmul(self.H, [[x], [y], [1.0]]) # H
              x, y = (pt[0][0] / pt[2][0]), (pt[1][0] / pt[2][0])
              self.x_list.append(x)
              self.y_list.append(y)
              self.frame_id_list.append(frame_id)
              self.person_id_list.append(person_id)
              if record_switch:
                vx = (x - mem_x) / (frame_id - mem_frame_id)
                vy = (y - mem_y) / (frame_id - mem_frame_id)
                self.vx_list.append(vx * 25)
                self.vy_list.append(vy * 25)
              else:
                record_switch = True
              mem_x = x
              mem_y = y
              mem_frame_id = frame_id

          vx = self.vx_list[-1]
          vy = self.vy_list[-1]
          self.vx_list.append(vx)
          self.vy_list.append(vy)

        f.close()

        #curr_std = np.std(np.linalg.norm(np.array([self.vx_list, self.vy_list]), axis = 0))
        #for i in range(len(self.vx_list)):
        #    self.vx_list[i] = self.vx_list[i] / curr_std
        #    self.vx_list[i] = self.vx_list[i] / curr_std

        print('File reading done!')
        return True

    def _organize_frame(self):

        # Connect paths for each individual person
        # For each person, the frame ids, the associated path 
        # coordinates and velocities are stored
        num_people = max(self.person_id_list)
        for i in range(num_people):
          got_start = False
          prev_frame = 0
          curr_frame = 0
          person_coords = []
          person_velocity = []
          mem_x = 0
          mem_y = 0
          mem_vel_x = 0
          mem_vel_y = 0

          for j in range(len(self.frame_id_list)):
            if self.person_id_list[j] == (i + 1):
              curr_frame = self.frame_id_list[j]
              if not got_start:
                got_start = True
                self.people_start_frame.append(curr_frame)
                person_coords.append((self.x_list[j], self.y_list[j]))
                person_velocity.append((self.vx_list[j], self.vy_list[j]))
              else:
                num_frames_interpolated = curr_frame - prev_frame
                for k in range(num_frames_interpolated):
                  ratio = float(k + 1) / float(num_frames_interpolated)
                  diff_x = ratio * (self.x_list[j] - mem_x)
                  diff_y = ratio * (self.y_list[j] - mem_y)
                  person_coords.append((mem_x + diff_x, mem_y + diff_y))

                  diff_vx = ratio * (self.vx_list[j] - mem_vel_x)
                  diff_vy = ratio * (self.vy_list[j] - mem_vel_y)
                  person_velocity.append((mem_vel_x + diff_vx, mem_vel_y + diff_vy))

              mem_x, mem_y = self.x_list[j], self.y_list[j]
              mem_vel_x, mem_vel_y = self.vx_list[j], self.vy_list[j]
              prev_frame = curr_frame

          if got_start:
            self.people_end_frame.append(curr_frame)
            self.people_coords_complete.append(person_coords)
            self.people_velocity_complete.append(person_velocity)

        if self.total_num_frames == -1:
            self.total_num_frames = max(self.people_end_frame)
        
        print('Frame organizing done!')
        return

    def _data_normalization(self, xy_separate = False):
        all_coords_x = []
        all_coords_y = []
        all_velocity_x = []
        all_velocity_y = []
        all_coords = []
        all_velocity = []
        for i in range(len(self.people_coords_complete)):
            for j in range(len(self.people_coords_complete[i])):
                all_coords_x.append(self.people_coords_complete[i][j][0])
                all_coords_y.append(self.people_coords_complete[i][j][1])
                all_coords.append(self.people_coords_complete[i][j][0])
                all_coords.append(self.people_coords_complete[i][j][1])
                all_velocity_x.append(self.people_velocity_complete[i][j][0])
                all_velocity_y.append(self.people_velocity_complete[i][j][1])
                all_velocity.append(self.people_velocity_complete[i][j][0])
                all_velocity.append(self.people_velocity_complete[i][j][1])

        self.coords_x_mean = np.mean(np.array(all_coords_x))
        self.coords_y_mean = np.mean(np.array(all_coords_y))
        self.coords_x_std = np.std(np.array(all_coords_x))
        self.coords_y_std = np.std(np.array(all_coords_y))
        self.coords_mean = np.mean(np.array(all_coords))
        self.coords_std = np.std(np.array(all_coords))
        self.velocity_x_mean = np.mean(np.array(all_velocity_x))
        self.velocity_y_mean = np.mean(np.array(all_velocity_y))
        self.velocity_x_std = np.std(np.array(all_velocity_x))
        self.velocity_y_std = np.std(np.array(all_velocity_y))
        self.velocity_mean = np.mean(np.array(all_velocity))
        self.velocity_std = np.std(np.array(all_velocity))

        for i in range(len(self.people_coords_complete)):
            for j in range(len(self.people_coords_complete[i])):
                if xy_separate:
                    new_coord_x = (self.people_coords_complete[i][j][0] - self.coords_x_mean) / \
                                  self.coords_x_std
                    new_coord_y = (self.people_coords_complete[i][j][1] - self.coords_y_mean) / \
                                  self.coords_y_std
                    new_vel_x = (self.people_velocity_complete[i][j][0] - self.velocity_x_mean) / \
                                 self.velocity_x_std
                    new_vel_y = (self.people_velocity_complete[i][j][1] - self.velocity_y_mean) / \
                                 self.velocity_y_std
                else:
                    new_coord_x = (self.people_coords_complete[i][j][0] - self.coords_mean) / \
                                   self.coords_std
                    new_coord_y = (self.people_coords_complete[i][j][1] - self.coords_mean) / \
                                   self.coords_std
                    new_vel_x = (self.people_velocity_complete[i][j][0] - self.velocity_mean) / \
                                 self.velocity_std
                    new_vel_y = (self.people_velocity_complete[i][j][1] - self.velocity_mean) / \
                                 self.velocity_std
                self.people_coords_complete[i][j] = (new_coord_x, new_coord_y)
                self.people_velocity_complete[i][j] = (new_vel_x, new_vel_y)

        return

    def _data_processing(self):
        # Precompute a 3d video array for displaying later
        # Frames x Clusters x Entities
        # Clustering is done using DBScan


        for i in range(self.total_num_frames):
            position_array = []
            velocity_array = []
            pedidx_array = []
            curr_frame = i + 1

            for j in range(len(self.people_start_frame)):
                curr_start = self.people_start_frame[j]
                curr_end = self.people_end_frame[j]
                if (curr_start <= curr_frame) and (curr_frame <= curr_end):
                    x, y = self.people_coords_complete[j][curr_frame - curr_start]
                    vx, vy = self.people_velocity_complete[j][curr_frame - curr_start]

                    #computes and gets the coords in pixels instead of meters (NOT YET)
                    position_array.append((float(x), float(y)))
                    velocity_array.append((float(vx), float(vy)))
                    pedidx_array.append(j)
              
            if len(position_array) > 0:
                self.video_position_matrix.append(position_array)
                self.video_velocity_matrix.append(velocity_array)
                self.video_pedidx_matrix.append(pedidx_array)
            else:
                self.video_position_matrix.append([])
                self.video_velocity_matrix.append([])
                self.video_pedidx_matrix.append([])

        print('Initial data processing done!')
        return

    def _load_parameters(self, dataset):
        offset = 12
        history = offset + 16
        seq_length = 16
        future = 0
        interval = 2
        self.ucy_scale = 1 #60
        pos = 2.0
        ori = 30
        vel = 1.0
        if dataset == 'eth':
            self.param = {'position_threshold': pos,
                     'orientation_threshold': ori / 180.0 * math.pi,
                     'velocity_threshold': vel,
                     'temporal_threshold': 0.3,
                     'velocity_ignore_threshold': 0.5,
                     'label_history_threshold': history,
                     'label_future_threshold': future,
                     'label_history_offset': offset,
                     'label_history_seqlength': seq_length,
                     'label_history_interval': interval}
        elif dataset == 'ucy':
            self.param = {'position_threshold': pos * self.ucy_scale,
                     'orientation_threshold': ori / 180.0 * math.pi,
                     'velocity_threshold': vel * self.ucy_scale, #16.5,
                     'temporal_threshold': 0.3 * self.ucy_scale,
                     'velocity_ignore_threshold': 0.5 * self.ucy_scale,
                     'label_history_threshold': history,
                     'label_future_threshold': future,
                     'label_history_offset': offset,
                     'label_history_seqlength': seq_length,
                     'label_history_interval': interval}
        else:
            raise Exception('Non existant dataset!')

        return

    def _DBScan_grouping(self, labels, properties, standard):
        max_lb = max(labels)
        for lb in range(max_lb + 1):
            sub_properties = []
            sub_idxes = []
            for i in range(len(labels)):
                if labels[i] == lb:
                    sub_properties.append(properties[i])
                    sub_idxes.append(i)
            if len(sub_idxes) > 1:
                db = DBSCAN(eps = standard, min_samples = 1)
                sub_labels = db.fit_predict(sub_properties)
                max_label = max(labels)
                for i, sub_lb in enumerate(sub_labels):
                    if sub_lb > 0:
                        labels[sub_idxes[i]] = max_label + sub_lb
        return labels

    def _check_history(self, label, frame_idx):
        history = self.param['label_history_threshold']
        if frame_idx < history:
            return False
        for i in range(frame_idx - history, frame_idx):
            if not (label in self.video_labels_matrix[i]):
                return False
        return True

    def _check_future(self, label, frame_idx):
        future = self.param['label_future_threshold']
        if frame_idx > (self.total_num_frames - future):
            end_idx = self.total_num_frames
        else:
            end_idx = frame_idx + future
        for i in range(frame_idx, end_idx):
            if not (label in self.video_labels_matrix[i]):
                return False
        return True

    def _social_grouping(self):
        prev_labels = []
        prev_pedidx = []
        for i in range(self.total_num_frames):
            # get grouping criterion
            position_array = self.video_position_matrix[i]
            velocity_array = self.video_velocity_matrix[i]
            pedidx_array = self.video_pedidx_matrix[i]
            num_people = len(position_array)
            if not (num_people > 0):
                prev_labels = []
                prev_pedidx = []
                self.video_labels_matrix.append([])
                self.video_debug_labels_matrix.append([])
                continue
            vel_orientation_array = []
            vel_magnitude_array = []
            for [vx, vy] in velocity_array:
                velocity_magnitude = math.sqrt(math.pow(vx, 2) + math.pow(vy, 2))
                if velocity_magnitude < self.param['velocity_ignore_threshold']:
                    vel_orientation_array.append((0.0, 0.0))
                    vel_magnitude_array.append((0.0, 0.0))
                else:
                    vel_orientation_array.append(
                        (vx / velocity_magnitude, vy / velocity_magnitude))
                    vel_magnitude_array.append(
                        (0.0, velocity_magnitude)) # Add 0 to fool DBSCAN
            # grouping in current frame
            labels = [0] * num_people
            labels = self._DBScan_grouping(labels, vel_orientation_array, 
                                           self.param['orientation_threshold'])
            labels = self._DBScan_grouping(labels, vel_magnitude_array, 
                                           self.param['velocity_threshold'])
            labels = self._DBScan_grouping(labels, position_array, 
                                           self.param['position_threshold'])

            # Fixes to ensure temporal consistency (cross frame comparison)
            if i == 0:
                temporal_labels = copy.deepcopy(labels)
            else:
                temporal_labels = [-1] * num_people
                # Get the temporal labels (labeled w.r.t. a close label in last frame)
                for j in range(num_people):
                    curr_idx = pedidx_array[j]                    
                    for k in range(len(prev_labels)):
                        if prev_pedidx[k] == curr_idx:
                            temporal_labels[j] = prev_labels[k]

                    """
                    curr_label = labels[j]
                    curr_pos = position_array[j]
                    distances = [0] * len(prev_labels)
                    min_dist = 100000
                    for k in range(len(prev_labels)):
                        target_pos = prev_positions[k]
                        distances[k] = math.sqrt(math.pow(curr_pos[0] - target_pos[0], 2) +
                                                 math.pow(curr_pos[1] - target_pos[1], 2))
                        if distances[k] < min_dist:
                            min_dist = distances[k]
                            min_idx = k
                    if min_dist < self.param['temporal_threshold']:
                        temporal_labels[j] = prev_labels[min_idx]
                    """

                # Figure out new groups
                for j in range(num_people):
                    curr_label = temporal_labels[j]
                    reference_label = labels[j]
                    # new group or join current group
                    if curr_label == -1:
                        found_group = False
                        for k in range(num_people):
                            if (labels[k] == reference_label) and (temporal_labels[k] != -1):
                                change_to_label = temporal_labels[k]
                                found_group = True
                        if not found_group:
                            change_to_label = max(self.num_groups, max(temporal_labels)) + 1
                        for k in range(j, num_people):
                            if labels[k] == reference_label:
                                temporal_labels[k] = change_to_label
                        
                # resolve splits and merges
                dynamics_array = []
                for j in range(num_people):
                    curr_label = temporal_labels[j]
                    reference_label = labels[j]
                    for k in range(num_people):
                        if (temporal_labels[k] != curr_label) and \
                           (labels[k] == reference_label):  #merges
                            change_to_label = max(self.num_groups, max(temporal_labels)) + 1
                            if (self._check_history(temporal_labels[k], i)) and \
                               (self._check_history(curr_label, i)):
                                dynamics_array.append((1, curr_label, temporal_labels[k], j))
                            for l in range(num_people):
                                if labels[l] == reference_label:
                                    temporal_labels[l] = change_to_label
                        if (temporal_labels[k] == curr_label) and \
                           (labels[k] != reference_label): #splits
                            change_to_label_1 = max(self.num_groups, max(temporal_labels)) + 1
                            change_to_label_2 = max(self.num_groups, max(temporal_labels)) + 2
                            if self._check_history(curr_label, i):
                                dynamics_array.append((-1, curr_label, j, k))
                            for l in range(num_people):
                                if (labels[l] == labels[k]):
                                    temporal_labels[l] = change_to_label_1
                                if (labels[l] == reference_label):
                                    temporal_labels[l] = change_to_label_2

                """
                dynamics_array = []
                for j in range(num_people):
                    curr_label = temporal_labels[j]
                    reference_label = labels[j]
                    for k in range(num_people):
                        if (temporal_labels[k] == curr_label) and \
                           (labels[k] != reference_label): #splits
                            change_to_label_1 = max(temporal_labels) + 1
                            change_to_label_2 = max(temporal_labels) + 2
                            if self._check_history(curr_label, i):
                                dynamics_array.append((-1, curr_label, j, k))
                            for l in range(num_people):
                                if (labels[l] == labels[k]):
                                    temporal_labels[l] = change_to_label_1
                                if (labels[l] == reference_label):
                                    temporal_labels[l] = change_to_label_2

                # resolve merges
                for j in range(num_people):
                    curr_label = temporal_labels[j]
                    reference_label = labels[j]
                    for k in range(num_people):
                        if (temporal_labels[k] != curr_label) and \
                           (labels[k] == reference_label):  #merges
                            change_to_label = max(temporal_labels) + 1
                            if (self._check_history(temporal_labels[k], i)) and \
                               (self._check_history(curr_label, i)):
                                dynamics_array.append((1, curr_label, temporal_labels[k], j))
                            for l in range(num_people):
                                if labels[l] == reference_label:
                                    temporal_labels[l] = change_to_label
                """

                for info in dynamics_array:
                    if info[0] == -1:
                        self.video_dynamics_matrix.append((-1, i, info[1], 
                                temporal_labels[info[2]], temporal_labels[info[3]]))
                    elif info[0] == 1:
                        self.video_dynamics_matrix.append((1, i, info[1], info[2],
                                temporal_labels[info[3]]))

            prev_labels = temporal_labels
            prev_pedidx = pedidx_array
            self.num_groups = max(self.num_groups, max(temporal_labels))
            self.video_labels_matrix.append(temporal_labels)
            self.video_debug_labels_matrix.append(labels)

        print('Social Grouping done!')
        return

    def _refine_split_merge(self):
        new_video_dynamics_matrix = []
        for i, action_info in enumerate(self.video_dynamics_matrix):
            action = action_info[0]
            action_frame = action_info[1]
            if action == 1:
                after_group_label = action_info[4]
                flag1 = self._check_future(after_group_label, action_frame)
                flag2 = True
            elif action == -1:
                after_group_1_label = action_info[3]
                after_group_2_label = action_info[4]
                flag1 = self._check_future(after_group_1_label, action_frame)
                flag2 = self._check_future(after_group_2_label, action_frame)
            if flag1 and flag2:
                new_video_dynamics_matrix.append(action_info)
        self.video_dynamics_matrix = new_video_dynamics_matrix
        return


    def _draw_social_shapes(self, frame, position, velocity, data_aug, draw = True):
        front_coeff = 1.0
        side_coeff = 2.0 / 3.0
        rear_coeff = 0.5
        total_increments = 20
        quater_increments = total_increments / 4
        angle_increment = 2 * math.pi / total_increments
        current_target = 0.8

        contour_points = []
        for i in range(len(position)):
            center_x = position[i][0]
            center_y = position[i][1]
            velocity_x = velocity[i][0]
            velocity_y = velocity[i][1]
            
            velocity_magnitude = math.sqrt(math.pow(velocity_x, 2) + math.pow(velocity_y, 2))
            velocity_angle = math.atan2(velocity_y, velocity_x)
            if self.dataset == 'eth':
                variance_front = max(0.5, front_coeff * velocity_magnitude)
            else:
                variance_front = max(0.5 * self.ucy_scale, front_coeff * velocity_magnitude) \
                                 * self.ucy_scale
            variance_side = side_coeff * variance_front
            variance_rear = rear_coeff * variance_front

            for j in range(total_increments):
                if (j / quater_increments) == 0:
                    prev_variance = variance_front
                    next_variance = variance_side
                elif (j / quater_increments) == 1:
                    prev_variance = variance_rear
                    next_variance = variance_side
                elif (j / quater_increments) == 2:
                    prev_variance = variance_rear
                    next_variance = variance_side
                else:
                    prev_variance = variance_front
                    next_variance = variance_side

                current_variance = prev_variance + (next_variance - prev_variance) * \
                                   (j % quater_increments) / float(quater_increments)
                #value = norm.ppf(current_target, scale = math.sqrt(current_variance))
                #value = 0.8416 * math.sqrt(current_variance)
                value = math.sqrt(0.354163 / ((math.cos(angle_increment * j) ** 2 / (2 * prev_variance)) + (math.sin(angle_increment * j) ** 2 / (2 * next_variance))))

                addition_angle = velocity_angle + angle_increment * j
                append_x = center_x + math.cos(addition_angle) * value
                append_y = center_y + math.sin(addition_angle) * value
                x, y = self._coordinate_transform((append_x, append_y))
                contour_points.append((y, x))

        convex_hull_vertices = []
        hull = ConvexHull(np.array(contour_points))
        for i in hull.vertices:
            hull_vertice = (contour_points[i][0], contour_points[i][1])
            if data_aug:
                convex_hull_vertices.append(self._aug_transform(hull_vertice))
            else:
                convex_hull_vertices.append(hull_vertice)

        if draw:
            cv2.fillConvexPoly(frame, np.array(convex_hull_vertices), (255, 255, 255))
            return frame
        else:
            return convex_hull_vertices

    def _draw_simulated_social_shapes(self, frame, position, data_aug, debug = False):
        robo_pos = self.curr_robo_pos
        ped_diameter = 0.5
        scan_res = 0.1 * math.pi / 180
        r_sq = (ped_diameter / 2.0) ** 2
        add_noise = True
        noise_limit = 0.05

        laser_points = []
        th = 0
        while th < (math.pi * 2):
            if not abs(th - math.pi / 2) < 1e-12:
                min_dist = 1e5
                laser_x = None
                laser_y = None
                for i in range(len(position)):
                    a = position[i][0] - robo_pos[0]
                    b = position[i][1] - robo_pos[1]
                    A = 1 + math.tan(th) ** 2
                    B = -2 * (a + b * math.tan(th))
                    C = a ** 2 + b ** 2 - r_sq
                    check_root = round(B ** 2 - 4 * A * C, 12)
                    if check_root >= 0:
                        x1 = (-B - math.sqrt(check_root)) / (2 * A)
                        y1 = x1 * math.tan(th)
                        x2 = (-B + math.sqrt(check_root)) / (2 * A)
                        y2 = x2 * math.tan(th)
                        mag1 = math.sqrt(x1 ** 2 + y1 ** 2)
                        mag2 = math.sqrt(x2 ** 2 + y2 ** 2)
                        if mag1 < mag2:
                            append_x = x1
                            append_y = y1
                            dist = mag1
                        else:
                            append_x = x2
                            append_y = y2
                            dist = mag2
                        if add_noise:
                            append_x += truncnorm.rvs(-noise_limit, noise_limit)
                            append_y += truncnorm.rvs(-noise_limit, noise_limit)
                        if dist < min_dist:
                            min_dist = dist
                            laser_x = append_x + robo_pos[0]
                            laser_y = append_y + robo_pos[1]
                if not (laser_x == None):
                    x, y = self._coordinate_transform((laser_x, laser_y))
                    laser_points.append((y, x))
            th += scan_res

        if len(laser_points) > 2:
            convex_hull_vertices = []
            hull = ConvexHull(np.array(laser_points))
            for i in hull.vertices:
                hull_vertice = (laser_points[i][0], laser_points[i][1])
                if data_aug:
                    convex_hull_vertices.append(self._aug_transform(hull_vertice))
                else:
                    convex_hull_vertices.append(hull_vertice)

            cv2.fillConvexPoly(frame, np.array(convex_hull_vertices), (255, 255, 255))
            if debug:
                for i in range(len(convex_hull_vertices)):
                    cv2.circle(frame, (convex_hull_vertices[i][0], 
                                       convex_hull_vertices[i][1]), 2, (0,0,255), 2)
        else:
            print('Warning: Laser did not pick up any scan point!')
        return frame

    def _draw_canvas_handler(self, canvas, frame_idx, group_label, data_aug, sim):
        positions, velocities, pedidx = self._find_label_properties(frame_idx, group_label)
        if sim:
            canvas = self._draw_simulated_social_shapes(canvas, positions, data_aug)
        else:
            canvas = self._draw_social_shapes(canvas, positions, velocities, data_aug)
        return canvas, pedidx

    def _set_aug_param(self, angle, translation):
        self.aug_angle = angle / 180.0 * math.pi
        self.aug_trans = translation
        return

    def _prepare_aug_param(self, frame_idx, group_1, group_2):
        #XXX XXX XXX XXX XXX DANGER!!!!!!!!!!!!!
        if group_2 == -1:
            positions, _, _ = self._find_label_properties(frame_idx, group_1)
            center = self._find_centroid(positions)
        else:
            positions_1, _, _ = self._find_label_properties(frame_idx, group_1)
            positions_2, _, _ = self._find_label_properties(frame_idx, group_2)
            center = self._find_centroid(positions_1 + positions_2)
        center[1], center[0] = self._coordinate_transform(center)
        self._set_aug_param(np.random.choice(360),
                            [self.frame_width / 2 - center[0],
                             self.frame_height / 2 - center[1]])
        return

    def _aug_transform(self, coord):
        x = coord[0] + self.aug_trans[0]
        y = coord[1] + self.aug_trans[1]
        x -= self.frame_width / 2
        y -= self.frame_height / 2
        nx = math.cos(self.aug_angle) * x - math.sin(self.aug_angle) * y
        ny = math.sin(self.aug_angle) * x + math.cos(self.aug_angle) * y
        nx += self.frame_width / 2
        ny += self.frame_height / 2
        return (int(nx), int(ny))

    def _reverse_aug_transform(self, coord):
        x = coord[0]
        y = coord[1]
        x -= self.frame_width / 2
        y -= self.frame_height / 2
        nx = math.cos(-self.aug_angle) * x - math.sin(-self.aug_angle) * y
        ny = math.sin(-self.aug_angle) * x + math.cos(-self.aug_angle) * y
        nx += self.frame_width / 2
        ny += self.frame_height / 2
        return (int(nx) - self.aug_trans[0], int(ny) - self.aug_trans[1])

    def _coordinate_transform(self, coord):
        pt = np.matmul(np.linalg.inv(self.H), [[coord[0]], [coord[1]], [1.0]])
        x = pt[0][0] / pt[2][0]
        y = pt[1][0] / pt[2][0]
        if self.dataset == 'ucy':
            tmp_y = y
            y = self.frame_width / 2 + x
            x = self.frame_height / 2 - tmp_y
        x = int(round(x))
        y = int(round(y))
        return x, y

    def _reverse_coordinate_transform(self, coord):
        x = coord[0]
        y = coord[1]
        if self.dataset == 'ucy':
            tmp_y = y
            y = x
            x = tmp_y
            x = x - self.frame_width / 2
            y = self.frame_height / 2 - y
        pt = np.matmul(self.H, [[x], [y], [1.0]])
        x = pt[0][0] / pt[2][0]
        y = pt[1][0] / pt[2][0]
        return x, y

    def _find_label_properties(self, frame_idx, label):
        positions = []
        velocities = []
        pedidx = []
        labels = self.video_labels_matrix[frame_idx]
        for i in range(len(labels)):
            if label == labels[i]:
                positions.append(self.video_position_matrix[frame_idx][i])
                velocities.append(self.video_velocity_matrix[frame_idx][i])
                pedidx.append(self.video_pedidx_matrix[frame_idx][i])
        return positions, velocities, pedidx

    def _find_centroid(self, positions):
        center = [0, 0]
        for ps in positions:
            center[0] += ps[0]
            center[1] += ps[1]
        center[0] = center[0] / float(len(positions))
        center[1] = center[1] / float(len(positions))
        return center

    def _find_action_location(self, frame_idx, label_1, label_2):
        positions_1, velocities_1, _ = self._find_label_properties(frame_idx, label_1)
        positions_2, velocities_2, _ = self._find_label_properties(frame_idx, label_2)
        if (positions_1 == []) or (positions_2 == []):
            print(label_1)
            print(label_2)
            print(self.video_labels_matrix[frame_idx])
            raise Exception('no labels in frame!')
            return (0, 0)
        convex_1 = self._draw_social_shapes([], positions_1, velocities_1, False, False)
        convex_2 = self._draw_social_shapes([], positions_2, velocities_2, False, False)
        convex_center_1 = self._find_centroid(positions_1)
        convex_center_2 = self._find_centroid(positions_2)
        convex_center_1[1], convex_center_1[0] = self._coordinate_transform(convex_center_1)
        convex_center_2[1], convex_center_2[0] = self._coordinate_transform(convex_center_2)

        min_dist_1 = 1000000
        min_dist_2 = 1000000
        for k in convex_1:
            tmp_dist = math.sqrt((k[0] - convex_center_2[0]) ** 2 + (k[1] - convex_center_2[1]) ** 2)
            if tmp_dist < min_dist_1:
                min_dist_1 = tmp_dist
                min_coords_1 = k
        for k in convex_2:
            tmp_dist = math.sqrt((k[0] - convex_center_1[0]) ** 2 + (k[1] - convex_center_1[1]) ** 2)
            if tmp_dist < min_dist_2:
                min_dist_2 = tmp_dist
                min_coords_2 = k
        action_coord = (int(round((min_coords_1[0] + min_coords_2[0]) / 2)),
                        int(round((min_coords_1[1] + min_coords_2[1]) / 2)))
        
        return action_coord

    def _prepare_process_image(self, img_seq):
        left = self.frame_width / 2
        up = self.frame_height / 2
        right = self.frame_width / 2
        low = self.frame_height / 2
        for img in img_seq:
            im = Image.fromarray(np.uint8(img))
            bbox = im.getbbox()
            if bbox == None:
                continue
            if bbox[0] < left:
                left = bbox[0]
            if bbox[1] < up:
                up = bbox[1]
            if bbox[2] > right:
                right = bbox[2]
            if bbox[3] > low:
                low = bbox[3]
        upper_left = (left, up)
        lower_right = (right, low)
        upper_left_p = (self.frame_width - upper_left[0], self.frame_height - upper_left[1])
        lower_right_p = (self.frame_width - lower_right[0], self.frame_height - lower_right[1])
        self.bbox_crop = (min(upper_left[0], lower_right[0], upper_left_p[0], lower_right_p[0]),
                     min(upper_left[1], lower_right[1], upper_left_p[1], lower_right_p[1]),
                     max(upper_left[0], lower_right[0], upper_left_p[0], lower_right_p[0]),
                     max(upper_left[1], lower_right[1], upper_left_p[1], lower_right_p[1]))
        box_size = (self.bbox_crop[2] - self.bbox_crop[0], 
                    self.bbox_crop[3] - self.bbox_crop[1])
        self.process_scale = 224.0 / max(box_size)
        self.new_size = (int(box_size[0] * self.process_scale), 
                         int(box_size[1] * self.process_scale))
        return

    def _process_image(self, img, debug = False):
        im = Image.fromarray(np.uint8(img))
        im = im.crop(self.bbox_crop)
        paste_im = im.resize(self.new_size)
        im = Image.new('RGB', (224, 224))
        im.paste(paste_im, (112 - paste_im.size[0] // 2, 112 - paste_im.size[1] // 2))
        if debug:
            return np.array(im)
        else:
            im = np.array(im) / 255.0
            im = im[:, :, 0]
            im = np.expand_dims(im, axis = 2)
            return im

    def _process_action_location(self, location):
        tmp_location = (location[0] - self.bbox_crop[0], location[1] - self.bbox_crop[1])
        tmp_location = (int(tmp_location[0] * self.process_scale), 
                        int(tmp_location[1] * self.process_scale))
        new_location = (tmp_location[0] + (112 - self.new_size[0] / 2),
                        tmp_location[1] + (112 - self.new_size[1] / 2))
        return new_location

    def _reverse_action_location(self, location):
        tmp_location = (location[0] - (112 - self.new_size[0] / 2), 
                        location[1] - (112 - self.new_size[1] / 2))
        tmp_location = (int(tmp_location[0] / self.process_scale), 
                        int(tmp_location[1] / self.process_scale))
        new_location = (tmp_location[0] + self.bbox_crop[0] - self.aug_trans[0], 
                        tmp_location[1] + self.bbox_crop[1] - self.aug_trans[1])
        return new_location

    def _generate_positive_datapoint(self, action_info, 
                                     debug = False, data_aug = False, into_future = False,
                                     sim = False):
        history = self.param['label_history_threshold']
        interval = self.param['label_history_interval']
        offset = self.param['label_history_offset']
        seq_length = self.param['label_history_seqlength']
        future = self.param['label_future_threshold']
        action = action_info[0]
        action_frame = action_info[1]
        img_seq = []
        all_positions_seq = []
        all_pedidx_seq = []
        frames_seq = []
        pedidx_seq = []
        pedidx_action_seq = []
        if sim:
            self.robo_pos = self._generate_robot_location(action_frame)
        if action == 1:
            pre_group_1_label = action_info[2]
            pre_group_2_label = action_info[3]
            after_group_label = action_info[4]
            action_location = self._find_action_location(action_frame - 1,
                                pre_group_1_label, pre_group_2_label)
            if data_aug:
                self._prepare_aug_param(action_frame - offset - 1, 
                                        pre_group_1_label, pre_group_2_label)
                action_location = self._aug_transform(action_location)
            for idx, i in enumerate(range(action_frame - offset - seq_length, 
                                          action_frame - offset, interval)):
                if sim:
                    self.curr_robo_pos = self.robo_pos[idx]
                canvas = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
                canvas, pedidx = self._draw_canvas_handler(
                                    canvas, i, pre_group_1_label, data_aug, sim)
                tmp_pedidx = pedidx
                canvas, pedidx = self._draw_canvas_handler(
                                    canvas, i, pre_group_2_label, data_aug, sim)

                pedidx_seq.append(pedidx + tmp_pedidx)
                pedidx_action_seq.append((pedidx, tmp_pedidx))
                all_positions_seq.append(self.video_position_matrix[i])
                all_pedidx_seq.append(self.video_pedidx_matrix[i])
                frames_seq.append(i)
                img_seq.append(canvas)

            if debug:
                canvas = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
                canvas, _ = self._draw_canvas_handler(
                                canvas, action_frame, after_group_label, data_aug, sim=False)
                cv2.circle(canvas, action_location, 8, (255, 0, 0), 2)
                img_seq.append(canvas)
            if into_future:
                for i in range(action_frame, action_frame + future, interval):
                    canvas = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
                    canvas, _ = self._draw_canvas_handler(
                                    canvas, i, after_group_label, data_aug, sim=False)
                    img_seq.append(canvas)

        elif action == -1:
            pre_group_label = action_info[2]
            after_group_1_label = action_info[3]
            after_group_2_label = action_info[4]
            action_location = self._find_action_location(action_frame,
                                after_group_1_label, after_group_2_label)
            if data_aug:
                self._prepare_aug_param(action_frame - offset - 1, pre_group_label, -1)
                action_location = self._aug_transform(action_location)
            for idx, i in enumerate(range(action_frame - offset - seq_length, 
                                          action_frame - offset, interval)):
                if sim:
                    self.curr_robo_pos = self.robo_pos[idx]
                canvas = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
                canvas, pedidx = self._draw_canvas_handler(
                                    canvas, i, pre_group_label, data_aug, sim)

                pedidx_seq.append(pedidx)
                _, _, pedidx1 = self._find_label_properties(action_frame, after_group_1_label)
                _, _, pedidx2 = self._find_label_properties(action_frame, after_group_2_label)
                pedidx_action_seq.append((pedidx1, pedidx2))
                all_positions_seq.append(self.video_position_matrix[i])
                all_pedidx_seq.append(self.video_pedidx_matrix[i])
                frames_seq.append(i)
                img_seq.append(canvas)

            if debug:
                canvas = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
                canvas, _ = self._draw_canvas_handler(
                                canvas, action_frame, after_group_1_label, data_aug, sim=False)
                canvas, _ = self._draw_canvas_handler(
                                canvas, action_frame, after_group_2_label, data_aug, sim=False)
                cv2.circle(canvas, action_location, 8, (255, 0, 0), 2)
                img_seq.append(canvas)
            if into_future:
                for i in range(action_frame, action_frame + future, interval):
                    canvas = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
                    canvas, _ = self._draw_canvas_handler(
                                    canvas, i, after_group_1_label, data_aug, sim=False)
                    canvas, _ = self._draw_canvas_handler(
                                    canvas, i, after_group_2_label, data_aug, sim=False)
                    img_seq.append(canvas)

        else:
            raise Exception('Impossible action!')
            return

        return img_seq, action, action_location, all_positions_seq, all_pedidx_seq, pedidx_seq, frames_seq, pedidx_action_seq

    def random_positive_data(self, desired_action, 
                             is_test = False, into_future = False, sim = False):
        future = self.param['label_future_threshold']
        num_dynamics = len(self.video_dynamics_matrix)
        rand_idx = np.random.choice(num_dynamics)
        action_info = self.video_dynamics_matrix[rand_idx]
        while action_info[0] != desired_action:
            rand_idx = np.random.choice(num_dynamics)
            action_info = self.video_dynamics_matrix[rand_idx]
        if (into_future) and (future > 0):
            img_seq, action, action_location, _, _, _, _, _ = \
                self._generate_positive_datapoint(action_info, False, is_test, True, sim)
        else:
            img_seq, action, action_location, _, _, _, _, _ = \
                self._generate_positive_datapoint(action_info, False, is_test, False, sim)
        for i in range(len(img_seq)):
            if i == 0:
                self._prepare_process_image(img_seq)
            img_seq[i] = self._process_image(img_seq[i], debug = False)
            
        return np.array(img_seq), action, self._process_action_location(action_location)

    def compile_reverse_params(self):
        params = {'frame_width': self.frame_width,
                  'frame_height': self.frame_height,
                  'H': self.H,
                  'aug_trans': self.aug_trans,
                  'aug_angle': self.aug_angle,
                  'bbox_crop': self.bbox_crop,
                  'process_scale': self.process_scale,
                  'new_size': self.new_size
                 }
        return params

    def compile_positive_data(self, sim = False, data_aug = True):
        num_dynamics = len(self.video_dynamics_matrix)
        img_seq_data = []
        action_data = []
        location_data = []
        meter_location_data = []
        process_location_data = []
        position_seq_data = []
        pedidx_seq_data = []
        action_pedidx_seq_data = []
        twogroup_pedidx_seq_data = []
        frames_seq_data = []
        reverse_loc_params = []
        for i in range(num_dynamics):
            action_info = self.video_dynamics_matrix[i]
            img_seq, action, location, all_position_seq, all_pedidx_seq, pedidx_seq, frames_seq, twogroup_pedidx_seq = self._generate_positive_datapoint(action_info, False, data_aug, False, sim = sim)
            for j in range(len(img_seq)):
                if j == 0:
                    self._prepare_process_image(img_seq)
                img_seq[j] = self._process_image(img_seq[j])
            process_location = self._process_action_location(location)
            img_seq_data.append(img_seq)
            action_data.append(action)
            location_data.append(location)
            if data_aug:
                location = self._reverse_aug_transform(location)
            meter_location = self._reverse_coordinate_transform(location)
            meter_location_data.append(meter_location)
            reverse_loc_params.append(self.compile_reverse_params())
            process_location_data.append(process_location)
            position_seq_data.append(all_position_seq)
            pedidx_seq_data.append(all_pedidx_seq)
            action_pedidx_seq_data.append(pedidx_seq)
            frames_seq_data.append(frames_seq)
            twogroup_pedidx_seq_data.append(twogroup_pedidx_seq)
            
        save_dict = {'image_sequence': img_seq_data,
                     'action': action_data,
                     'location': location_data,
                     'meter_location': meter_location_data,
                     'processed_location': process_location_data,
                     'position_sequence': position_seq_data,
                     'pedidx_sequence': pedidx_seq_data,
                     'action_pedidx_sequence': action_pedidx_seq_data,
                     'twogroup_pedidx_sequence': twogroup_pedidx_seq_data,
                     'frames': frames_seq_data,
                     'reverse_loc_params': reverse_loc_params}
        pickle.dump(save_dict, open('test_data_complete/' + self.dataset + '_' + str(self.flag) + '_pos.p', 'wb'))
        return 

    def generate_positive_dataset(self, num_samples, debug = False, sim = False):
        merge_counter = 0
        split_counter = 0
        for i, action_info in enumerate(self.video_dynamics_matrix[:num_samples]):
            print([i, action_info, self.total_num_frames])
            action = action_info[0]
            action_frame = action_info[1]
            img_seq, action, action_location, _, _, _, _, _ =  \
                self._generate_positive_datapoint(action_info, debug, True, sim = sim)
            img_seq_debug, _, _, _, _, _, _, _ = \
                self._generate_positive_datapoint(action_info, debug, sim = sim)
            if action == 1:
                if self.merge_category_array[merge_counter] == 0:
                    purpose_dir = 'train_data/'
                else:
                    purpose_dir = 'test_data/'
                fpath = purpose_dir + '1/' + str(action_location[0]) + '_' + \
                        str(action_location[1]) + '_' + str(action_frame) + '_' + \
                        self.dataset + '_' + str(self.flag)
                os.mkdir(fpath)
                fname = fpath + '/'
                for j in range(len(img_seq)):
                    if j == 0:
                        self._prepare_process_image(img_seq)
                        action_location = self._process_action_location(action_location)
                    img = self._process_image(img_seq[j], True)
                    #if j == (len(img_seq) - 1):
                    #    cv2.circle(img, action_location, 16, (0, 255, 0), 2)
                    cv2.imwrite(fname + str(j) + '.jpg', img)
                    cv2.imwrite(fname + 'd_' + str(j) + '.jpg', img_seq_debug[j])
                merge_counter += 1
            elif action == -1:
                if self.split_category_array[split_counter] == 0:
                    purpose_dir = 'train_data/'
                else:
                    purpose_dir = 'test_data/'
                fpath = purpose_dir + '2/' + str(action_location[0]) + '_' + \
                        str(action_location[1]) + '_' + str(action_frame) + '_' + \
                        self.dataset + '_' + str(self.flag)
                os.mkdir(fpath)
                fname = fpath + '/'
                for j in range(len(img_seq)):
                    if j == 0:
                        self._prepare_process_image(img_seq)
                        action_location = self._process_action_location(action_location)
                    img = self._process_image(img_seq[j], True)
                    #if j == (len(img_seq) - 1):
                    #    cv2.circle(img, action_location, 16, (0, 255, 0), 2)
                    cv2.imwrite(fname + str(j) + '.jpg', img)
                    cv2.imwrite(fname + 'd_' + str(j) + '.jpg', img_seq_debug[j])
                split_counter += 1
            else:
                raise Exception('Impossible action!')
                return
        return 

    def check_group_integrity(self):
        for group_idx in range(self.num_groups):
            appearance_array = self._group_appearance_array(group_idx, faster_flag = False)
            for i in range(1, len(appearance_array)):
                if (appearance_array[i] - appearance_array[i - 1]) != 1:
                    print(group_idx)
                    print(appearance_array)
                    return False
                
        return True

    def _group_appearance_array(self, group_idx, faster_flag = True):
        appearance_array = []
        found_flag = False
        for i in range(self.total_num_frames):
            if group_idx in self.video_labels_matrix[i]:
                appearance_array.append(i)
                found_flag = True
            elif found_flag and faster_flag:
                break

        return appearance_array

    def _sample_group_frame(self):
        history = self.param['label_history_threshold']
        group = np.random.randint(self.num_groups)
        appearance_array = self._group_appearance_array(group)
        while (len(appearance_array) < history):
            group = np.random.randint(self.num_groups)
            appearance_array = self._group_appearance_array(group)
        frame = np.random.choice(appearance_array[(history - 1):]) + 1
        return group, frame

    def _another_complete_group(self, group, frame, nn_flag = False):
        history = self.param['label_history_threshold']
        offset = self.param['label_history_offset']
        complete_groups = set(self.video_labels_matrix[frame - history])
        for i in range(frame - history + 1, frame):
            complete_groups &= set(self.video_labels_matrix[i])
        if not (group in complete_groups):
            raise Exception('Group in question should be in the complete groups set!')
            return False, -1
        if len(complete_groups) <= 1:
            return False, -1
        complete_groups.remove(group)
        if nn_flag:
            min_dist = 5 # Important parameter
            group_positions, _, _ = self._find_label_properties(frame - offset - 1, group)
            group_center = self._find_centroid(group_positions)
            group_exist = False
            confirm_group = -1
            while (len(complete_groups) > 0):
                candidate_group = complete_groups.pop()
                candidate_group_positions, _, _ = self._find_label_properties(
                                               frame - offset - 1, candidate_group)
                candidate_group_center = self._find_centroid(candidate_group_positions)
                candidate_dist = math.sqrt((candidate_group_center[0] - group_center[0]) ** 2 + \
                                 (candidate_group_center[1] - group_center[1]) ** 2)
                if candidate_dist < min_dist:
                    min_dist = candidate_dist
                    group_exist = True
                    confirm_group = candidate_group
            return group_exist, confirm_group
        else:
            return True, complete_groups.pop()    

    def _generate_negative_datapoint(self, img_type, data_aug = False, sim = False):
        history = self.param['label_history_threshold']
        interval = self.param['label_history_interval']
        offset = self.param['label_history_offset']
        seq_length = self.param['label_history_seqlength']
        img_seq = []
        all_positions_seq = []
        all_pedidx_seq = []
        pedidx_seq = []
        frames_seq = []
        if img_type == 0:
            group_idx, frame_idx = self._sample_group_frame()
            if sim:
                self.robo_pos = self._generate_robot_location(frame_idx)
            if data_aug:
                self._prepare_aug_param(frame_idx - offset - 1, group_idx, -1)
            for idx, i in enumerate(range(frame_idx - offset - seq_length, 
                                          frame_idx - offset, interval)):
                if sim:
                    self.curr_robo_pos = self.robo_pos[idx]
                canvas = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
                canvas, pedidx = self._draw_canvas_handler(
                                    canvas, i, group_idx, data_aug, sim)

                pedidx_seq.append(pedidx)
                all_positions_seq.append(self.video_position_matrix[i])
                all_pedidx_seq.append(self.video_pedidx_matrix[i])
                frames_seq.append(i)
                img_seq.append(canvas)
        else:
            group_idx, frame_idx = self._sample_group_frame()
            group_exists, group_2_idx = self._another_complete_group(
                                        group_idx, frame_idx, nn_flag = True)
            while not group_exists:
                group_idx, frame_idx = self._sample_group_frame()
                group_exists, group_2_idx = self._another_complete_group(
                                            group_idx, frame_idx, nn_flag = True)
            if sim:
                self.robo_pos = self._generate_robot_location(frame_idx)
            if data_aug:
                self._prepare_aug_param(frame_idx - offset - 1, group_idx, group_2_idx)
            for idx, i in enumerate(range(frame_idx - offset - seq_length, 
                                          frame_idx - offset, interval)):
                if sim:
                    self.curr_robo_pos = self.robo_pos[idx]
                canvas = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
                canvas, pedidx = self._draw_canvas_handler(
                                    canvas, i, group_idx, data_aug, sim)
                tmp_pedidx = pedidx
                canvas, pedidx = self._draw_canvas_handler(
                                    canvas, i, group_2_idx, data_aug, sim)

                pedidx_seq.append(pedidx + tmp_pedidx)
                all_positions_seq.append(self.video_position_matrix[i])
                all_pedidx_seq.append(self.video_pedidx_matrix[i])
                frames_seq.append(i)
                img_seq.append(canvas)

        return img_seq, frame_idx, all_positions_seq, all_pedidx_seq, pedidx_seq, frames_seq

    def random_negative_data(self, is_test = False, sim = False):
        if np.random.rand() < 0.5:
            img_seq, _, _, _, _, _ = self._generate_negative_datapoint(0, is_test, sim)
        else:
            img_seq, _, _, _, _, _ = self._generate_negative_datapoint(1, is_test, sim)
        for i in range(len(img_seq)):
            if i == 0:
                self._prepare_process_image(img_seq)
            img_seq[i] = self._process_image(img_seq[i])
        return np.array(img_seq), 0, (127, 127)

    def compile_negative_data(self, num_sample, sim = False, data_aug = True):
        img_seq_data = []
        action_data = []
        location_data = []
        meter_location_data = []
        process_location_data = []
        position_seq_data = []
        pedidx_seq_data = []
        action_pedidx_seq_data = []
        frames_seq_data = []
        reverse_loc_params = []
        num_sample = int(round(num_sample))
        for i in range(num_sample):
            img_seq, _, all_position_seq, all_pedidx_seq, pedidx_seq, frames_seq = self._generate_negative_datapoint(i % 2, data_aug, sim = sim)
            for j in range(len(img_seq)):
                if j == 0:
                    self._prepare_process_image(img_seq)
                img_seq[j] = self._process_image(img_seq[j])
            img_seq_data.append(img_seq)
            action_data.append(0)
            location = (127, 127)
            location_data.append(location)
            if data_aug:
                location = self._reverse_aug_transform(location)
            meter_location = self._reverse_coordinate_transform(location)
            meter_location_data.append(meter_location)
            reverse_loc_params.append(self.compile_reverse_params())
            process_location_data.append((127, 127))
            position_seq_data.append(all_position_seq)
            pedidx_seq_data.append(all_pedidx_seq)
            action_pedidx_seq_data.append(pedidx_seq)
            frames_seq_data.append(frames_seq)
            
        save_dict = {'image_sequence': img_seq_data,
                     'action': action_data,
                     'location': location_data,
                     'meter_location': meter_location_data,
                     'processed_location': process_location_data,
                     'position_sequence': position_seq_data,
                     'pedidx_sequence': pedidx_seq_data,
                     'action_pedidx_sequence': action_pedidx_seq_data,
                     'twogroup_pedidx_sequence': action_pedidx_seq_data,
                     'frames': frames_seq_data,
                     'reverse_loc_params': reverse_loc_params}
        pickle.dump(save_dict, open('test_data_complete/' + self.dataset + '_' + str(self.flag) + '_neg.p', 'wb'))
        return

    def generate_negative_dataset(self, num_samples, sim = False):
        i = 0
        while i < num_samples:
            print([i, num_samples])
            if (i <= int(num_samples * self.train_ratio)):
                purpose_dir = 'train_data/'
            else:
                purpose_dir = 'test_data/'
            img_seq, frame_idx, _, _, _, _ = \
                self._generate_negative_datapoint(i % 2, True, sim = sim)
            fpath = purpose_dir + '0/' + str(i) + '_' + str(i) + \
                    '_' + str(frame_idx) + '_' + self.dataset + '_' + str(self.flag)
            if os.path.exists(fpath):
                continue
            os.mkdir(fpath)
            fname = fpath + '/'
            for j in range(len(img_seq)):
                if j == 0:
                    self._prepare_process_image(img_seq)
                img = self._process_image(img_seq[j], True)
                cv2.imwrite(fname + str(j) + '.jpg', img)
            i += 1
        return

    def _check_robot_collision(self, robo_pos, action_frame):
        interval = self.param['label_history_interval']
        offset = self.param['label_history_offset']
        seq_length = self.param['label_history_seqlength']
        min_robot_prox = 1.0
        
        for idx, i in enumerate(range(action_frame - offset - seq_length, 
                                      action_frame - offset, interval)):
            ped_pos = self.video_position_matrix[i]
            num_ped = len(ped_pos)
            for j in range(num_ped):
                pos = ped_pos[j]
                dist = math.sqrt((pos[0] - robo_pos[idx][0]) ** 2 + (pos[1] - robo_pos[idx][1]) ** 2)
                if dist < min_robot_prox:
                    return False
        return True

    def _generate_robot_location(self, action_frame):
        width = self.frame_width
        height = self.frame_height
        valid_robo_path = False
        frame_rate = 25
        frame_time = 1.0 / 25
        speed_range = 0
        interval = self.param['label_history_interval']
        seq_length = self.param['label_history_seqlength']
        frame_len = int(math.ceil(seq_length / interval))
        
        while not valid_robo_path:
            rand_x = int(np.random.rand() * width) 
            rand_y = int(np.random.rand() * height) 
            pt = np.matmul(self.H, [[rand_x], [rand_y], [1.0]])
            robo_pos_init = ((pt[0] / pt[2]), (pt[1] / pt[2]))
            robo_speed = np.random.rand() * speed_range
            robo_direction = np.random.rand() * 2 * np.pi

            robo_pos_interval = np.array(range(frame_len)) * frame_time * robo_speed
            robo_pos = np.array([robo_pos_interval * np.cos(robo_direction) + robo_pos_init[0],
                                 robo_pos_interval * np.sin(robo_direction) + robo_pos_init[1]])
            robo_pos = np.transpose(robo_pos)
            valid_robo_path = self._check_robot_collision(robo_pos, action_frame)
        return robo_pos

    def test_social_space_img(self, frame_idx, sim = False, name='a_', robo_pos = (0, 0)):
        if not self.has_video:
            frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        else:
            cap = cv2.VideoCapture(self.fname)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            _, frame = cap.read()
        cv2.imwrite('test_ss_img/' + name + 'original.jpg', frame)

        labels = self.video_labels_matrix[frame_idx]
        num_people = len(labels)
        counted = []
        canvas = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        for j in range(num_people):
            if not (labels[j] in counted):
                counted.append(labels[j])
                position_array, velocity_array, _ = self._find_label_properties(frame_idx, labels[j])
                if sim:
                    self.curr_robo_pos = robo_pos
                    frame = self._draw_simulated_social_shapes(frame, position_array, False, True)
                    canvas = self._draw_simulated_social_shapes(canvas, position_array, False, True)
                    x, y = self._coordinate_transform(robo_pos)
                    cv2.circle(frame, (y, x), 8, (255,0,0), 2)
                    cv2.circle(canvas, (y, x), 8, (255,0,0), 2)
                else:
                    frame = self._draw_social_shapes(
                                frame, position_array, velocity_array, False)
                    canvas = self._draw_social_shapes(
                                canvas, position_array, velocity_array, False)
            x = self.video_position_matrix[frame_idx][j][0]
            y = self.video_position_matrix[frame_idx][j][1]
            x, y = self._coordinate_transform((x, y))
            cv2.circle(frame, (y, x), 2, (0,0,255), 2)
            #cv2.circle(canvas, (y, x), 2, (0,0,255), 2)
        cv2.imwrite('test_ss_img/' + name + 'rst.jpg', frame)
        cv2.imwrite('test_ss_img/' + name + 'rst_b.jpg', canvas)

        return

    def test_grouping_on_video(self, name = 'a_'):
        i = 0
        num_colors = 14
        magnify_constant = 10
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(name + 'output.avi', fourcc, 25.0, 
                              (self.frame_width, self.frame_height))
        if self.has_video:
            cap = cv2.VideoCapture(self.fname)

        while(((not self.has_video) or (cap.isOpened())) and (i < self.total_num_frames)):
            if not self.has_video:
                frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
                ret = True
            else:
                ret, frame = cap.read()
            frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
            if (ret == True): # and (len(self.video_position_matrix[i]) > 0):
                labels = self.video_labels_matrix[i]
                num_people = len(labels)
                counted = []
                for j in range(num_people):
                    if not (labels[j] in counted):
                        counted.append(labels[j])
                        position_array, velocity_array, _ = self._find_label_properties(i, labels[j])
                        frame = self._draw_social_shapes(
                                    frame, position_array, velocity_array, False)

                for j in range(num_people):
                    x = self.video_position_matrix[i][j][0]
                    y = self.video_position_matrix[i][j][1]
                    x, y = self._coordinate_transform((x, y))
                    vx = self.video_velocity_matrix[i][j][0] * magnify_constant
                    vy = self.video_velocity_matrix[i][j][1] * magnify_constant
                    vx = int(round(vx))
                    vy = int(round(vy))
                    label = self.video_labels_matrix[i][j]
                    pedidx = self.video_pedidx_matrix[i][j]
                    
                    if label % num_colors == 13:
                        color = (0, 0, 0)
                    elif label % num_colors == 0:
                        color = (255, 0, 0)
                    elif label % num_colors == 1:
                        color = (0, 255, 0)
                    elif label % num_colors == 2:
                        color = (0, 0, 255)
                    elif label % num_colors == 3:
                        color = (255, 0, 255)
                    elif label % num_colors == 4:
                        color = (255, 255, 0)
                    elif label % num_colors == 5:
                        color = (0, 255, 255)
                    elif label % num_colors == 6:
                        color = (127, 127, 127)
                    elif label % num_colors == 7:
                        color = (127, 0, 0)
                    elif label % num_colors == 8:
                        color = (0, 127, 0)
                    elif label % num_colors == 9:
                        color = (0, 0, 127)
                    elif label % num_colors == 10:
                        color = (127, 0, 127)
                    elif label % num_colors == 11:
                        color = (127, 127, 0)
                    elif label % num_colors == 12:
                        color = (0, 127, 127)

                    cv2.circle(frame, (y, x), 10, color, 2)
                    if self.dataset == 'eth':
                        cv2.line(frame, (y, x), (y + vy, x + vx), color, 2)
                    else:
                        cv2.line(frame, (y, x), (y + vx, x - vy), color, 2)

                    history = 1
                    while (i - history >= 0) and (pedidx in self.video_pedidx_matrix[i - history]):
                        k = self.video_pedidx_matrix[i - history].index(pedidx)
                        x = self.video_position_matrix[i - history][k][0]
                        y = self.video_position_matrix[i - history][k][1]
                        x, y = self._coordinate_transform((x, y))
                        cv2.circle(frame, (y, x), 2, color, 2)
                        history += 1

                i += 1
                if i % 100 == 0:
                    print([i, self.total_num_frames])
                out.write(frame)
            elif ret == True:
                i += 1
            else:
                print('Read Ends!')
                break
        if self.has_video:
            cap.release()
        out.release()
        return

    def test_compiled_data_location(self, path, idx, name = 'test_location'):
        d = pickle.load(open(path))
        position = d['position_sequence'][idx][-1]
        action = d['action'][idx]
        location = d['location'][idx]
        meter_location = d['meter_location'][idx]
        frame_idx = d['frames'][idx][-1]
        if action == 1:
            frame_idx -= 1
        if not self.has_video:
            frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        else:
            cap = cv2.VideoCapture(self.fname)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            _, frame = cap.read()

        num_people = len(position)
        for i in range(num_people):
            pos = position[i]
            x, y = self._coordinate_transform(pos)
            cv2.circle(frame, (y, x), 10, (255, 0, 0), 2)
        x, y = self._coordinate_transform(meter_location)
        cv2.circle(frame, (x, y), 5, (255, 255, 255), 2)
        x = location[0]
        y = location[1]
        if action == -1:
            cv2.circle(frame, (x, y), 10, (0, 255, 0), 2)
        elif action == 1:
            cv2.circle(frame, (x, y), 10, (0, 0, 255), 2)
        cv2.imwrite('test_ss_img/' + name + '.jpg', frame)
        return
