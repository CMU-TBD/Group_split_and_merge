from __future__ import print_function

from PIL import Image
import numpy as np
import tensorflow as tf
import os
from os import listdir
import cv2

class DataLoader(object):

	def __init__(self):
		self.path = '../train_videos_exp/' 
		self.flag = 'no'
		self.train_ratio = 0.8
		self.sample_number = 500

		self.seq_length = 16
		self.start_offset = self.seq_length - 10
		self._load_video_info()
		self.candidate_img_seqs, self.candidate_action_seqs, self.candidate_location_seqs = \
			self._load_positive_sequence()
		print(len(self.candidate_action_seqs))
		self.num_positive_data = len(self.candidate_action_seqs)
		self.num_train_data = int(round(self.num_positive_data * self.train_ratio))
		self.positive_idxes = np.random.permutation(self.num_positive_data)

		self.test_img_seqs = []
		self.test_action_seqs = []
		self.test_location_seqs = []
		self.train_idxes = []
		for i in range(self.num_positive_data):
			idx = self.positive_idxes[i]
			if i < self.num_train_data:
				self.train_idxes.append(idx)
			else:
				self.test_img_seqs.append(self.candidate_img_seqs[idx, :, :, :, :])
                		self.test_action_seqs.append(self.candidate_action_seqs[idx])
          			self.test_location_seqs.append(self.candidate_location_seqs[idx, :])

		self.split_idxes = []
		self.merge_idxes = []
		for idx in self.train_idxes:
			action = self.candidate_action_seqs[idx]
			if action == -1:
				self.split_idxes.append(idx)
			elif action == 1:
				self.merge_idxes.append(idx)
			else:
				print('Error in actions')
				return

		self.num_positive_train_data = self.num_train_data
		self.num_positive_test_data = self.num_positive_data - self.num_positive_train_data

		num_data = self.num_positive_test_data / 2
		candidate_negative_seqs = self._add_negative_sequences(num_data, is_test=True)
		self.final_test_img_seqs = np.concatenate((self.test_img_seqs, 
						           candidate_negative_seqs), axis=0)
		self.final_test_action_seqs = np.concatenate((self.test_action_seqs,
							      np.zeros(num_data)), axis=0)
		self.final_test_location_seqs = np.concatenate((self.test_location_seqs,
						                np.zeros((num_data, 2))), axis=0)
		
		self._resample_data()
		self.train_pointer = 0

		return

	def _load_video_info(self):
		files = listdir(self.path)
		valid_txt_files = []
		for f in files:
			f_split = f.split('_')
			f_flag = f_split[-1]
			if (len(f) > 4) and (f[-4:] == '.txt') and (f_flag[:-4] == self.flag):
				print(f)
				valid_txt_files.append(f)

		# get the essential split and merge info from the txt files
		self.vid_idxes = []
		self.vid_info = []
		for fi in valid_txt_files:
			f = open(self.path + fi)
			data_name = fi.split('_')
			data_name = data_name[2]
			line = f.readline()
			while line <> '':
				vid_idx = int(line)
				frame_idxes = []
				split_or_merge = []
				locations = []
				line = f.readline()
				line_info = line.split()
				while len(line_info) == 4:
					frame_idxes.append(int(line_info[0]) - self.start_offset)
					split_or_merge.append(int(line_info[1]))
					locations.append((int(line_info[2]), int(line_info[3])))
					line = f.readline()
					line_info = line.split()
				self.vid_idxes.append(vid_idx)
				self.vid_info.append({'data_name' : data_name,
						      'frame_idxes': frame_idxes,
						      'actions': split_or_merge,
						      'locations': locations,
							  'test_idxes': []})
			f.close()
		return

	def _load_positive_sequence(self):
		# read the 10 frames before each split/merge point
		candidate_img_seqs = []
		candidate_action_seqs = []
		candidate_location_seqs = []
		for i in range(len(self.vid_idxes)):
			if (i % 100) == 0:
				print(str(i) + '/' + str(len(self.vid_idxes)))
			vid_idx = self.vid_idxes[i]
			data_name = self.vid_info[i]['data_name']
			frame_idxes = self.vid_info[i]['frame_idxes']
			split_or_merge = self.vid_info[i]['actions']
			locations = self.vid_info[i]['locations']
			cap = cv2.VideoCapture(self.path + data_name + '_' + self.flag + '_' + 
					       str(vid_idx) + '.avi')
			for current_idx, current_frame in enumerate(frame_idxes):
				vid_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
				if not self._is_valid_start_frame(current_frame, vid_num_frames, []):
					continue
				cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
				img_sequence = []
				for j in range(self.seq_length):
					ret, frame = cap.read()
					while (ret == False):
						print(str(cap.get(cv2.CAP_PROP_FRAME_COUNT)) + 
								'/' + str(vid_num_frames))
						print('Read Failure!(1)')
						ret, frame = cap.read()
					img = self._process_img(frame)
					img_sequence.append(img)
				img_sequence = np.expand_dims(np.array(img_sequence), axis=3)
				candidate_img_seqs.append(img_sequence)
				candidate_action_seqs.append(split_or_merge[current_idx])
				candidate_location_seqs.append(locations[current_idx])

			cap.release()
		return np.array(candidate_img_seqs), np.array(candidate_action_seqs), \
		       np.array(candidate_location_seqs)

	def _add_negative_sequences(self, samples_needed, is_test=False):
		num_videos = len(self.vid_idxes)
		candidate_negative_seqs = []
		i = 0
		while i < samples_needed:
			if (i % 100) == 0:
				print(str(i) + '/' + str(samples_needed))
			rand_idx = np.random.randint(num_videos)
			vid_idx = self.vid_idxes[rand_idx]
			data_name = self.vid_info[rand_idx]['data_name']
			frame_idxes = self.vid_info[rand_idx]['frame_idxes']
			if not is_test:
				test_idxes = self.vid_info[rand_idx]['test_idxes']
				frame_idxes += test_idxes
			cap = cv2.VideoCapture(self.path + data_name + '_' + self.flag + '_' +
					       str(vid_idx) + '.avi')
			vid_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
			if vid_num_frames < self.seq_length:
				continue
			start_frame = np.random.randint(vid_num_frames)
			while not self._is_valid_start_frame(start_frame, vid_num_frames, 
							     frame_idxes):
				start_frame = np.random.randint(vid_num_frames)
			if is_test:
				self.vid_info[rand_idx]['test_idxes'].append(start_frame)
			cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

			img_sequence = []
			for j in range(self.seq_length):
				ret, frame = cap.read()
				while (ret == False):
					print('Read Failure!(2)')
					ret, frame = cap.read()
				img = self._process_img(frame)
				img_sequence.append(img)
			img_sequence = np.expand_dims(np.array(img_sequence), axis=3)
			candidate_negative_seqs.append(img_sequence)
			i += 1
			cap.release()

		return np.array(candidate_negative_seqs)

	def _is_valid_start_frame(self, start_frame, total_frame, frame_idxes):
		if start_frame > (total_frame - self.seq_length):
			return False
		if start_frame < 0:
			return False
		for frame_idx in frame_idxes:
			if start_frame == frame_idx:
				return False
		return True

	def _process_img(self, frame):
		im = Image.fromarray(np.uint8(frame))
		im = im.resize((224, 224))
		im = np.array(im) / 255.0
		return im[:,:,0]

	def _resample_data(self):
		sub_split_idxes = np.random.permutation(len(self.split_idxes))
		sub_merge_idxes = np.random.permutation(len(self.merge_idxes))
		candidate_positive_img_seqs = []
		candidate_positive_action_seqs = []
		candidate_positive_location_seqs = []
		for i in range(self.sample_number):
			candidate_positive_img_seqs.append(
				self.candidate_img_seqs[self.split_idxes[sub_split_idxes[i]]])
			candidate_positive_action_seqs.append(
				self.candidate_action_seqs[self.split_idxes[sub_split_idxes[i]]])
			candidate_positive_location_seqs.append(
				self.candidate_location_seqs[self.split_idxes[sub_split_idxes[i]]])
			candidate_positive_img_seqs.append(
				self.candidate_img_seqs[self.merge_idxes[sub_merge_idxes[i]]])
			candidate_positive_action_seqs.append(
				self.candidate_action_seqs[self.merge_idxes[sub_merge_idxes[i]]])
			candidate_positive_location_seqs.append(
				self.candidate_location_seqs[self.merge_idxes[sub_merge_idxes[i]]])
		candidate_positive_img_seqs = np.array(candidate_positive_img_seqs)
		candidate_positive_action_seqs = np.array(candidate_positive_action_seqs)
		candidate_positive_location_seqs = np.array(candidate_positive_location_seqs)

		num_data = self.sample_number
		candidate_negative_seqs = self._add_negative_sequences(num_data)
		self.final_img_seqs = np.concatenate((candidate_positive_img_seqs, 
						     candidate_negative_seqs), axis=0)
		self.final_action_seqs = np.concatenate((candidate_positive_action_seqs,
							np.zeros(num_data)), axis=0)
		self.final_location_seqs = np.concatenate((candidate_positive_location_seqs,
						np.zeros((num_data, 2))), axis=0)
		self.num_train_data = np.shape(self.final_img_seqs)[0]
		self.train_idxes = np.random.permutation(self.num_train_data)

		num_data = self.num_positive_test_data / 2
		self.validation_img_seqs = self.final_img_seqs[:(num_data * 2), :, :, :, :]
		self.validation_action_seqs = self.final_action_seqs[:(num_data * 2)]
		self.validation_location_seqs = self.final_location_seqs[:(num_data * 2), :]
		self.validation_img_seqs = np.concatenate((self.validation_img_seqs, 
					     candidate_negative_seqs[:num_data, :, :, :, :]), axis=0)
		self.validation_action_seqs = np.concatenate((self.validation_action_seqs,
							np.zeros(num_data)), axis=0)
		self.validation_location_seqs = np.concatenate((self.validation_location_seqs,
						np.zeros((num_data, 2))), axis=0)

		return

	def _action_mapping(self, action):
		if action == -1:
			return [0, 1, 0]
		elif action == 0:
			return [1, 0, 0]
		elif action == 1:
			return [0, 0, 1]
		else:
			print('Warning: action cannot be other values!')
			return [0, 0, 1]

	def get_seq(self, batch_size):
		if (self.train_pointer + batch_size) < self.num_train_data:
			idxes = range(self.train_pointer, self.train_pointer + batch_size)
			self.train_pointer += batch_size
		else:
			idxes = range(self.train_pointer, self.num_train_data)
			new_end = batch_size - len(idxes)
			idxes += range(new_end)
			self.train_pointer = new_end
			self._resample_data()
		
		img_seqs = []
		actions = []
		locations = []
		for idx in idxes:
			seq = self.final_img_seqs[self.train_idxes[idx]]
			action = self.final_action_seqs[self.train_idxes[idx]]
			location = self.final_location_seqs[self.train_idxes[idx]]
			img_seqs.append(seq)
			actions.append(self._action_mapping(action))
			locations.append(location)

			#img_x_seq = []
			#for i, name in enumerate(seq):
			#	img = Image.open(name)
			#	img = img.convert('L')
			#	img = img.resize((32, 32))
			#	img_x_seq.append(np.array(img))
			#x.append(img_x_seq)

		x = {'img_seq': np.array(img_seqs), 
		     'action': np.array(actions), 
		     'location': np.array(locations)}
		return x

	def get_test_samples(self):
		action_seq = []
		for action in self.final_test_action_seqs:
			action_seq.append(self._action_mapping(action))
		x = {'img_seq': self.final_test_img_seqs, 
		     'action': np.array(action_seq), 
		     'location': self.final_test_location_seqs}
		return x
		
	def get_validation_samples(self):
		action_seq = []
		for action in self.validation_action_seqs:
			action_seq.append(self._action_mapping(action))
		x = {'img_seq': self.validation_img_seqs, 
		     'action': np.array(action_seq), 
		     'location': self.validation_location_seqs}
		return x

	def test_positive_videos(self):
		num_test = 10
		time_limit = 20
		extra_num_frame = 5
		output_path = 'validation_vid/'
		fourcc = cv2.VideoWriter_fourcc(*'MJPG')
		for i in range(num_test):
			choice_idx = np.random.choice(np.array(range(len(self.vid_idxes))))
                        frame_idxes = self.vid_info[choice_idx]['frame_idxes']
			while (len(frame_idxes) == 0):
				choice_idx = np.random.choice(np.array(range(len(self.vid_idxes))))
                        	frame_idxes = self.vid_info[choice_idx]['frame_idxes']
				
			vid_idx = self.vid_idxes[choice_idx]
                        data_name = self.vid_info[choice_idx]['data_name']
                        split_or_merge = self.vid_info[choice_idx]['actions']
                        locations = self.vid_info[choice_idx]['locations']
                        cap = cv2.VideoCapture(self.path + data_name + '_' + self.flag + '_' +
                                               str(vid_idx) + '.avi')
			vid_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
			width = int(cap.get(3))
			height = int(cap.get(4))
                        out = cv2.VideoWriter(output_path + str(i) + '.avi', 
						fourcc, 10.0, (width, height))
			sub_idx = np.random.choice(np.array(range(len(frame_idxes))))
			current_frame = frame_idxes[sub_idx]
			time_out_counter = 0
			while (not self._is_valid_start_frame(current_frame, 
				vid_num_frames - extra_num_frame, [])) and \
				(time_out_counter < time_limit):
				sub_idx = np.random.choice(np.array(range(len(frame_idxes))))
				current_frame = frame_idxes[sub_idx]
				time_out_counter += 1
			if time_out_counter == time_limit:
				continue
			action = split_or_merge[sub_idx]
			location = locations[sub_idx]
			cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
			for j in range(self.seq_length + extra_num_frame):
				ret, frame = cap.read()
				while (ret == False):
					print(str(cap.get(cv2.CAP_PROP_FRAME_COUNT)) + 
							'/' + str(vid_num_frames))
					print('Read Failure!(3)')
					ret, frame = cap.read()
				if j == self.seq_length:
					if action == -1:
						color = (255, 0, 0)
					elif action == 1:
						color = (0, 0, 255)
					else:
						print('Wierd things happened!')
						color = (0, 0, 0)
					cv2.circle(frame, location, 8, color, 2)	
				out.write(frame)

			out.release()
			cap.release()

		return
