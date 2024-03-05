from __future__ import print_function

from PIL import Image
import numpy as np
import tensorflow as tf
import os
from os import listdir

class DataLoader(object):

	def __init__(self):

		self.candidate_seqs = self._load_sequence()
		print(len(self.candidate_seqs))
		self.train_idxes = np.random.permutation(len(self.candidate_seqs))
		self.num_train_data = len(self.candidate_seqs)

		self.train_pointer = 0

	def _load_sequence(self):
		path = '../train_data/'
		files = listdir(path)

		img_files = []
		for f in files:
			if (len(f) > 4) and (f[-4:] == '.jpg'):
				img_files.append(f)

		num_data = 0
		min_index = 1000
		for im_name in img_files:
			im_info = im_name.split('_')
			idx = int(im_info[0])
			if (idx > num_data):
				num_data = idx
			if (idx < min_index):
				min_index = idx

		model_category = 'no'
		data_seqs = []
		for i in range(num_data - min_index + 1):
			data_seqs.append([])
		seq_size = 20
		for i in range(len(img_files)):
			im_name = img_files[i]
			im_info = im_name.split('_')
			if (im_info[3] == model_category) or (im_info[2] == model_category):
				seq = data_seqs[int(im_info[0]) - min_index]
				seq.append(im_name)

		candidate_seqs = []
		for seq in data_seqs:
			if len(seq) > 0:
				cand_seq = [''] * seq_size
				for im_name in seq:
					im_info = im_name.split('_')
					idx = im_info[-1]
					idx = int(idx[:-4])
					cand_seq[idx % seq_size] = path + im_name
				candidate_seqs.append(cand_seq)
		return candidate_seqs

	def get_seq(self, batch_size):
		if (self.train_pointer + batch_size) < self.num_train_data:
			idxes = range(self.train_pointer, self.train_pointer + batch_size)
			self.train_pointer += batch_size
		else:
			idxes = range(self.train_pointer, self.num_train_data)
			new_end = batch_size - len(idxes)
			idxes += range(new_end)
			self.train_pointer = new_end
			self.train_idxes = np.random.permutation(len(self.candidate_seqs))
		
		x = []
		for idx in idxes:
			seq = self.candidate_seqs[self.train_idxes[idx]]
			img_x_seq = []
			for i, name in enumerate(seq):
				img = Image.open(name)
				img = img.convert('L')
				img = img.resize((32, 32))
				img_x_seq.append(np.array(img))
			x.append(img_x_seq)

		x = np.array(x) / 255.0
		x = np.expand_dims(x, axis=4)
		return x
