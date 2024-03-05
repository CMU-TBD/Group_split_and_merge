from __future__ import print_function

import cv2
import numpy as np
import tensorflow as tf
import os
from os import listdir

class DataLoader(object):

	def __init__(self):

		self.candidate_seqs = np.transpose(np.load('../mnist_test_seq.npy'), (1, 0, 2, 3))
		print(len(self.candidate_seqs))
		self.train_idxes = np.random.permutation(len(self.candidate_seqs))
		self.num_train_data = len(self.candidate_seqs)

		self.train_pointer = 0

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
			for i, img in enumerate(seq):
				img_x_seq.append(img)
			x.append(img_x_seq)

		x = np.array(x) / 255.0
		x = x[:,:,::2,::2]
		x = np.expand_dims(x, axis=4)
		return x
