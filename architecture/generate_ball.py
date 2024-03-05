import os
import os.path
import time

import numpy as np
import tensorflow as tf
import cv2

import bouncing_balls as b
import layer_def as ld
import BasicConvLSTMCell

def generate_bouncing_ball_sample(batch_size, seq_length, shape, num_balls):
  dat = np.zeros((batch_size, seq_length, shape, shape, 3))
  for i in xrange(batch_size):
    print(i)
    dat[i, :, :, :, :] = b.bounce_vec(32, num_balls, seq_length)
  return dat

seq = generate_bouncing_ball_sample(1, 10, 32, 2)
print(seq)
print(np.max(seq))
#np.save('bouncing_ball_seq.npy', seq)
