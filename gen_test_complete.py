"""
Code used to precompute splits and merges for baseline codes.
Only pickle data generated. Convert to numpy when needed.
"""

from __future__ import print_function

import numpy as np
import os, sys
from os import listdir

import Social_grouping as sg

sim = True
data_aug = True

a = sg.SocialGrouping(dataset = 'eth', flag = 0)
b = sg.SocialGrouping(dataset = 'eth', flag = 1)
c = sg.SocialGrouping(dataset = 'ucy', flag = 0)
d = sg.SocialGrouping(dataset = 'ucy', flag = 1)
e = sg.SocialGrouping(dataset = 'ucy', flag = 2)
#f = sg.SocialGrouping(dataset = 'ucy', flag = 3)
#g = sg.SocialGrouping(dataset = 'ucy', flag = 4)

num_split = 0
num_merge = 0
num_split += a.num_split
num_split += b.num_split
num_split += c.num_split
num_split += d.num_split
num_split += e.num_split
num_merge += a.num_merge
num_merge += b.num_merge
num_merge += c.num_merge
num_merge += d.num_merge
num_merge += e.num_merge
print('total number of split: ' + str(num_split))
print('total number of merge: ' + str(num_merge))

a.compile_positive_data(sim = sim, data_aug = data_aug)
a.compile_negative_data(int(a.num_split + a.num_merge)/2, sim = sim, data_aug = data_aug)
print('a')

b.compile_positive_data(sim = sim, data_aug = data_aug)
b.compile_negative_data(int(b.num_split + b.num_merge)/2, sim = sim, data_aug = data_aug)
print('b')

c.compile_positive_data(sim = sim, data_aug = data_aug)
c.compile_negative_data(int(c.num_split + c.num_merge)/2, sim = sim, data_aug = data_aug)
print('c')

d.compile_positive_data(sim = sim, data_aug = data_aug)
d.compile_negative_data(int(d.num_split + d.num_merge)/2, sim = sim, data_aug = data_aug)
print('d')

e.compile_positive_data(sim = sim, data_aug = data_aug)
e.compile_negative_data(int(e.num_split + e.num_merge)/2, sim = sim, data_aug = data_aug)
print('e')

#f.compile_positive_data()
#f.compile_negative_data()
#print('f')

#g.compile_positive_data()
#g.compile_negative_data()
#print('g')
