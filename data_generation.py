"""
Code used to check number of splits and merges. Also precompute positive 
and negative datasets. (Actual code generate data in real time)
"""

import Social_grouping as sg

print('a')
a = sg.SocialGrouping(dataset = 'eth', flag = 0)
#a.test_grouping_on_video('a_')
#a.generate_positive_dataset(True)
print(a.check_group_integrity())
#a.generate_negative_dataset((a.num_merge + a.num_split) / 2)

print('b')
b = sg.SocialGrouping(dataset = 'eth', flag = 1)
#b.test_grouping_on_video('b_')
#b.generate_positive_dataset(True)
print(b.check_group_integrity())
#b.generate_negative_dataset((b.num_merge + b.num_split) / 2)

print('c')
c = sg.SocialGrouping(dataset = 'ucy', flag = 0)
#c.test_grouping_on_video('c_')
#c.generate_positive_dataset(True)
print(c.check_group_integrity())
#c.generate_negative_dataset((c.num_merge + c.num_split) / 2)

print('d')
d = sg.SocialGrouping(dataset = 'ucy', flag = 1)
#d.test_grouping_on_video('d_')
#d.generate_positive_dataset(True)
print(d.check_group_integrity())
#d.generate_negative_dataset((d.num_merge + d.num_split) / 2)

print('e')
e = sg.SocialGrouping(dataset = 'ucy', flag = 2)
#e.test_grouping_on_video('e_')
#e.generate_positive_dataset(True)
print(e.check_group_integrity())
#e.generate_negative_dataset((e.num_merge + e.num_split) / 2)

print('f')
f = sg.SocialGrouping(dataset = 'ucy', flag = 3)
#f.test_grouping_on_video('f_')
#f.generate_positive_dataset(True)
print(f.check_group_integrity())
#f.generate_negative_dataset((f.num_merge + f.num_split) / 2)

print('g')
g = sg.SocialGrouping(dataset = 'ucy', flag = 4)
#g.test_grouping_on_video('g_')
#g.generate_positive_dataset(True)
print(g.check_group_integrity())
#g.generate_negative_dataset((g.num_merge + g.num_split) / 2)

print([a.num_merge, a.num_split, a.num_merge + a.num_split])
print([b.num_merge, b.num_split, b.num_merge + b.num_split])
print([c.num_merge, c.num_split, c.num_merge + c.num_split])
print([d.num_merge, d.num_split, d.num_merge + d.num_split])
print([e.num_merge, e.num_split, e.num_merge + e.num_split])
print([f.num_merge, f.num_split, f.num_merge + f.num_split])
print([g.num_merge, g.num_split, g.num_merge + g.num_split])
