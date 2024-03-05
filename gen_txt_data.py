import csv
import numpy as np
import os, sys
from os import listdir
from sets import Set

import Social_grouping as sg

def prepare_txt_data(class_data):

    skip = 2

    positions = class_data.video_position_matrix
    idxes = class_data.video_pedidx_matrix
    
    frames_arr = []
    ped_idx_arr = []
    pos_x_arr = []
    pos_y_arr = []

    for i in range(0, len(positions), skip):
        num_people = len(idxes[i])
        if num_people > 0:
            for j in range(num_people):
                frames_arr.append(i + 1)
                ped_idx_arr.append(idxes[i][j] + 1)
                pos_x_arr.append(positions[i][j][0])
                pos_y_arr.append(positions[i][j][1])

    return frames_arr, ped_idx_arr, pos_x_arr, pos_y_arr

def revise_pedidx(frames_arr, ped_idx_arr):
    skip = 2
    arr_length = len(frames_arr)
    for i in range(arr_length):
        max_ped_id = max(ped_idx_arr)
        cur_frame_id = frames_arr[i]
        cur_ped_id = ped_idx_arr[i]

        check_frame_id = cur_frame_id - skip
        frame_visit = []
        for j in range(i):
            if ped_idx_arr[j] == cur_ped_id:
                frame_visit.append(frames_arr[j])
        if len(frame_visit) == 0:
            continue
        if frame_visit[-1] == check_frame_id:
            continue

        for j in range(i, arr_length):
            if ped_idx_arr[j] == cur_ped_id:
                ped_idx_arr[j] = max_ped_id + 1

    return ped_idx_arr

a = sg.SocialGrouping(dataset = 'eth', flag = 0)
b = sg.SocialGrouping(dataset = 'eth', flag = 1)
c = sg.SocialGrouping(dataset = 'ucy', flag = 0)
d = sg.SocialGrouping(dataset = 'ucy', flag = 1)
e = sg.SocialGrouping(dataset = 'ucy', flag = 2)

class_list = [a, b, c, d, e]
names_list = ['eth', 'hotel', 'zara1', 'zara2', 'univ']

for idx, c in enumerate(class_list):
    frames_arr, ped_idx_arr, pos_x_arr, pos_y_arr = prepare_txt_data(c)
    num_dynamics = len(c.video_dynamics_matrix)
    frames_in_action = Set([])
    for i in range(num_dynamics):
        action_info = c.video_dynamics_matrix[i]
        action_frame = action_info[1]
        for j in range(action_frame - 30, action_frame + 16):
            frames_in_action |= Set([j])
    frames_arr_a = []
    ped_idx_arr_a = []
    pos_x_arr_a = []
    pos_y_arr_a = []
    for i in range(len(frames_arr)):
        if frames_arr[i] in frames_in_action:
            frames_arr_a.append(frames_arr[i])
            ped_idx_arr_a.append(ped_idx_arr[i])
            pos_x_arr_a.append(pos_x_arr[i])
            pos_y_arr_a.append(pos_y_arr[i])

    ped_idx_arr_a = revise_pedidx(frames_arr_a, ped_idx_arr_a)

    total_length = len(frames_arr)
    action_length = len(frames_arr_a)
    num_copy = 2 ** (int(round(total_length * 1.0 / action_length)) - 1) - 1

    dt_name = 'txt_data/srlstm/' + names_list[idx]
    with open(dt_name + '_true_pos_.csv', 'wb') as f:
        writer = csv.writer(f, delimiter = ',')
        writer.writerow(frames_arr)
        writer.writerow(ped_idx_arr)
        writer.writerow(pos_x_arr)
        writer.writerow(pos_y_arr)
    for i in range(num_copy):
        with open(dt_name + '_action_' + str(i) + '.csv', 'wb') as f:
            writer = csv.writer(f, delimiter = ',')
            writer.writerow(frames_arr_a)
            writer.writerow(ped_idx_arr_a)
            writer.writerow(pos_x_arr_a)
            writer.writerow(pos_y_arr_a)
        
    dt_name = 'txt_data/slstm/' + names_list[idx]
    pos_x_arr_pix = []
    pos_y_arr_pix = []
    pos_x_arr_a_pix = []
    pos_y_arr_a_pix = []
    for i in range(total_length):
        coord_x = pos_x_arr[i]
        coord_y = pos_y_arr[i]
        x, y = c._coordinate_transform((coord_x, coord_y))
        pos_x_arr_pix.append(x * 1.0 / c.frame_height)
        pos_y_arr_pix.append(y * 1.0 / c.frame_width)
    for i in range(action_length):
        coord_x = pos_x_arr_a[i]
        coord_y = pos_y_arr_a[i]
        x, y = c._coordinate_transform((coord_x, coord_y))
        pos_x_arr_a_pix.append(x * 1.0 / c.frame_height)
        pos_y_arr_a_pix.append(y * 1.0 / c.frame_width)
    with open(dt_name + '_complete.csv', 'wb') as f:
        writer = csv.writer(f, delimiter = ',')
        writer.writerow(frames_arr)
        writer.writerow(ped_idx_arr)
        writer.writerow(pos_x_arr_pix)
        writer.writerow(pos_y_arr_pix)
    for i in range(num_copy):
        with open(dt_name + '_action_' + str(i)  + '.csv', 'wb') as f:
            writer = csv.writer(f, delimiter = ',')
            writer.writerow(frames_arr_a)
            writer.writerow(ped_idx_arr_a)
            writer.writerow(pos_x_arr_a_pix)
            writer.writerow(pos_y_arr_a_pix)

    dt_name = 'txt_data/sgan/' + names_list[idx]
    with open(dt_name + '_complete.txt', 'wb') as f:
        for i in range(total_length):
            msg = str(frames_arr[i]) + ' ' + str(ped_idx_arr[i]) + ' ' \
                  + str(pos_x_arr[i]) + ' ' + str(pos_y_arr[i]) + '\n'
            f.write(msg)
    for j in range(num_copy):
        with open(dt_name + '_action_' + str(j) + '.txt', 'wb') as f:
            for i in range(action_length):
                msg = str(frames_arr_a[i]) + ' ' + str(ped_idx_arr_a[i]) + ' ' \
                      + str(pos_x_arr_a[i]) + ' ' + str(pos_y_arr_a[i]) + '\n'
                f.write(msg)


