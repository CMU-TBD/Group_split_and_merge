import csv
import pickle
import numpy as np

def prepare_txt_data(d):
    pedidx_data = d['pedidx_sequence']
    action_pedidx_data = d['action_pedidx_sequence']
    positions_data = d['position_sequence']
    action_data = d['action']
    location_data = d['location']
    frames_data = d['frames']

    frames_arr = []
    ped_idx_arr = []
    pos_x_arr = []
    pos_y_arr = []
    fix_arr = []

    skip = 2
    num_data = len(frames_data)
    for i in range(num_data):
        pedidx = pedidx_data[i]
        positions = positions_data[i]
        frames = frames_data[i]
        seq_len = len(frames)
        for j in range(seq_len):
            frame = frames[j]
            #if frame % 2 == 0:
            #    frame += 1
            pedidx_detail = pedidx[j]
            positions_detail = positions[j]
            num_people = len(pedidx_detail)
            for k in range(num_people):
                frames_arr.append(frame)
                ped_idx_arr.append(pedidx_detail[k])
                pos_x_arr.append(positions_detail[k][0])
                pos_y_arr.append(positions_detail[k][1])
                fix_arr.append(0)

        for j in range(seq_len + 1):
            frame = frames[seq_len - 1] + skip * (j + 1)
            #if frame % 2 == 0:
            #    frame += 1
            pedidx_detail = pedidx[seq_len - 1]
            positions_detail = positions[seq_len - 1]
            num_people = len(pedidx_detail)
            for k in range(num_people):
                frames_arr.append(frame)
                ped_idx_arr.append(pedidx_detail[k])
                pos_x_arr.append(positions_detail[k][0])
                pos_y_arr.append(positions_detail[k][1])
                fix_arr.append(1)

    sort_idxes = np.argsort(frames_arr)
    num_points = len(sort_idxes)
    frames_arr_real = [0] * num_points
    ped_idx_arr_real = [0] * num_points
    pos_x_arr_real = [0] * num_points
    pos_y_arr_real = [0] * num_points
    fix_arr_real = [0] * num_points
    for i in range(num_points):
        idx = sort_idxes[i]
        frames_arr_real[i] = frames_arr[idx]
        ped_idx_arr_real[i] = ped_idx_arr[idx]
        pos_x_arr_real[i] = pos_x_arr[idx]
        pos_y_arr_real[i] = pos_y_arr[idx]
        fix_arr_real[i] = fix_arr[idx]

    frames_odd = []
    ped_idx_odd = []
    pos_x_odd = []
    pos_y_odd = []
    fix_odd = []
    frames_even = []
    ped_idx_even = []
    pos_x_even = []
    pos_y_even = []
    fix_even = []
    for i in range(num_points):
        if frames_arr_real[i] % 2 == 1:
            frames_odd.append(frames_arr_real[i])
            ped_idx_odd.append(ped_idx_arr_real[i])
            pos_x_odd.append(pos_x_arr_real[i])
            pos_y_odd.append(pos_y_arr_real[i])
            fix_odd.append(fix_arr_real[i])
        else:
            frames_even.append(frames_arr_real[i] + 1)
            ped_idx_even.append(ped_idx_arr_real[i])
            pos_x_even.append(pos_x_arr_real[i])
            pos_y_even.append(pos_y_arr_real[i])
            fix_even.append(fix_arr_real[i])

    return frames_odd, ped_idx_odd, pos_x_odd, pos_y_odd, frames_even, ped_idx_even, pos_x_even, pos_y_even, fix_odd, fix_even

def filter_patches(frames_arr, ped_idx_arr, pos_x_arr, pos_y_arr, fix_arr):
    num_data = len(frames_arr)
    del_idxes = []
    for i in range(num_data):
        frame = frames_arr[i]
        ped_idx = ped_idx_arr[i]
        if fix_arr[i] == 1:
            j1 = i - 1
            j2 = i + 1
            found = False
            while (0 <= j1) and (frames_arr[j1] == frame):
                if (ped_idx_arr[j1] == ped_idx) and (fix_arr[j1] == 0):
                    found = True
                    break
                j1 -= 1
            if not found:
                while (j2 < num_data) and (frames_arr[j2] == frame):
                    if (ped_idx_arr[j2] == ped_idx) and (fix_arr[j2] == 0):
                        found = True
                        break
                    j2 += 1
            if found:
                del_idxes.append(1)
            else:
                del_idxes.append(0)
        else:
            del_idxes.append(0)
    i = 0
    while (i < len(del_idxes)):
        if del_idxes[i] == 1:
            del del_idxes[i]
            del frames_arr[i]
            del ped_idx_arr[i]
            del pos_x_arr[i]
            del pos_y_arr[i]
        else:
            i += 1
    return frames_arr, ped_idx_arr, pos_x_arr, pos_y_arr

def revise_pedidx_2(frames_arr, ped_idx_arr):
    i = 0
    skip = 2
    num_points = len(frames_arr)
    max_ped_id = max(ped_idx_arr)
    while (i <= max_ped_id):
        frames = []
        ped_locations = []
        for j in range(num_points):
            if ped_idx_arr[j] == i:
                frames.append(frames_arr[j])
                ped_locations.append(j)
        if len(frames) == 0:
            i += 1
            continue
        stopping_points = []
        prev_frame = -1
        for j in range(len(frames)):
            f = frames[j]
            if (not (prev_frame == -1)) and (f - prev_frame > skip):
                stopping_points.append(j)
            prev_frame = f
        for j in range(len(stopping_points)):
            max_ped_id += 1
            stop_pt = stopping_points[j]
            for k in range(stop_pt, len(ped_locations)):
                ped_idx_arr[ped_locations[k]] = max_ped_id
        i += 1
    return ped_idx_arr

def refine_pedidx(frames_arr, ped_idx_arr, pos_x_arr, pos_y_arr):
    num_points = len(frames_arr)
    check_array = [0] * num_points
    for i in range(num_points):
        frame = frames_arr[i]
        ped_idx = ped_idx_arr[i]
        for j in range(i + 1, num_points):
            if (frame == frames_arr[j]) and (ped_idx == ped_idx_arr[j]):
                check_array[j] = 1
            if (frame != frames_arr[j]):
                break

    i = 0
    while (i < len(check_array)):
        if check_array[i] == 1:
            del check_array[i]
            del frames_arr[i]
            del ped_idx_arr[i]
            del pos_x_arr[i]
            del pos_y_arr[i]
        else:
            i += 1
    return frames_arr, ped_idx_arr, pos_y_arr, pos_x_arr

'''
def revise_pedidx(frames_arr, ped_idx_arr):
    skip = 2
    arr_length = len(frames_arr)
    for i in range(arr_length):
        if i % 100 == 0:
            print([i, arr_length])
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
'''

path1 = 'test_data_complete/eth_0_pos.p'
path2 = 'test_data_complete/eth_0_neg.p'
path3 = 'test_data_complete/eth_1_pos.p'
path4 = 'test_data_complete/eth_1_neg.p'
path5 = 'test_data_complete/ucy_0_pos.p'
path6 = 'test_data_complete/ucy_0_neg.p'
path7 = 'test_data_complete/ucy_1_pos.p'
path8 = 'test_data_complete/ucy_1_neg.p'
path9 = 'test_data_complete/ucy_2_pos.p'
path10 = 'test_data_complete/ucy_2_neg.p'

paths = [[path1, path2], [path3, path4], [path5, path6], [path7, path8], [path9, path10]]
save_paths = ['../SR-LSTM/data/eth/univ/',
              '../SR-LSTM/data/eth/hotel/',
              '../SR-LSTM/data/ucy/zara/zara01/',
              '../SR-LSTM/data/ucy/zara/zara02/',
              '../SR-LSTM/data/ucy/univ/']

for i, p in enumerate(paths):
    p1 = p[0]
    p2 = p[1]
    with open(p1, 'rb') as f1:
        u1 = pickle._Unpickler(f1)
        u1.encoding = 'latin1'
        d1 = u1.load()
    with open(p2, 'rb') as f2:
        u2 = pickle._Unpickler(f2)
        u2.encoding = 'latin1'
        d2 = u2.load()
    
    dt_name = save_paths[i]
    frames_arr, ped_idx_arr, pos_x_arr, pos_y_arr, frames_1, ped_idx_1, pos_x_1, pos_y_1, fix_arr, fix_arr_1 = prepare_txt_data(d1)
    frames_arr, ped_idx_arr, pos_x_arr, pos_y_arr = (
        filter_patches(frames_arr, ped_idx_arr, pos_x_arr, pos_y_arr, fix_arr))
    ped_idx_arr = revise_pedidx_2(frames_arr, ped_idx_arr)
    frames_arr, ped_idx_arr, pos_x_arr, pos_y_arr = (
        refine_pedidx(frames_arr, ped_idx_arr, pos_x_arr, pos_y_arr))
    with open(dt_name + '0/0/eval_pos_.csv', 'w') as f:
        writer = csv.writer(f, delimiter = ',')
        writer.writerow(frames_arr)
        writer.writerow(ped_idx_arr)
        writer.writerow(pos_x_arr)
        writer.writerow(pos_y_arr)
    frames_arr = frames_1
    ped_idx_arr = ped_idx_1
    pos_x_arr = pos_x_1
    pos_y_arr = pos_y_1
    fix_arr = fix_arr_1
    frames_arr, ped_idx_arr, pos_x_arr, pos_y_arr = (
        filter_patches(frames_arr, ped_idx_arr, pos_x_arr, pos_y_arr, fix_arr))
    ped_idx_arr = revise_pedidx_2(frames_arr, ped_idx_arr)
    frames_arr, ped_idx_arr, pos_x_arr, pos_y_arr = (
        refine_pedidx(frames_arr, ped_idx_arr, pos_x_arr, pos_y_arr))
    with open(dt_name + '0/1/eval_pos_.csv', 'w') as f:
        writer = csv.writer(f, delimiter = ',')
        writer.writerow(frames_arr)
        writer.writerow(ped_idx_arr)
        writer.writerow(pos_x_arr)
        writer.writerow(pos_y_arr)
    frames_arr = frames_1

    frames_arr, ped_idx_arr, pos_x_arr, pos_y_arr, frames_1, ped_idx_1, pos_x_1, pos_y_1, fix_arr, fix_arr_1 = prepare_txt_data(d2)
    frames_arr, ped_idx_arr, pos_x_arr, pos_y_arr = (
        filter_patches(frames_arr, ped_idx_arr, pos_x_arr, pos_y_arr, fix_arr))
    ped_idx_arr = revise_pedidx_2(frames_arr, ped_idx_arr)
    frames_arr, ped_idx_arr, pos_x_arr, pos_y_arr = (
        refine_pedidx(frames_arr, ped_idx_arr, pos_x_arr, pos_y_arr))
    with open(dt_name + '1/0/eval_pos_.csv', 'w') as f:
        writer = csv.writer(f, delimiter = ',')
        writer.writerow(frames_arr)
        writer.writerow(ped_idx_arr)
        writer.writerow(pos_x_arr)
        writer.writerow(pos_y_arr)
    frames_arr = frames_1
    ped_idx_arr = ped_idx_1
    pos_x_arr = pos_x_1
    pos_y_arr = pos_y_1
    fix_arr = fix_arr_1
    frames_arr, ped_idx_arr, pos_x_arr, pos_y_arr = (
        filter_patches(frames_arr, ped_idx_arr, pos_x_arr, pos_y_arr, fix_arr))
    ped_idx_arr = revise_pedidx_2(frames_arr, ped_idx_arr)
    frames_arr, ped_idx_arr, pos_x_arr, pos_y_arr = (
        refine_pedidx(frames_arr, ped_idx_arr, pos_x_arr, pos_y_arr))
    with open(dt_name + '1/1/eval_pos_.csv', 'w') as f:
        writer = csv.writer(f, delimiter = ',')
        writer.writerow(frames_arr)
        writer.writerow(ped_idx_arr)
        writer.writerow(pos_x_arr)
        writer.writerow(pos_y_arr)
    frames_arr = frames_1
    print(i)
