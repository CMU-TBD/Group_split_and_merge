import os
import os.path
import time

import math
import numpy as np
import tensorflow as tf
import cv2

import bouncing_balls as b
import layer_def as ld
import BasicConvLSTMCell

import sys
import Data_loader_aug as dt
sys.path.append('/home/allanwan/Private/social_grouping_project')
import Social_grouping as sg


def main(argv=None):  # pylint: disable=unused-argument
    export_dir = 'checkpoints/train_store_a_test/'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    sess = tf.Session(config=config)

    saver = tf.train.import_meta_graph(export_dir + 'model.ckpt-198000.meta')
    saver.restore(sess, tf.train.latest_checkpoint(export_dir))

    graph = tf.get_default_graph()
    #print(tf.all_variables())
    #print([node.name for node in tf.get_default_graph().as_graph_def().node])
    x = graph.get_tensor_by_name('Placeholder:0')
    is_train = graph.get_tensor_by_name('Placeholder_3:0')

    y = graph.get_tensor_by_name('logits_fc/logits_fc:0')
    y_loc = graph.get_tensor_by_name('location_fc/location_fc:0')

    s = sg.SocialGrouping('eth', flag=0)
    
    if not s.has_video:
        raise Exception('No original video present')

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('network_inference_eth_cp_nw.avi', fourcc, 25.0, (s.frame_width, s.frame_height))
    cap = cv2.VideoCapture(s.fname)
    i = 0
    seq_length = 16

    while (cap.isOpened() and (i < s.total_num_frames)):
        ret, frame = cap.read()
        #if i == 1000: break
        if (ret == True) and (len(s.video_position_matrix[i]) > 0) and (i >= seq_length):
            labels = s.video_labels_matrix[i]
            num_people = len(labels)
            label_set = []
            label_centroids = []
            for j in range(num_people):
                if not (labels[j] in label_set):
                    label_set.append(labels[j])
                    position_array, velocity_array, _ = s._find_label_properties(i, labels[j])
                    center = s._find_centroid(position_array)
                    center[1], center[0] = s._coordinate_transform(center)
                    label_centroids.append(center)
                    frame = s._draw_social_shapes(
                                frame, position_array, velocity_array, False)

            for j, lb in enumerate(label_set):
                    center = label_centroids[j]
                    # check split
                    img_seq = []
                    s._set_aug_param(0, [s.frame_width / 2 - center[0], 
                                         s.frame_height / 2 - center[1]])
                    stop_flag = False
                    for k in range(i - seq_length + 1, i + 1):
                        canvas = np.zeros((s.frame_height, s.frame_width, 3), dtype=np.uint8)
                        if (lb in s.video_labels_matrix[k]):
                            positions, velocities, _ = s._find_label_properties(k, lb)
                            canvas = s._draw_social_shapes(canvas, positions, velocities, True)
                        else:
                            stop_flag = True
                        img_seq.append(canvas)
                    if stop_flag:
                        continue
                    for k in range(len(img_seq)):
                        if k == 0:
                            s._prepare_process_image(img_seq)
                        img_seq[k] = s._process_image(img_seq[k])
                    img_seq = np.expand_dims(np.array(img_seq), axis=0)
                    y_pd, y_pd_loc = sess.run([tf.nn.softmax(y), y_loc],
                                              feed_dict={x:img_seq, is_train:False})
                    event_prob = y_pd[0][2]
                    event_prob = round(event_prob, 2)
                    if (np.argmax(y_pd) == 2) and (event_prob > 0.5):
                        y_pd_loc = s._reverse_action_location(y_pd_loc[0])
                        cv2.circle(frame, y_pd_loc, 8, (int(255 * event_prob), 0, 0), 2)
                        cv2.putText(frame, str(event_prob), (y_pd_loc[0], y_pd_loc[1] - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))

                    #check merge
                    img_seq = []
                    min_dist = 5000
                    lb_2 = lb
                    for k in range(len(label_set)):
                        if not (j == k):
                            center_2 = label_centroids[k]
                            center_dist = math.sqrt((center[0] - center_2[0]) ** 2 + 
                                                    (center[1] - center_2[1]) ** 2)
                            if center_dist < min_dist:
                                min_dist = center_dist
                                lb_2 = label_set[k]
                    if lb_2 == lb:
                        continue
                    positions_1, _, _ = s._find_label_properties(i, lb)
                    positions_2, _, _ = s._find_label_properties(i, lb_2)
                    center = s._find_centroid(positions_1 + positions_2)
                    center[1], center[0] = s._coordinate_transform(center)
                    s._set_aug_param(0, [s.frame_width / 2 - center[0], 
                                         s.frame_height / 2 - center[1]])
                    stop_flag = False
                    for k in range(i - seq_length + 1, i + 1):
                        canvas = np.zeros((s.frame_height, s.frame_width, 3), dtype=np.uint8)
                        if (lb in s.video_labels_matrix[k]):
                            positions, velocities, _ = s._find_label_properties(k, lb)
                            canvas = s._draw_social_shapes(canvas, positions, velocities, True)
                        else:
                            stop_flag = True
                        if (lb_2 in s.video_labels_matrix[k]):
                            positions, velocities, _ = s._find_label_properties(k, lb_2)
                            canvas = s._draw_social_shapes(canvas, positions, velocities, True)
                        else:
                            stop_flag = True
                        img_seq.append(canvas)
                    if stop_flag:
                        continue
                    for k in range(len(img_seq)):
                        if k == 0:
                            s._prepare_process_image(img_seq)
                        img_seq[k] = s._process_image(img_seq[k])
                    img_seq = np.expand_dims(np.array(img_seq), axis=0)
                    y_pd, y_pd_loc = sess.run([tf.nn.softmax(y), y_loc],
                                              feed_dict={x:img_seq, is_train:False})
                    event_prob = y_pd[0][1]
                    event_prob = round(event_prob, 2)
                    if (np.argmax(y_pd) == 1) and (event_prob > 0.5):
                        y_pd_loc = s._reverse_action_location(y_pd_loc[0])
                        cv2.circle(frame, y_pd_loc, 8, (0, int(255 * event_prob), 0), 2)
                        cv2.putText(frame, str(event_prob), (y_pd_loc[0], y_pd_loc[1] - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
                    
            i += 1
            print([i, s.total_num_frames])
            out.write(frame)
        elif (ret == True):
            i += 1
        else:
            print('read ends')
            break

    cap.release()
    out.release()

    return
	
if __name__ == '__main__':
	tf.app.run()

