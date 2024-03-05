import os
import os.path
import time

import numpy as np
import tensorflow as tf
import cv2

import bouncing_balls as b
import layer_def as ld
import BasicConvLSTMCell

import Data_loader_aug as dt
import Social_grouping as sg

def restore_params(a, params):
    a.frame_width = params['frame_width']
    a.frame_height = params['frame_height']
    a.H = params['H']
    a.aug_trans = params['aug_trans']
    a.aug_angle = params['aug_angle']
    a.bbox_crop = params['bbox_crop']
    a.process_scale = params['process_scale']
    a.new_size = params['new_size']
    return a

def main(argv=None):  # pylint: disable=unused-argument
    a = sg.SocialGrouping('ucy')

    export_dir = 'checkpoints/laser_a_debugged/'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    sess = tf.Session(config=config)

    saver = tf.train.import_meta_graph(export_dir + 'model.ckpt-66000.meta')
    saver.restore(sess, export_dir + 'model.ckpt-66000')

    graph = tf.get_default_graph()
    #print(tf.all_variables())
    #print([node.name for node in tf.get_default_graph().as_graph_def().node])
    x = graph.get_tensor_by_name('Placeholder:0')
    is_train = graph.get_tensor_by_name('Placeholder_3:0')

    y = graph.get_tensor_by_name('logits_fc/logits_fc:0')
    y_loc = graph.get_tensor_by_name('location_fc/location_fc:0')

    sim = True
    dt_loader = dt.DataLoader(sim = sim)
    dat = dt_loader.get_test_samples()

    y_gt = dat['action']
    y_gt_lc = dat['location']
    reverse_loc_params = dat['reverse_loc_params']
    num_total = np.shape(y_gt)[0]
    label_gt = np.argmax(y_gt, axis=1)
    label_pd = []
    loc_accuracies = []
    for idx in range(num_total):
        y_pd, y_lc = sess.run([tf.nn.softmax(y), y_loc], 
                              feed_dict={x:np.expand_dims(dat['img_seq'][idx], axis=0),
                                         is_train: False})
        label_pd.append(np.argmax(y_pd))
        if label_gt[idx] > 0:
            params = reverse_loc_params[idx]
            #a = restore_params(a, params)
            #loc_accuracies.append(np.linalg.norm(a._reverse_coordinate_transform(a._reverse_action_location(y_lc[0])) - y_gt_lc[idx]))
            loc_accuracies.append(np.linalg.norm(y_lc[0] - y_gt_lc[idx]))
    num_correct = np.sum(label_gt == np.array(label_pd))
    t_acc = num_correct * 1.0 / num_total
    t_loc_acc = np.mean(np.array(loc_accuracies))
    confusion_matrix = np.zeros((4, 4))
    for i in range(len(label_gt)):
        confusion_matrix[label_gt[i], label_pd[i]] += 1
        confusion_matrix[label_gt[i], 3] += 1
        confusion_matrix[3, label_pd[i]] += 1
    confusion_matrix[3, 3] = len(label_gt)
    print(confusion_matrix)
    print('Test Accuracy: ' + str(t_acc))
    print('Test Location Accuracy: (' + str(len(loc_accuracies)) + ') ' + str(t_loc_acc))

    return
	
if __name__ == '__main__':
	tf.app.run()

