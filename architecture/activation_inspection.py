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

def deconv_layer(inputs, strides, weights, biases, num_features, sess):
    inputs = np.expand_dims(inputs, axis=0)
    inputs_shape = np.shape(inputs)
    d_stride = strides[0]
    h_stride = strides[1]
    w_stride = strides[2]
    output_shape = tf.stack([inputs_shape[0], inputs_shape[1] * d_stride, 
                             inputs_shape[2] * h_stride, inputs_shape[3] * w_stride, 
                             num_features])
    x = tf.placeholder(tf.float32, inputs_shape)
    y = tf.nn.relu(x)
    #y = tf.nn.bias_add(y, -biases)
    y = tf.nn.conv3d_transpose(y, weights, output_shape, 
                               strides=[1, d_stride, h_stride, w_stride, 1],
                               padding='SAME')
    
    output = sess.run(y, feed_dict={x:inputs})
    
    return output[0]

def unpool_3d(inputs, switches, strides):
    inputs_shape = np.shape(inputs)
    switches_shape = np.shape(switches)
    if not (inputs_shape == switches_shape):
        raise Exception('shapes must be the same!')
    d_stride = strides[0]
    h_stride = strides[1]
    w_stride = strides[2]

    output = np.zeros((inputs_shape[0] * d_stride, inputs_shape[1] * h_stride, 
                       inputs_shape[2] * w_stride, inputs_shape[3]))
    for i in range(inputs_shape[0]):
        for j in range(inputs_shape[1]):
            for k in range(inputs_shape[2]):
                for l in range(inputs_shape[3]):
                    tmp_val = inputs[i][j][k][l]
                    switch = switches[i][j][k][l]
                    flag_d = (switch > 3)
                    flag_h = (switch % 4 > 1)
                    flag_w = (switch % 2 == 1)
                    output[i*d_stride+flag_d][j*h_stride+flag_h][k*w_stride+flag_w][l] = tmp_val
    return output

def get_switches(act_layer, strides):
    act_layer = act_layer[0]
    act_shape = np.shape(act_layer)
    d_stride = strides[0]
    h_stride = strides[1]
    w_stride = strides[2]

    switches = np.zeros((act_shape[0] / d_stride, act_shape[1] / h_stride, 
                         act_shape[2] / w_stride, act_shape[3]))
    for i in range(0, act_shape[0], d_stride):
        for j in range(0, act_shape[1], h_stride):
            for k in range(0, act_shape[2], w_stride):
                for l in range(0, act_shape[3]):
                    max_val = 0
                    max_idx = 0
                    for x in range(d_stride):
                        for y in range(h_stride):
                            for z in range(w_stride):
                                tmp_val = act_layer[i+x][j+y][k+z][l]
                                if tmp_val > max_val:
                                    max_val = tmp_val
                                    max_idx = x * 4 + y * 2 + z
                    switches[i / d_stride][j / h_stride][k / w_stride][l] = max_idx
    return switches

def main(argv=None):  # pylint: disable=unused-argument
    export_dir = 'checkpoints/train_store_e_test/'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
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

    act_conv1 = graph.get_tensor_by_name('encode_1_conv/encode_1_3dconv:0')
    act_conv2 = graph.get_tensor_by_name('encode_2_conv/encode_2_3dconv:0')
    act_conv3 = graph.get_tensor_by_name('encode_3_2_conv/encode_3_2_3dconv:0')
    act_conv4 = graph.get_tensor_by_name('encode_4_2_conv/encode_4_2_3dconv:0')
    act_conv5 = graph.get_tensor_by_name('encode_5_2_conv/encode_5_2_3dconv:0')
    weights1 = graph.get_tensor_by_name('encode_1_conv/weights:0')
    biases1 = graph.get_tensor_by_name('encode_1_conv/biases:0')
    weights2 = graph.get_tensor_by_name('encode_2_conv/weights:0')
    biases2 = graph.get_tensor_by_name('encode_2_conv/biases:0')
    weights3_1 = graph.get_tensor_by_name('encode_3_1_conv/weights:0')
    biases3_1 = graph.get_tensor_by_name('encode_3_1_conv/biases:0')
    weights3_2 = graph.get_tensor_by_name('encode_3_2_conv/weights:0')
    biases3_2 = graph.get_tensor_by_name('encode_3_2_conv/biases:0')
    weights4_1 = graph.get_tensor_by_name('encode_4_1_conv/weights:0')
    biases4_1 = graph.get_tensor_by_name('encode_4_1_conv/biases:0')
    weights4_2 = graph.get_tensor_by_name('encode_4_2_conv/weights:0')
    biases4_2 = graph.get_tensor_by_name('encode_4_2_conv/biases:0')
    weights5_1 = graph.get_tensor_by_name('encode_5_1_conv/weights:0')
    biases5_1 = graph.get_tensor_by_name('encode_5_1_conv/biases:0')
    weights5_2 = graph.get_tensor_by_name('encode_5_2_conv/weights:0')
    biases5_2 = graph.get_tensor_by_name('encode_5_2_conv/biases:0')

    sim = False
    dt_loader = dt.DataLoader(sim = sim)
    dat = dt_loader.get_seq(1)
    x_seq = dat['img_seq']
    y_gt = dat['action']
    y_gt = np.argmax(y_gt)
    flag = True
    while flag:
        y_pd = sess.run(y, feed_dict={x:x_seq, is_train: False})
        y_pd = np.argmax(y_pd)
        if (y_pd == y_gt) and (y_pd == 1) and (np.max(y_pd) > 0.75):
            flag = False
        else:
            dat = dt_loader.get_seq(1)
            x_seq = dat['img_seq']
            y_gt = dat['action']
            y_gt = np.argmax(y_gt)
       
 
    act_conv1_val = sess.run(act_conv1, feed_dict={x:x_seq, is_train: False})
    act_conv2_val = sess.run(act_conv2, feed_dict={x:x_seq, is_train: False})
    act_conv3_val = sess.run(act_conv3, feed_dict={x:x_seq, is_train: False})
    act_conv4_val = sess.run(act_conv4, feed_dict={x:x_seq, is_train: False})
    a = sess.run(act_conv5, feed_dict={x:x_seq, is_train: False})
    max_val = np.max(a)
    max_idx = np.unravel_index(np.argmax(a, axis=None), a.shape)
    act_conv5_val = np.zeros_like(a)
    act_conv5_val[max_idx[0], max_idx[1], max_idx[2], max_idx[3], max_idx[4]] = max_val
    
    print('Getting_switches 1 ...')
    switches_conv1 = get_switches(act_conv1_val, [1,2,2])
    print('Getting_switches 2 ...')
    switches_conv2 = get_switches(act_conv2_val, [2,2,2])
    print('Getting_switches 3 ...')
    switches_conv3 = get_switches(act_conv3_val, [2,2,2])
    print('Getting_switches 4 ...')
    switches_conv4 = get_switches(act_conv4_val, [2,2,2])

    print('Deconv layer 5 ...')
    tmp = deconv_layer(act_conv5_val[0], [1, 1, 1], weights5_2, biases5_2, 512, sess)
    tmp = deconv_layer(tmp, [1, 1, 1], weights5_1, biases5_1, 512, sess)
    print('Deconv layer 4 ...')
    tmp = unpool_3d(tmp, switches_conv4, [2,2,2])
    tmp = deconv_layer(tmp, [1, 1, 1], weights4_2, biases4_2, 512, sess)
    tmp = deconv_layer(tmp, [1, 1, 1], weights4_1, biases4_1, 256, sess)
    print('Deconv layer 3 ...')
    tmp = unpool_3d(tmp, switches_conv3, [2,2,2])
    tmp = deconv_layer(tmp, [1, 1, 1], weights3_2, biases3_2, 256, sess)
    tmp = deconv_layer(tmp, [1, 1, 1], weights3_1, biases3_1, 128, sess)
    print('Deconv layer 2 ...')
    tmp = unpool_3d(tmp, switches_conv2, [2,2,2])
    tmp = deconv_layer(tmp, [1, 1, 1], weights2, biases2, 64, sess)
    print('Deconv layer 1 ...')
    tmp = unpool_3d(tmp, switches_conv1, [1,2,2])
    input_features = deconv_layer(tmp, [1, 1, 1], weights1, biases1, 1, sess)
    np.save('inputs.npy', x_seq)
    np.save('input_features.npy', input_features)

    return
	
if __name__ == '__main__':
	tf.app.run()

