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

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './checkpoints/kernel_7',
                            """dir to store trained net""")
tf.app.flags.DEFINE_integer('seq_length', 16,
                            """size of hidden layer""")
tf.app.flags.DEFINE_integer('seq_start', 16,
                            """ start of seq generation""")
tf.app.flags.DEFINE_integer('max_step', 100000,
                            """max num of steps""")
tf.app.flags.DEFINE_float('keep_prob', 1.0,
                            """for dropout""")
tf.app.flags.DEFINE_float('lr', 0.00001,
                            """for dropout""")
tf.app.flags.DEFINE_integer('batch_size', 1,
                            """batch size for training""")
tf.app.flags.DEFINE_float('weight_init', .1,
                            """weight init for fully connected layers""")

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') 

def generate_bouncing_ball_sample(batch_size, seq_length, shape, num_balls):
  dat = np.zeros((batch_size, seq_length, shape, shape, 3))
  for i in xrange(batch_size):
    dat[i, :, :, :, :] = b.bounce_vec(32, num_balls, seq_length)
  return dat 

def network(inputs, hidden, lstm=True):
  """
  #conv1
  conv1 = ld.conv_layer(inputs, 3, 2, 64, "encode_1")
  # conv2
  conv2 = ld.conv_layer(conv1, 3, 1, 64, "encode_2")
  # conv3
  conv3 = ld.conv_layer(conv2, 3, 2, 64, "encode_3")
  # conv4
  conv4 = ld.conv_layer(conv3, 1, 1, 64, "encode_4")
  y_0 = conv4
  """
  # conv1
  conv1 = ld.conv_layer(inputs, 11, 4, 96, "encode_1")
  max1 = tf.layers.max_pooling2d(conv1, 3, 2)
  # conv2
  conv2 = ld.conv_layer(max1, 5, 1, 256, "encode_2")
  max2 = tf.layers.max_pooling2d(conv2, 3, 2)
  # conv3
  conv3 = ld.conv_layer(max2, 3, 1, 384, "encode_3")
  # conv4
  conv4 = ld.conv_layer(conv3, 3, 1, 384, "encode_4")
  # conv5
  conv5 = ld.conv_layer(conv4, 3, 1, 256, "encode_5")
  y_0 = tf.layers.max_pooling2d(conv5, 3, 2)
  #y_0 = tf.layers.max_pooling2d(conv2, 3, 2)
  if lstm:
    # conv lstm cell 
    with tf.variable_scope('conv_lstm', initializer = tf.random_uniform_initializer(-.01, 0.1)):
      cell = BasicConvLSTMCell.BasicConvLSTMCell([6,6], [3,3], 256)
      if hidden is None:
        hidden = cell.zero_state(FLAGS.batch_size, tf.float32) 
      y_1, hidden = cell(y_0, hidden)
  else:
    y_1 = ld.conv_layer(y_0, 3, 1, 256, "encode_6")
    #y_1 = ld.conv_layer(y_0, 3, 1, 8, "encode_3")
 
  # fc1
  hidden_flat = tf.reshape(y_1, [-1, 6 * 6 * 256])
  dense1 = ld.fc_layer(hidden_flat, 2048, "fc_1", False)
  drop1 = tf.nn.dropout(dense1, FLAGS.keep_prob)
  # fc2
  dense2 = ld.fc_layer(drop1, 2048, "fc_2", False)
  drop2 = tf.nn.dropout(dense2, FLAGS.keep_prob)

  y = ld.fc_layer(drop2, 3, "logits", False, linear = True)

  return y, hidden

def network_3dconv(inputs, is_train):
  weights = tf.constant(1.0, dtype=np.float32, shape=[3,1,1,1,1])
  inputs = tf.nn.conv3d(inputs, weights, strides=[1,1,1,1,1], padding='SAME')

  conv1 = ld.conv3d_layer(inputs, 3, 1, 64, 'SAME', "encode_1", is_train)
  #max1 = ld.conv3d_layer(conv1, 1, [1, 1, 2, 2, 1], 64, "max_1", is_train)
  max1 = tf.layers.max_pooling3d(conv1, (1, 2, 2), (1, 2, 2))
  
  conv2 = ld.conv3d_layer(max1, 3, 1, 128, 'VALID', "encode_2", is_train)
  #max2 = ld.conv3d_layer(conv2, 1, 2, 128, "max_2", is_train)
  max2 = tf.layers.max_pooling3d(conv2, (1, 2, 2), (1, 2, 2))
  
  conv3_1 = ld.conv3d_layer(max2, 3, 1, 256, 'VALID', "encode_3_1", is_train)
  conv3_2 = ld.conv3d_layer(conv3_1, 3, 1, 256, 'VALID', "encode_3_2", is_train)
  #max3 = ld.conv3d_layer(conv3_2, 1, 2, 256, "max_3", is_train)
  max3 = tf.layers.max_pooling3d(conv3_2, (1, 2, 2), (1, 2, 2))
  
  conv4_1 = ld.conv3d_layer(max3, 3, 1, 512, 'VALID', "encode_4_1", is_train)
  conv4_2 = ld.conv3d_layer(conv4_1, 3, 1, 512, 'VALID', "encode_4_2", is_train)
  #max4 = ld.conv3d_layer(conv4_2, 1, 2, 512, "max_4", is_train)
  max4 = tf.layers.max_pooling3d(conv4_2, (1, 2, 2), (1, 2, 2))
  
  conv5_1 = ld.conv3d_layer(max4, 3, 1, 512, 'VALID', "encode_5_1", is_train)
  conv5_2 = ld.conv3d_layer(conv5_1, 3, 1, 512, 'VALID', "encode_5_2", is_train)
  #max5 = ld.conv3d_layer(conv5_2, 1, 2, 512, "max_5", is_train)
  max5 = tf.layers.max_pooling3d(conv5_2, (1, 2, 2), (1, 2, 2))
  
  hidden_flat = tf.reshape(max5, [1, 2 * 3 * 3 * 512])
  dense1 = ld.fc_layer(hidden_flat, 4096, "fc_1", is_train)
  drop1 = tf.nn.dropout(dense1, FLAGS.keep_prob)
  dense2 = ld.fc_layer(drop1, 4096, "fc_2", is_train)
  drop2 = tf.nn.dropout(dense2, FLAGS.keep_prob)
  y = ld.fc_layer(drop2, 3, "logits", is_train, linear = True)
  return y, (max2, max5)

def network_3dconv_same(inputs, is_train):
  weights = tf.constant(1.0, dtype=np.float32, shape=[5,1,1,1,1])
  inputs = tf.nn.conv3d(inputs, weights, strides=[1,1,1,1,1], padding='SAME')

  conv1 = ld.conv3d_layer(inputs, 7, 1, 64, 'SAME', "encode_1", is_train)
  #max1 = ld.conv3d_layer(conv1, 1, [1, 1, 2, 2, 1], 64, "max_1", is_train)
  max1 = tf.layers.max_pooling3d(conv1, (1, 2, 2), (1, 2, 2))

  conv2 = ld.conv3d_layer(max1, 7, 1, 128, 'SAME', "encode_2", is_train)
  #max2 = ld.conv3d_layer(conv2, 1, 2, 128, "max_2", is_train)
  max2 = tf.layers.max_pooling3d(conv2, (2, 2, 2), (2, 2, 2))

  conv3_1 = ld.conv3d_layer(max2, 7, 1, 256, 'SAME', "encode_3_1", is_train)
  conv3_2 = ld.conv3d_layer(conv3_1, 7, 1, 256, 'SAME', "encode_3_2", is_train)
  #max3 = ld.conv3d_layer(conv3_2, 1, 2, 256, "max_3", is_train)
  max3 = tf.layers.max_pooling3d(conv3_2, (2, 2, 2), (2, 2, 2))

  with tf.device('/gpu:1'):
      conv4_1 = ld.conv3d_layer(max3, 7, 1, 512, 'SAME', "encode_4_1", is_train)
      conv4_2 = ld.conv3d_layer(conv4_1, 7, 1, 512, 'SAME', "encode_4_2", is_train)
      #max4 = ld.conv3d_layer(conv4_2, 1, 2, 512, "max_4", is_train)
      max4 = tf.layers.max_pooling3d(conv4_2, (2, 2, 2), (2, 2, 2))

      conv5_1 = ld.conv3d_layer(max4, 7, 1, 512, 'SAME', "encode_5_1", is_train)
      conv5_2 = ld.conv3d_layer(conv5_1, 7, 1, 512, 'SAME', "encode_5_2", is_train)
      #max5 = ld.conv3d_layer(conv5_2, 1, 2, 512, "max_5", is_train)
      max5 = tf.layers.max_pooling3d(conv5_2, (2, 2, 2), (2, 2, 2))

  hidden_flat = tf.reshape(max5, [1, 7 * 7 * 512])
  if is_train == True:
      dense1 = ld.fc_layer(hidden_flat, 4096, "fc_1", is_train)
      drop1 = tf.nn.dropout(dense1, FLAGS.keep_prob)
      dense2 = ld.fc_layer(drop1, 4096, "fc_2", is_train)
      drop2 = tf.nn.dropout(dense2, FLAGS.keep_prob)
      y = ld.fc_layer(drop2, 3, "logits", is_train, linear = True)
  else:
      dense1 = ld.fc_layer(hidden_flat, 4096, "fc_1", is_train)
      dense2 = ld.fc_layer(dense1, 4096, "fc_2", is_train)
      y = ld.fc_layer(dense2, 3, "logits", is_train, linear = True)
  dense1_location = ld.fc_layer(hidden_flat, 4096, "fc_1_loc", is_train)
  dense2_location = ld.fc_layer(dense1_location, 4096, "fc_2_loc", is_train)
  y_loc = ld.fc_layer(dense2_location, 2, "location", is_train, linear = True)
  return y, y_loc, (max2, max4)

def network_3dconv_simple(inputs, is_train):
  weights = tf.constant(1.0, dtype=np.float32, shape=[3,1,1,1,1])
  inputs = tf.nn.conv3d(inputs, weights, strides=[1,1,1,1,1], padding='SAME')

  conv1 = ld.conv3d_layer(inputs, 3, 1, 4, 'SAME', "encode_1", is_train)
  max1 = ld.conv3d_layer(conv1, 3, 2, 4, "max_1", is_train)
  #max1 = tf.layers.max_pooling3d(conv1, (1, 2, 2), (1, 2, 2), padding='same')

  conv2 = ld.conv3d_layer(max1, 3, 1, 8, 'SAME', "encode_2", is_train)
  max2 = ld.conv3d_layer(conv2, 3, 2, 8, "max_2", is_train)
  #max2 = tf.layers.max_pooling3d(conv2, (2, 2, 2), (2, 2, 2), padding='same')

  conv3_1 = ld.conv3d_layer(max2, 3, 1, 16, 'SAME', "encode_3_1", is_train)
  conv3_2 = ld.conv3d_layer(conv3_1, 3, 1, 16, 'SAME', "encode_3_2", is_train)
  max3 = ld.conv3d_layer(conv3_2, 3, 2, 16, "max_3", is_train)
  #max3 = tf.layers.max_pooling3d(conv3_2, (2, 2, 2), (2, 2, 2), padding='same')

  conv4_1 = ld.conv3d_layer(max3, 3, 1, 32, 'SAME', "encode_4_1", is_train)
  conv4_2 = ld.conv3d_layer(conv4_1, 3, 1, 32, 'SAME', "encode_4_2", is_train)
  max4 = ld.conv3d_layer(conv4_2, 3, 2, 32, "max_4", is_train)
  #max4 = tf.layers.max_pooling3d(conv4_2, (2, 2, 2), (2, 2, 2), padding='same')

  conv5_1 = ld.conv3d_layer(max4, 3, 1, 64, 'SAME', "encode_5_1", is_train)
  conv5_2 = ld.conv3d_layer(conv5_1, 3, 1, 64, 'SAME', "encode_5_2", is_train)
  max5 = ld.conv3d_layer(conv5_2, 3, 2, 64, "max_5", is_train)
  #max5 = tf.layers.max_pooling3d(conv5_2, (2, 2, 2), (2, 2, 2), padding='same')

  hidden_flat = tf.reshape(max5, [1, 7 * 7 * 64])
  #tmp_weights = tf.get_variable('weights', [8, 56, 56, 8, 16],
  #              initializer=tf.truncated_normal_initializer(stddev=0.01))
  #tmp_biases = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0.0))
  #tmp = tf.nn.bias_add(tf.nn.conv3d(max2, tmp_weights, strides=[1, 1, 1, 1, 1], padding='VALID'),
  #                     tmp_biases)
  #hidden_flat = tf.reshape(tmp, [1, 16])
  dense1 = ld.fc_layer(hidden_flat, 2048, "fc_1", is_train)
  #drop1 = tf.nn.dropout(dense1, FLAGS.keep_prob)
  dense2 = ld.fc_layer(dense1, 512, "fc_2", is_train)
  #drop2 = tf.nn.dropout(dense2, FLAGS.keep_prob)
  y = ld.fc_layer(dense2, 3, "logits", is_train, linear = True)
  return y, (max2, max5)

def network_3dconv_fpn(inputs, is_train):
  conv1 = ld.conv3d_layer(inputs, 3, 1, 64, 'SAME', "encode_1", is_train)
  max1 = tf.layers.max_pooling3d(conv1, (2, 2, 2), (2, 2, 2), padding='same')

  conv2 = ld.conv3d_layer(max1, 3, 1, 128, 'SAME', "encode_2", is_train)
  max2 = tf.layers.max_pooling3d(conv2, (2, 2, 2), (2, 2, 2), padding='same')

  conv3_1 = ld.conv3d_layer(max2, 3, 1, 256, 'SAME', "encode_3_1", is_train)
  conv3_2 = ld.conv3d_layer(conv3_1, 3, 1, 256, 'SAME', "encode_3_2", is_train)
  max3 = tf.layers.max_pooling3d(conv3_2, (2, 2, 2), (2, 2, 2), padding='same')

  conv4_1 = ld.conv3d_layer(max3, 3, 1, 512, 'SAME', "encode_4_1", is_train)
  conv4_2 = ld.conv3d_layer(conv4_1, 3, 1, 512, 'SAME', "encode_4_2", is_train)
  max4 = tf.layers.max_pooling3d(conv4_2, (2, 2, 2), (2, 2, 2), padding='same')

  conv5_1 = ld.conv3d_layer(max4, 3, 1, 512, 'SAME', "encode_5_1", is_train)
  conv5_2 = ld.conv3d_layer(conv5_1, 3, 1, 512, 'SAME', "encode_5_2", is_train)

  upconv4 = ld.transpose_3dconv_layer(conv5_2, 3, 2, 512, "decode_4", is_train)
  concat4 = tf.concat((upconv4, conv4_2), axis = 4, name = "concat_4")
  fusion4 = ld.conv3d_layer(concat4, 3, 1, 512, 'SAME', "fuse_4", is_train)
  #fusion4 = tf.add(upconv4, conv4_2)

  upconv3 = ld.transpose_3dconv_layer(fusion4, 3, 2, 256, "decode_3", is_train)
  concat3 = tf.concat((upconv3, conv3_2), axis = 4, name = "concat_3")
  fusion3 = ld.conv3d_layer(concat3, 3, 1, 256, 'SAME', "fuse_3", is_train)
  #fusion3 = tf.add(upconv3, conv3_2)

  upconv2 = ld.transpose_3dconv_layer(fusion3, 3, 2, 128, "decode_2", is_train)
  concat2 = tf.concat((upconv2, conv2), axis = 4, name = "concat_2")
  fusion2 = ld.conv3d_layer(concat2, 3, 1, 128, 'SAME', "fuse_2", is_train)
  #fusion2 = tf.add(upconv2, conv2)

  upconv1 = ld.transpose_3dconv_layer(fusion2, 3, 2, 64, "decode_1", is_train)
  concat1 = tf.concat((upconv1, conv1), axis = 4, name = "concat_1")
  fusion1 = ld.conv3d_layer(concat1, 3, 1, 64, 'SAME', "fuse_1", is_train)
  #fusion1 = tf.add(upconv1, conv1)

  fc1 = ld.conv3d_layer(fusion1, 1, 1, 3, 'SAME', "fc_1", is_train)
  fc2_weights = tf.get_variable('weights', [16, 224, 224, 3, 3], 
                initializer=tf.truncated_normal_initializer(stddev=0.01))
  fc2_biases = tf.get_variable('biases', [3], initializer=tf.constant_initializer(0.0))
  fc2 = tf.nn.bias_add(tf.nn.conv3d(fc1, fc2_weights, strides=[1, 1, 1, 1, 1], padding='VALID'), 
                       fc2_biases)
  y = tf.reshape(fc2, [-1, 3])
  return y, max4

# make a template for reuse
network_template = tf.make_template('network', network)

def train():
  """Train ring_net for a number of steps."""
  
  sim = False
  with tf.Graph().as_default():
    # make inputs
    x = tf.placeholder(tf.float32, [None, FLAGS.seq_length, 224, 224, 1])
    y_actual = tf.placeholder(tf.float32, [None, 3])
    loc_actual = tf.placeholder(tf.float32, [None, 2])
    is_train = tf.placeholder(tf.bool)

    # conv network
    #hidden = None
    #for i in xrange(FLAGS.seq_length-1):
    #    y, hidden = network_template(x[:,i,:,:,:], hidden)

    y, y_loc, debug_info = network_3dconv_same(x, is_train)    

    """
    # this part will be used for generating video
    x_unwrap_g = []
    hidden_g = None
    for i in xrange(50):
      if i < FLAGS.seq_start:
        x_1_g, hidden_g = network_template(x_dropout[:,i,:,:,:], hidden_g)
      else:
        x_1_g, hidden_g = network_template(x_1_g, hidden_g)
      x_unwrap_g.append(x_1_g)

    # pack them generated ones
    x_unwrap_g = tf.stack(x_unwrap_g)
    x_unwrap_g = tf.transpose(x_unwrap_g, [1,0,2,3,4])
    """

    # calc total loss (compare x_t to x_t+1)
    y_logits = y
    loss_list = tf.nn.softmax_cross_entropy_with_logits(labels=y_actual, logits=y_logits)

    tmp_loc_loss = tf.nn.l2_loss(y_loc - loc_actual)
    loc_loss = tmp_loc_loss * \
               tf.clip_by_value(tf.cast(tf.argmax(y_actual, axis=1), tf.float32), 0.0, 1.0)

    loss = tf.reduce_mean(loss_list + 0.005 * loc_loss)
    tf.summary.scalar('loss', loss)

    # training
    gstep = tf.train.create_global_step()
    train_op = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss, global_step = gstep)
    
    # List of all Variables
    variables = tf.global_variables()

    # Start running operations on the Graph.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
    sess = tf.Session(config=config)

    # init if this is the very time training
    if os.path.exists(FLAGS.train_dir):
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(FLAGS.train_dir)
        print ('loading model: ', ckpt)
        saver.restore(sess, ckpt)
        st_step = sess.run(tf.train.get_global_step())
    else:
        print("init network from scratch")
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver(tf.global_variables())   
        st_step = 0

    # Summary op
    summary_op = tf.summary.merge_all()
    graph_def = sess.graph.as_graph_def(add_shapes=True)
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph_def=graph_def)
    data_loader = dt.DataLoader(sim = sim)

    for step in xrange(st_step, FLAGS.max_step+1):
      #dat = generate_bouncing_ball_sample(FLAGS.batch_size, FLAGS.seq_length, 32, FLAGS.num_balls)
      t = time.time()
      dat = data_loader.get_seq(FLAGS.batch_size)
      _ = sess.run(train_op, 
                   feed_dict={x:dat['img_seq'], 
							  y_actual:dat['action'],
                              loc_actual:dat['location'],
							  is_train:True})
      elapsed = time.time() - t

      if (step % 100 == 0) and (step != 0):
        summary_str, loss_r = sess.run(
                            [summary_op, loss], 
                            feed_dict={x:dat['img_seq'], 
							           y_actual:dat['action'],
                                       loc_actual:dat['location'],
							           is_train:True})
        summary_writer.add_summary(summary_str, step) 
        print("time per batch is " + str(elapsed))
        print(step)
        print(loss_r)
      
        assert not np.isnan(loss_r), 'Model diverged with loss = NaN'

      if step % 2000 == 0:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)  
        print("saved to " + FLAGS.train_dir)

      if step % 2000 == 0:
        # Validation Accuracy
        """
        dat = data_loader.get_validation_samples()
        y_gt = dat['action']
        y_gt_lc = dat['location']
        num_total = np.shape(y_gt)[0]
        label_gt = np.argmax(y_gt, axis=1)
        label_pd = []
        loc_accuracies = []
        for idx in range(num_total):
          y_pd, y_lc = sess.run([y_logits, y_loc], 
                                feed_dict= {x:np.expand_dims(dat['img_seq'][idx], axis=0),
						                    is_train:False})
          label_pd.append(np.argmax(y_pd, axis=1)[0])
          if label_gt[idx] > 0:
            loc_accuracies.append(np.linalg.norm(y_lc[0] - y_gt_lc[idx]))
        """
        """
        #summary_img_vals = []
        for idx in range(10):
          debug_img_1, debug_img_2 = sess.run([debug_info[0], debug_info[1]], feed_dict= {x:np.expand_dims(dat['img_seq'][idx], axis=0),
						                               is_train:False})
          print(np.shape(debug_img_1))
          print(np.shape(debug_img_2))
          print(np.min(debug_img_1[0], axis=(0,1,2))[:10])
          print(np.min(debug_img_2[0], axis=(0,1,2))[:10])
          print(np.max(debug_img_1[0], axis=(0,1,2))[:10])
          print(np.max(debug_img_2[0], axis=(0,1,2))[:10])
          print('====================================')
          #debug_img = np.transpose(debug_img[0], (3, 1, 2, 0))
          #summary_img_vals.append(tf.summary.image(str(idx), tf.convert_to_tensor(debug_img)))
        #merged = sess.run(tf.summary.merge(summary_img_vals))
        #summary_writer.add_summary(merged)
        """
        """
        num_correct = np.sum(label_gt == np.array(label_pd))
        v_acc = num_correct * 1.0 / num_total
        v_loc_acc = np.mean(np.array(loc_accuracies))
        print('Validation Accuracy: ' + str(v_acc))
        print('Validation Location Accuracy: (' + str(len(loc_accuracies)) + ') ' + str(v_loc_acc))
        summary = tf.Summary(value = [tf.Summary.Value(tag = 'V Accuracy', simple_value = v_acc)])
        summary_writer.add_summary(summary, step)
        summary = tf.Summary(value = [tf.Summary.Value(tag = 'V Location Error', 
                                                       simple_value = v_loc_acc)])
        summary_writer.add_summary(summary, step)
        """
        # Test Accuracy
        dat = data_loader.get_test_samples()
        y_gt = dat['action']
        y_gt_lc = dat['location']
        num_total = np.shape(y_gt)[0]
        label_gt = np.argmax(y_gt, axis=1)
        label_pd = []
        loc_accuracies = []
        for idx in range(num_total):
          y_pd, y_lc = sess.run([y_logits, y_loc], 
                                feed_dict= {x:np.expand_dims(dat['img_seq'][idx], axis=0),
						                    is_train:False})
          label_pd.append(np.argmax(y_pd, axis=1)[0])
          if label_gt[idx] > 0:
            loc_accuracies.append(np.linalg.norm(y_lc[0] - y_gt_lc[idx]))
        num_correct = np.sum(label_gt == np.array(label_pd))
        t_acc = num_correct * 1.0 / num_total
        t_loc_acc = np.mean(np.array(loc_accuracies))
        confusion_matrix = np.zeros((4, 4))
        for i in range(len(label_gt)):
            confusion_matrix[int(label_gt[i]), label_pd[i]] += 1
            confusion_matrix[int(label_gt[i]), 3] += 1
            confusion_matrix[3, label_pd[i]] += 1
        confusion_matrix[3, 3] = len(label_gt)
        print(confusion_matrix)
        print('Test Accuracy: ' + str(t_acc))
        print('Test Location Accuracy: (' + str(len(loc_accuracies)) + ') ' + str(t_loc_acc))
        summary = tf.Summary(value = [tf.Summary.Value(tag = 'T Accuracy', simple_value = t_acc)])
        summary_writer.add_summary(summary, step)
        summary = tf.Summary(value = [tf.Summary.Value(tag = 'T Location Error', 
                                                       simple_value = t_loc_acc)])
        summary_writer.add_summary(summary, step)


        """
        # make video
        print("now generating video!")
        video = cv2.VideoWriter()
        success = video.open("generated_conv_lstm_video.mov", fourcc, 4, (180, 180), True)
        dat_gif = dat
        ims = sess.run([x_unwrap_g],feed_dict={x:dat_gif, keep_prob:FLAGS.keep_prob})
        ims = np.repeat(ims[0][0], 3, axis=3)
        #ims = ims[0][0]
        print(ims.shape)
        for i in xrange(50 - FLAGS.seq_start):
          x_1_r = np.uint8(np.maximum(ims[i,:,:,:], 0) * 255)
          new_im = cv2.resize(x_1_r, (180,180))
          video.write(new_im)
        video.release()
        """

def main(argv=None):  # pylint: disable=unused-argument
  #if tf.gfile.Exists(FLAGS.train_dir):
  #  tf.gfile.DeleteRecursively(FLAGS.train_dir)
  #tf.gfile.MakeDirs(FLAGS.train_dir)
  train()

if __name__ == '__main__':
  tf.app.run()


