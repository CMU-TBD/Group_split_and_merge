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

tf.app.flags.DEFINE_string('train_dir', './checkpoints/train_store_conv_lstm',
                            """dir to store trained net""")
tf.app.flags.DEFINE_integer('seq_length', 8,
                            """size of hidden layer""")
tf.app.flags.DEFINE_integer('seq_start', 8,
                            """ start of seq generation""")
tf.app.flags.DEFINE_integer('max_step', 200000,
                            """max num of steps""")
tf.app.flags.DEFINE_float('keep_prob', 1.0,
                            """for dropout""")
tf.app.flags.DEFINE_float('lr', .000001,
                            """for dropout""")
tf.app.flags.DEFINE_integer('batch_size', 1,
                            """batch size for training""")
tf.app.flags.DEFINE_float('weight_init', .01,
                            """weight init for fully connected layers""")

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') 

def generate_bouncing_ball_sample(batch_size, seq_length, shape, num_balls):
  dat = np.zeros((batch_size, seq_length, shape, shape, 3))
  for i in xrange(batch_size):
    dat[i, :, :, :, :] = b.bounce_vec(32, num_balls, seq_length)
  return dat 

def network(inputs, hidden, is_train, lstm=True):
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
  conv1 = ld.conv_layer(inputs, 5, 1, 64, "encode_1")
  max1 = tf.layers.max_pooling2d(conv1, 2, 2)
  # conv2
  conv2 = ld.conv_layer(max1, 5, 1, 128, "encode_2")
  max2 = tf.layers.max_pooling2d(conv2, 2, 2)
  # conv3
  conv3_1 = ld.conv_layer(max2, 3, 1, 256, "encode_3_1")
  conv3_2 = ld.conv_layer(conv3_1, 3, 1, 256, "encode_3_2")
  max3 = tf.layers.max_pooling2d(conv3_2, 2, 2)
  # conv4
  conv4_1 = ld.conv_layer(max3, 3, 1, 512, "encode_4_1")
  conv4_2 = ld.conv_layer(conv4_1, 3, 1, 512, "encode_4_2")
  max4 = tf.layers.max_pooling2d(conv4_2, 2, 2)
  # conv5
  conv5_1 = ld.conv_layer(max4, 3, 1, 512, "encode_5_1")
  conv5_2 = ld.conv_layer(conv5_1, 3, 1, 512, "encode_5_2")
  y_0 = tf.layers.max_pooling2d(conv5_2, 2, 2)
  # y_0 = tf.layers.max_pooling2d(conv2, 3, 2)
  if lstm:
    # conv lstm cell 
    with tf.variable_scope('conv_lstm', initializer = tf.random_uniform_initializer(-.01, 0.1)):
      cell = BasicConvLSTMCell.BasicConvLSTMCell([7,7], [3,3], 4)
      if hidden is None:
        hidden = cell.zero_state(FLAGS.batch_size, tf.float32) 
      y_1, hidden = cell(y_0, hidden)
  else:
    y_1 = ld.conv_layer(y_0, 3, 1, 256, "encode_6")
    #y_1 = ld.conv_layer(y_0, 3, 1, 8, "encode_3")
 
  # fc1
  hidden_flat = tf.reshape(y_1, [-1, 7 * 7 * 4])
  dense1 = ld.fc_layer(hidden_flat, 4096, "fc_1", True)
  drop1 = tf.nn.dropout(dense1, 1.0)
  # fc2
  dense2 = ld.fc_layer(drop1, 4096, "fc_2", True)
  drop2 = tf.nn.dropout(dense2, 1.0)

  y = ld.fc_layer(drop2, 3, "logits", True, linear = True)

  dense1_location = ld.fc_layer(hidden_flat, 4096, "fc_1_loc", is_train)
  dense2_location = ld.fc_layer(dense1_location, 4096, "fc_2_loc", is_train)
  y_loc = ld.fc_layer(dense2_location, 2, "location", is_train, linear = True)

  return y, y_loc, hidden

def network_3dconv(inputs):
  conv1 = ld.conv3d_layer(inputs, 3, 1, 64, "encode_1")
  max1 = tf.layers.max_pooling3d(conv1, (1, 2, 2), (1, 2, 2))
  
  conv2 = ld.conv3d_layer(max1, 3, 1, 128, "encode_2")
  max2 = tf.layers.max_pooling3d(conv2, (2, 2, 2), (2, 2, 2))
  
  conv3_1 = ld.conv3d_layer(max2, 3, 1, 256, "encode_3_1")
  conv3_2 = ld.conv3d_layer(conv3_1, 3, 1, 256, "encode_3_2")
  max3 = tf.layers.max_pooling3d(conv3_2, (2, 2, 2), (2, 2, 2))
  
  conv4_1 = ld.conv3d_layer(max3, 3, 1, 512, "encode_4_1")
  conv4_2 = ld.conv3d_layer(conv4_1, 3, 1, 512, "encode_4_2")
  max4 = tf.layers.max_pooling3d(conv4_2, (2, 2, 2), (2, 2, 2))
  
  conv5_1 = ld.conv3d_layer(max4, 3, 1, 512, "encode_5_1")
  conv5_2 = ld.conv3d_layer(conv5_1, 3, 1, 512, "encode_5_2")
  max5 = tf.layers.max_pooling3d(conv5_2, (2, 2, 2), (2, 2, 2))
  
  hidden_flat = tf.reshape(max5, [-1, 7 * 7 * 512])
  dense1 = ld.fc_layer(hidden_flat, 4096, "fc_1")
  drop1 = tf.nn.dropout(dense1, 0.8)
  dense2 = ld.fc_layer(drop1, 4096, "fc_2")
  drop2 = tf.nn.dropout(dense2, 0.8)
  y = ld.fc_layer(drop2, 3, "logits", linear = True)
  return y

# make a template for reuse
network_template = tf.make_template('network', network)

def train():
  """Train ring_net for a number of steps."""
  with tf.Graph().as_default():
    # make inputs
    x = tf.placeholder(tf.float32, [None, FLAGS.seq_length, 224, 224, 1])
    y_actual = tf.placeholder(tf.float32, [None, 3])
    loc_actual = tf.placeholder(tf.float32, [None, 2])
    is_train = tf.placeholder(tf.bool)

    # conv network
    hidden = None
    for i in xrange(FLAGS.seq_length-1):
       y, y_loc, hidden = network_template(x[:,i,:,:,:], hidden, is_train)

    #y = network_3dconv(x)    

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
    train_op = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss)
    
    # List of all Variables
    variables = tf.global_variables()

    # Build a saver
    saver = tf.train.Saver(tf.global_variables())   

    # Summary op
    summary_op = tf.summary.merge_all()
 
    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    sess = tf.Session(config=config)

    # init if this is the very time training
    print("init network from scratch")
    sess.run(init)

    # Summary op
    graph_def = sess.graph.as_graph_def(add_shapes=True)
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph_def=graph_def)
    data_loader = dt.DataLoader()

    for step in xrange(FLAGS.max_step+1):
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

      if step%5000 == 0:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)  
        print("saved to " + FLAGS.train_dir)

      if step%2000 == 0:
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
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()

if __name__ == '__main__':
  tf.app.run()


