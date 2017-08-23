from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, weight_decay_ratio=None, stddev=0.0):
    """Helper to create an initialized Variable with weight decay.
    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      weight_decay_ratio: add L2 Loss weight decay multiplied by this float.
    Returns:
      Variable Tensor
    """
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(name, shape,
                           tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))  # 截断高斯
    if weight_decay_ratio is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), weight_decay_ratio, name='weight_loss')
        # add the loss of weight decay to losses
        tf.add_to_collection('losses', weight_decay)
        # tf.summary.scalar('weight_loss', weight_decay)
    return var


def conv_3d(filter_name, biases_name, input,
            filter_shape, biases_shape, filter_weight_decay=None, biases_weight_decay=None,
            filter_stddev=0.0, biases_stddev=0.0):
    """Computes a 3-D convolution given 5-D `input` and `filter` tensors.
    
      Args:
          biases_name: 
          filter_name: A name for the operation.
          input: [batch, in_channels, in_depth, in_height, in_width].
          filter_shape: [filter_depth, filter_height, filter_width, filter_channels, out_channels]
          biases_shape: same as filter_channels of filter
          filter_weight_decay:
          biases_weight_decay:
          filter_stddev:
          biases_stddev:
      Returns:
        tensor of convolution result
      """
    filter = _variable_with_weight_decay(filter_name, filter_shape, filter_weight_decay, stddev=filter_stddev)
    conv = tf.nn.conv3d(input, filter, [1, 1, 1, 1, 1], padding='SAME')
    biases = _variable_with_weight_decay(biases_name, biases_shape, biases_weight_decay, stddev=biases_stddev)
    pre_activation = tf.nn.bias_add(conv, biases)
    return pre_activation


def max_pool(name, l_input, depth):
    return tf.nn.max_pool3d(l_input, ksize=[1, depth, 2, 2, 1],
                            strides=[1, depth, 2, 2, 1], padding='SAME',
                            name=name)


def inference_c3d(video_clip, dropout_ratio, is_feature_extractor=False):
    """Generate the 3d convolution classification output according to the input video_clip
  
    Args:
        video_clip: Data Input, the shape of the Data Input is [batch_size, channel, length, height, weight]
        dropout_ratio: Tensor for scalar, diff for train or val
        is_feature_extractor: used as feature extractor or not
    Return:
      out: classification result, the shape is [batch_size, num_classes]
    """

    # Conv1 Layer
    with tf.variable_scope('conv1') as scope:
        tf.summary.image('before', video_clip[0], 1)

        conv1 = conv_3d('weight', 'biases', video_clip,
                        [3, 3, 3, FLAGS.video_clip_channels, 64], [64],
                        filter_weight_decay=FLAGS.weight_decay_ratio,
                        biases_weight_decay=None,
                        filter_stddev=(2.0 / (3 ** 3 * 3)) ** 0.5)
        conv1 = tf.nn.relu(conv1, name=scope.name)

        visual = tf.expand_dims(tf.transpose(conv1[0], perm=[3, 0, 1, 2]), 4)
        tf.summary.image('feature_map', visual[0], 1)

    # pool1
    pool1 = max_pool('pool1', conv1, 1)

    # Conv2 Layer
    with tf.variable_scope('conv2') as scope:
        conv2 = conv_3d('weight', 'biases', pool1,
                        [3, 3, 3, 64, 128], [128],
                        filter_weight_decay=FLAGS.weight_decay_ratio,
                        biases_weight_decay=None,
                        filter_stddev=(2.0 / (3 ** 3 * 64)) ** 0.5)
        conv2 = tf.nn.relu(conv2, name=scope.name)

        visual = tf.expand_dims(tf.transpose(conv2[0], perm=[3, 0, 1, 2]), 4)
        tf.summary.image('feature_map', visual[0], 1)

    # pool2
    pool2 = max_pool('pool2', conv2, 2)

    # Conv3 Layer
    with tf.variable_scope('conv3') as scope:
        conv3 = conv_3d('weight_a', 'biases_a', pool2,
                        [3, 3, 3, 128, 256], [256],
                        filter_weight_decay=FLAGS.weight_decay_ratio,
                        biases_weight_decay=None,
                        filter_stddev=(2.0 / (3 ** 3 * 128)) ** 0.5)
        conv3 = conv_3d('weight_b', 'biases_b', conv3,
                        [3, 3, 3, 256, 256], [256],
                        filter_weight_decay=FLAGS.weight_decay_ratio,
                        biases_weight_decay=None,
                        filter_stddev=(2.0 / (3 ** 3 * 256)) ** 0.5)
        conv3 = tf.nn.relu(conv3, name=scope.name + 'b')

        visual = tf.expand_dims(tf.transpose(conv3[0], perm=[3, 0, 1, 2]), 4)
        tf.summary.image('feature_map', visual[0], 1)

    # pool3
    pool3 = max_pool('pool3', conv3, 2)

    # Conv4 Layer
    with tf.variable_scope('conv4') as scope:
        conv4 = conv_3d('weight_a', 'biases_a', pool3,
                        [3, 3, 3, 256, 512], [512],
                        filter_weight_decay=FLAGS.weight_decay_ratio,
                        biases_weight_decay=None,
                        filter_stddev=(2.0 / (3 ** 3 * 256)) ** 0.5)
        conv4 = tf.nn.relu(conv4, name=scope.name + 'a')

        conv4 = conv_3d('weight_b', 'biases_b', conv4,
                        [3, 3, 3, 512, 512], [512],
                        filter_weight_decay=FLAGS.weight_decay_ratio,
                        biases_weight_decay=None,
                        filter_stddev=(2.0 / (3 ** 3 * 512)) ** 0.5)
        conv4 = tf.nn.relu(conv4, name=scope.name + 'b')

        visual = tf.expand_dims(tf.transpose(conv4[0], perm=[3, 0, 1, 2]), 4)
        tf.summary.image('feature_map', visual[0], 1)

    # pool4
    pool4 = max_pool('pool4', conv4, 2)

    # Conv5 Layer
    with tf.variable_scope('conv5') as scope:
        conv5 = conv_3d('weight_a', 'biases_a', pool4,
                        [3, 3, 3, 512, 512], [512],
                        filter_weight_decay=FLAGS.weight_decay_ratio,
                        biases_weight_decay=None,
                        filter_stddev=(2.0 / (3 ** 3 * 512)) ** 0.5)
        conv5 = tf.nn.relu(conv5, name=scope.name + 'a')

        conv5 = conv_3d('weight_b', 'biases_b', conv5,
                        [3, 3, 3, 512, 512], [512],
                        filter_weight_decay=FLAGS.weight_decay_ratio,
                        biases_weight_decay=None,
                        filter_stddev=(2.0 / (3 ** 3 * 512)) ** 0.5)
        conv5 = tf.nn.relu(conv5, name=scope.name + 'b')

        visual = tf.expand_dims(tf.transpose(conv5[0], perm=[3, 0, 1, 2]), 4)
        tf.summary.image('feature_map', visual[0], 1)

    # pool5
    pool5 = max_pool('pool5', conv5, 2)

    # local6
    with tf.variable_scope('local6') as scope:
        weights = _variable_with_weight_decay('weights', [8192, 4096],
                                              weight_decay_ratio=FLAGS.weight_decay_ratio,
                                              stddev=1.0 / 8192)
        biases = _variable_with_weight_decay('biases', [4096])
        pool5 = tf.transpose(pool5, perm=[0, 1, 4, 2, 3])
        local6 = tf.reshape(pool5, [-1, weights.get_shape().as_list()[0]])
        local6 = tf.nn.relu(tf.matmul(local6, weights) + biases, name=scope.name)
        if is_feature_extractor:
            return local6
        local6 = tf.nn.dropout(local6, dropout_ratio)
        tf.summary.histogram('histogram', local6)

    # local7
    with tf.variable_scope('local7') as scope:
        weights = _variable_with_weight_decay('weights', [4096, 4096],
                                              weight_decay_ratio=FLAGS.weight_decay_ratio,
                                              stddev=1.0 / 4096)
        biases = _variable_with_weight_decay('biases', [4096])
        local7 = tf.nn.relu(tf.matmul(local6, weights) + biases, name=scope.name)
        local7 = tf.nn.dropout(local7, dropout_ratio)
        tf.summary.histogram('histogram', local7)

    # linear layer(Wx + b)
    with tf.variable_scope('softmax_lineaer') as scope:
        weights = _variable_with_weight_decay('weights', [4096, FLAGS.num_classes],
                                              weight_decay_ratio=FLAGS.weight_decay_ratio,
                                              stddev=1.0 / 4096)
        biases = _variable_with_weight_decay('biases', [FLAGS.num_classes])
        softmax_linear = tf.add(tf.matmul(local7, weights), biases, name=scope.name)
        tf.summary.histogram('histogram', softmax_linear)

    return softmax_linear
