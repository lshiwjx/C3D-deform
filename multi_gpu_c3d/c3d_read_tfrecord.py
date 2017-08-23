from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import numpy as np
import tensorflow as tf
from skimage.viewer import ImageViewer

FLAGS = tf.app.flags.FLAGS


def decode_from_tfrecord(filename_queue):
    """
    Decode and preprocess the data from tfrecord.
    :param filename_queue: 
    :return: tensor of data and label
    """
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'clip': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })

    video_clip = tf.decode_raw(features['clip'], tf.uint8)
    video_clip = tf.reshape(video_clip, [FLAGS.video_clip_length, FLAGS.video_clip_height,
                                         FLAGS.video_clip_width, FLAGS.video_clip_channels])

    # clip process input si NLHWC
    video_clip = tf.transpose(video_clip, perm=[3, 0, 1, 2])
    video_clip = tf.cast(video_clip, tf.float32)
    # crop_mean = np.load(crop_mean)
    # crop_mean = tf.transpose(crop_mean, perm=[0, 3, 1, 2])
    video_clip = tf.random_crop(video_clip,
                                [FLAGS.video_clip_channels, FLAGS.video_clip_length, FLAGS.crop_size, FLAGS.crop_size])
    crop_mean0 = tf.fill([1, FLAGS.video_clip_length, FLAGS.crop_size, FLAGS.crop_size], FLAGS.crop_mean0)
    crop_mean1 = tf.fill([1, FLAGS.video_clip_length, FLAGS.crop_size, FLAGS.crop_size], FLAGS.crop_mean1)
    crop_mean2 = tf.fill([1, FLAGS.video_clip_length, FLAGS.crop_size, FLAGS.crop_size], FLAGS.crop_mean2)
    crop_mean = tf.concat([crop_mean0, crop_mean1, crop_mean2], 0)
    video_clip -= crop_mean
    video_clip = tf.transpose(video_clip, perm=[1, 2, 3, 0])

    label = tf.cast(features['label'], tf.int32)
    label = tf.reshape(label, [1])

    return video_clip, label


def read_data_batch(tf_record_file):
    """Reads input data num_epochs times.
    Args:
      tf_record_file: Selects between the training (True) and validation (False) data.
    Returns:
      A tuple (video_clip_batch, label_batch), where:
      * video_clip_batch is a float tensor with shape [batch_size, ]
      * label_batch is an int32 tensor with shape [batch_size]
      Note that an tf.train.QueueRunner is added to the graph, which
      must be run using e.g. tf.train.start_queue_runners().
    """

    num_epochs = FLAGS.num_epochs
    with tf.name_scope('data_input'):
        # create a queue and enqueue with the filename
        filename_queue = tf.train.string_input_producer(
            [tf_record_file], num_epochs=num_epochs, capacity=32)

        # read the img as tensor
        video_clip, label = decode_from_tfrecord(filename_queue)

        # TODO: tf.train.shuffle_batch_join if it is slow
        # img_list = [decode_from_tfrecord(filename_queue,crop_mean) for _ in range(3) ]
        # imgs, label_batch = tf.train.shuffle_batch_join(img_list, batch_size=batch_size,

        num_treads = 10
        # minimum number for examples, or will block dequeue
        min_after_dequeue = FLAGS.min_after_dequeue
        safety_margin = 3
        # capacity of the queue, or will block enqueue
        capacity = min_after_dequeue + (num_treads + safety_margin) * FLAGS.batch_size
        # return a batch of tensor, will build a queue, a queue runner and the enqueue/dequeue operations.
        video_clip_batch, label_batch = tf.train.shuffle_batch(
            [video_clip, label], batch_size=FLAGS.batch_size, num_threads=num_treads,
            capacity=capacity, min_after_dequeue=min_after_dequeue)

        return video_clip_batch, label_batch
