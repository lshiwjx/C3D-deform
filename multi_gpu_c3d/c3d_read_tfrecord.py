from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import numpy as np
import tensorflow as tf
from skimage.viewer import ImageViewer

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('record_dir', './', "")
tf.app.flags.DEFINE_string('train_file', 'rgb_train_uint8.tfrecords', "")
tf.app.flags.DEFINE_string('val_file', 'rgb_val_uint8.tfrecords', "")
tf.app.flags.DEFINE_string('video_clip_channels', 3, "")
tf.app.flags.DEFINE_string('video_clip_length', 16, "the number of frame for a clip of video")
tf.app.flags.DEFINE_string('video_clip_height', 120, "")
tf.app.flags.DEFINE_string('video_clip_width', 160, "")
tf.app.flags.DEFINE_string('crop_size', 112, "")
tf.app.flags.DEFINE_string('crop_mean', [101.6089356, 97.6201375, 90.34391897], "")


def decode_from_tfrecord(filename_queue):
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

    # clip process
    video_clip = tf.transpose(video_clip, perm=[3, 0, 1, 2])
    video_clip = tf.cast(video_clip, tf.float32)
    # crop_mean = np.load(crop_mean)
    # crop_mean = tf.transpose(crop_mean, perm=[0, 3, 1, 2])
    video_clip = tf.random_crop(video_clip, [FLAGS.video_clip_channels, FLAGS.video_clip_length, FLAGS.crop_size, FLAGS.crop_size])
    crop_mean = np.zeros(
        [FLAGS.video_clip_channels, FLAGS.video_clip_length, FLAGS.crop_size, FLAGS.crop_size], np.float32)
    crop_mean[0, :, :, :] = FLAGS.crop_mean[0]
    crop_mean[1, :, :, :] = FLAGS.crop_mean[1]
    crop_mean[2, :, :, :] = FLAGS.crop_mean[2]
    video_clip -= crop_mean

    label = tf.cast(features['label'], tf.int32)
    label = tf.reshape(label, [1])

    return video_clip, label


def read_data_batch(is_training, batch_size, num_epochs=None):
    """Reads input data num_epochs times.
    Args:
      is_training: Selects between the training (True) and validation (False) data.
      batch_size: Number of examples per returned batch.
      num_epochs: Number of times to read the input data, or 0/None to
         train forever.
    Returns:
      A tuple (video_clip_batch, label_batch), where:
      * video_clip_batch is a float tensor with shape [batch_size, ]
      * label_batch is an int32 tensor with shape [batch_size]
      Note that an tf.train.QueueRunner is added to the graph, which
      must be run using e.g. tf.train.start_queue_runners().
    """

    if is_training:
        filename = os.path.join(FLAGS.record_dir, FLAGS.train_file)
        print('Train with: ' + filename)
    else:
        filename = os.path.join(FLAGS.record_dir, FLAGS.val_file)
        print('Validate with: ' + filename)

    if not num_epochs:
        num_epochs = None
    with tf.name_scope('input'):
        # create a queue and enqueue with the filename
        filename_queue = tf.train.string_input_producer(
            [filename], num_epochs=num_epochs, capacity=32)

        # read the img as tensor
        video_clip, label = decode_from_tfrecord(filename_queue)

        # TODO: tf.train.shuffle_batch_join if it is slow
        # img_list = [decode_from_tfrecord(filename_queue,crop_mean) for _ in range(3) ]
        # imgs, label_batch = tf.train.shuffle_batch_join(img_list, batch_size=batch_size,

        num_treads = 2
        # minimum number for examples, or will block dequeue
        min_after_dequeue = 100
        safety_margin = 3
        # capacity of the queue, or will block enqueue
        capacity = min_after_dequeue + (num_treads + safety_margin) * batch_size
        # return a batch of tensor, will build a queue, a queue runner and the enqueue/dequeue operations.
        video_clip_batch, label_batch = tf.train.shuffle_batch(
            [video_clip, label], batch_size=batch_size, num_threads=num_treads,
            capacity=capacity, min_after_dequeue=min_after_dequeue)

        return video_clip_batch, label_batch


def main(_):
    img, label = read_data_batch(1, 4, 1)
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        video_clip = img[0].eval()  # here is your video_clip Tensor :)
        tmp = np.asarray(np.rollaxis(video_clip[0], 0, 3))
        ImageViewer(tmp).show()

        video_clip1 = img[1].eval()  # here is your video_clip Tensor :)
        tmp1 = np.asarray(np.rollaxis(video_clip1[0], 0, 3))
        ImageViewer(tmp1).show()

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    tf.app.run()
