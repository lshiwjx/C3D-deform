from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import skimage.transform
import skimage.data
import os

# TODO change the directory and file record_name
img_channels = 3
clip_length = 16
clip_height = 120
clip_width = 160
img_name_list_dir = "./"
img_train_name_list = "train.txt"
img_valid_name_list = "val.txt"


def read_clip_and_label_1(line):
    """
    read pic according the dir path, return the clip for tf record writer, have overlap
    :param line: img dir and label: /d1/d2/ 0
    :return: clip(bytes) and label(int)
    """
    img_dir, label, _ = line.split(' ')
    print('current dir: %s' % img_dir)
    label = int(label)
    clips = []
    name_img = sorted(os.listdir(img_dir))
    for i in range(len(name_img)//(clip_length//2)-1):
        clip = []
        for j in range(clip_length):
            img_name = os.path.join(img_dir, name_img[i*8+j])
            img = skimage.data.imread(img_name)
            # format lhwc
            img_resize = skimage.transform.resize(img, (clip_height, clip_width), preserve_range=True)
            clip.append(img_resize)
        clip_array = np.array(clip, np.uint8)
        clips.append(clip_array.tobytes())
    return clips, label


def read_clip_and_label(line):
    """
    read pic according the dir path, return the clip for tf record writer
    :param line: img dir and label: /d1/d2/ 0
    :return: clip(bytes) and label(int)
    """
    img_dir, label, _ = line.split(' ')
    label = int(label)
    clips = []
    clip = []
    i = 0
    for name in sorted(os.listdir(img_dir)):
        print('current dir: %s' % name)
        img_name = os.path.join(img_dir, name)
        img = skimage.data.imread(img_name)
        # format lhwc
        img_resize = skimage.transform.resize(img, (clip_height, clip_width), preserve_range=True)
        clip.append(img_resize)
        i += 1
        # make a clip
        if i == clip_length:
            clip_array = np.array(clip, np.uint8)
            clips.append(clip_array.tobytes())
            clip = []
            i = 0

    return clips, label


def convert_to(img_name_list, record_name):
    img_dir_name_lines = open(img_name_list_dir + img_name_list, 'r').readlines()

    record_writer = tf.python_io.TFRecordWriter(record_name + '.tfrecords')
    print('Writing data to ', record_name)

    for line in img_dir_name_lines:
        clips, label = read_clip_and_label_1(line)
        for clip in clips:
            # Example -> Features ->Feature(is a dict)
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                        'clip': tf.train.Feature(bytes_list=tf.train.BytesList(value=[clip]))}
                )
            )
            record_writer.write(example.SerializeToString())
    record_writer.close()


def main(_):
    convert_to(img_valid_name_list, 'rgb_8_train_uint8')
    convert_to(img_valid_name_list, 'rgb_8_val_uint8')


if __name__ == '__main__':
    tf.app.run()
