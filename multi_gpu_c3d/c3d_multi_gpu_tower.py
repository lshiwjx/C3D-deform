from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import tensorflow as tf

import c3d_model

FLAGS = tf.app.flags.FLAGS


def batch_loss_accu(logits, labels):
    """ 
    Args:
      logits: Logits from inference()
      labels: Labels from dataset. 1-D tensor of shape [batch_size]

    Returns:
      Loss and acc of type float
    """
    labels_one_hot = tf.one_hot(labels, depth=FLAGS.num_classes, on_value=1, off_value=0, axis=-1, )
    # whether the logits is same as label
    label_one_dim = tf.reshape(labels, [FLAGS.batch_size])
    is_in_top_1 = tf.nn.in_top_k(logits, label_one_dim, 1)
    accu_batch = tf.div(tf.reduce_sum(tf.cast(is_in_top_1, tf.float32)), FLAGS.batch_size)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=labels_one_hot, name='cross_entropy_per_example')
    loss_batch = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', loss_batch)

    return loss_batch, accu_batch


def tower_loss_accuracy(scope, video_clip, labels, dropout_ratio):
    """Calculate the total loss on a single tower.
    Args:
      scope: unique prefix string identifying the tower, e.g. 'tower_0'
      video_clip: Images. 4D tensor of shape [batch_size, height, width, 3].
      labels: Labels. 1D tensor of shape [batch_size].
      dropout_ratio:
    Returns:
       Tensor of shape [] containing the total loss and accuracy for a batch of data
    """
    # Build inference Graph.
    logits = c3d_model.inference_c3d(video_clip, dropout_ratio)

    # labels = tf.cast(tf.reduce_sum(labels, 1),tf.int32)

    # calculate the loss and accuracy of a batch.
    loss, accuracy = batch_loss_accu(logits, labels)
    # tf.summary.scalar('loss_inference', loss)

    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses', scope)

    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_loss')
    tf.summary.scalar('tower_loss', total_loss)

    return total_loss, accuracy


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension represent the tower. too mach time
            # expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(g)

        # Average over the 'tower' dimension.
        # grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grads, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
