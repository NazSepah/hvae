# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility Functions for the GECO-objective.

(GECO is described in `Taming VAEs`, see https://arxiv.org/abs/1810.00597).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import sonnet as snt
import tensorflow.compat.v1 as tf


def ce_loss(logits, labels, sample_weights):
  """Computes the cross-entropy loss.

  Optionally a mask and a top-k percentage for the used pixels can be specified.

  The top-k mask can be produced deterministically or sampled.
  Args:
    logits: A tensor of shape (b,h,w,num_classes)
    labels: A tensor of shape (b,h,w,num_classes)
    mask: None or a tensor of shape (b,h,w).
    top_k_percentage: None or a float in (0.,1.]. If None, a standard
      cross-entropy loss is calculated.
    deterministic: A Boolean indicating whether or not to produce the
      prospective top-k mask deterministically.

  Returns:
    A dictionary holding the mean and the pixelwise sum of the loss for the
    batch as well as the employed loss mask.
  """
  num_classes = logits.shape.as_list()[-1]
  y_flat = tf.reshape(logits, (-1, num_classes), name='reshape_y')
  t_flat = tf.reshape(labels, (-1, num_classes), name='reshape_t')
  mask = tf.ones(shape=(t_flat.shape.as_list()[0],))
  xe = tf.nn.weighted_cross_entropy_with_logits(labels=t_flat, logits=y_flat, pos_weight=sample_weights)

  # Calculate batch-averages for the sum and mean of the loss
  batch_size = labels.shape.as_list()[0]
  xe = tf.reshape(xe, shape=(batch_size, -1))
  mask = tf.reshape(mask, shape=(batch_size, -1))

  ce_sum_per_instance = tf.reduce_mean(xe, axis=1)
  ce_sum = tf.reduce_mean(ce_sum_per_instance, axis=0)
  ce_mean = tf.reduce_sum(xe) / tf.reduce_sum(mask)

  return {'mean': ce_mean, 'sum': ce_sum, 'mask': mask, 'weight': sample_weights}
