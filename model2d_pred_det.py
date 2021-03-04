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

"""Open Source Version of the Hierarchical Probabilistic U-Net."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import geco_utils_det
import sonnet as snt
import tensorflow as tf
import unet_utils


class _HierarchicalCore(snt.AbstractModule):
    """A U-Net encoder-decoder with a full encoder and a truncated decoder.

    The truncated decoder is interleaved with the hierarchical latent space and
    has as many levels as there are levels in the hierarchy plus one additional
    level.
    """

    def __init__(self, latent_dims, channels_per_block,
                 down_channels_per_block=None, activation_fn=tf.nn.relu,
                 initializers=None, regularizers=None, convs_per_block=1,
                 blocks_per_level=1, name='HierarchicalDecoderDist'):
        """Initializes a HierarchicalCore.

        Args:
          latent_dims: List of integers specifying the dimensions of the latents at
            each scale. The length of the list indicates the number of U-Net decoder
            scales that have latents.
          channels_per_block: A list of integers specifying the number of output
            channels for each encoder block.
          down_channels_per_block: A list of integers specifying the number of
            intermediate channels for each encoder block or None. If None, the
            intermediate channels are chosen equal to channels_per_block.
          activation_fn: A callable activation function.
          initializers: Optional dict containing ops to initialize the filters (with
            key 'w') or biases (with key 'b'). The default initializer for the
            weights is a truncated normal initializer, which is commonly used when
            the inputs are zero centered (see
            https://arxiv.org/pdf/1502.03167v3.pdf). The default initializer for the
            bias is a zero initializer.
          regularizers: Optional dict containing regularizers for the filters
            (with key 'w') and the biases (with key 'b'). As a default, no
            regularizers are used. A regularizer should be a function that takes a
            single `Tensor` as an input and returns a scalar `Tensor` output, e.g.
            the L1 and L2 regularizers in `tf.contrib.layers`.
          convs_per_block: An integer specifying the number of convolutional layers.
          blocks_per_level: An integer specifying the number of residual blocks per
            level.
          name: A string specifying the name of the module.
        """
        super(_HierarchicalCore, self).__init__(name=name)
        self._latent_dims = latent_dims
        self._channels_per_block = channels_per_block
        self._activation_fn = activation_fn
        self._initializers = initializers
        self._regularizers = regularizers
        self._convs_per_block = convs_per_block
        self._blocks_per_level = blocks_per_level
        if down_channels_per_block is None:
            self._down_channels_per_block = channels_per_block
        else:
            self._down_channels_per_block = down_channels_per_block
        self._name = name

    def _build(self, inputs, mean=False):
        """A build-method allowing to sample from the module as specified.

        Args:
          inputs: A tensor of shape (b,h,w,c). When using the module as a prior the
          `inputs` tensor should be a batch of images. When using it as a posterior
          the tensor should be a (batched) concatentation of images and
          segmentations.
          mean: A boolean or a list of booleans. If a boolean, it specifies whether
            or not to use the distributions' means in ALL latent scales. If a list,
            each bool therein specifies whether or not to use the scale's mean. If
            False, the latents of the scale are sampled.
          z_q: None or a list of tensors. If not None, z_q provides external latents
            to be used instead of sampling them. This is used to employ posterior
            latents in the prior during training. Therefore, if z_q is not None, the
            value of `mean` is ignored. If z_q is None, either the distributions
            mean is used (in case `mean` for the respective scale is True) or else
            a sample from the distribution is drawn.
        Returns:
          A Dictionary holding the output feature map of the truncated U-Net
          decoder under key 'decoder_features', a list of the U-Net encoder features
          produced at the end of each encoder scale under key 'encoder_outputs', a
          list of the predicted distributions at each scale under key
          'distributions', a list of the used latents at each scale under the key
          'used_latents'.
        """

        self._num_base_filters = 4
        self._channels_per_block = (self._num_base_filters * 1, self._num_base_filters * 2, self._num_base_filters * 4,
                                    self._num_base_filters * 8, self._num_base_filters * 8, self._num_base_filters * 8,
                                    self._num_base_filters * 8, self._num_base_filters * 8)
        self._latent_dims = (1, 1, 1, 1)
        self._down_channels_per_block = (self._num_base_filters * 8, self._num_base_filters * 8, self._num_base_filters * 8,
                                         self._num_base_filters * 8, self._num_base_filters * 8, self._num_base_filters * 4,
                                         self._num_base_filters * 2, self._num_base_filters)
        self._blocks_per_level = 3
        self._convs_per_block = 3

        encoder_features = inputs
        encoder_outputs = []
        num_levels = len(self._channels_per_block)
        num_latent_levels = len(self._latent_dims)

        # Iterate the descending levels in the U-Net encoder.
        for level in range(num_levels):
            # Iterate the residual blocks in each level.
            for i in range(self._blocks_per_level):
                encoder_features = unet_utils.res_block2d(
                    input_features=encoder_features,
                    n_channels=self._channels_per_block[level],
                    n_down_channels=self._down_channels_per_block[level],
                    activation_fn=self._activation_fn,
                    initializers=self._initializers,
                    regularizers=self._regularizers,
                    convs_per_block=self._convs_per_block,
                    name='{}-{}'.format(level, i))

            #print("encoder:", encoder_features)

            encoder_outputs.append(encoder_features)
            if level != num_levels - 1:
                scale = 3 if level==6 else 2
                encoder_features = unet_utils.resize_down2d(encoder_features, scale=scale)

        # Iterate the ascending levels in the (truncated) U-Net decoder.
        decoder_features = encoder_outputs[-1]
        for level in range(num_latent_levels):
            scale = 3 if level == 0 else 2
            decoder_output_hi = unet_utils.resize_up2d(decoder_features, scale=scale)
            decoder_features = tf.concat(
                [decoder_output_hi, encoder_outputs[::-1][level + 1]], axis=-1)

            # Iterate the residual blocks in each level.
            for i in range(self._blocks_per_level):
                decoder_features = unet_utils.res_block2d(
                    input_features=decoder_features,
                    n_channels=self._channels_per_block[::-1][level + 1],
                    n_down_channels=self._down_channels_per_block[::-1][level + 1],
                    activation_fn=self._activation_fn,
                    initializers=self._initializers,
                    regularizers=self._regularizers,
                    convs_per_block=self._convs_per_block,
                    name='{}-{}'.format(level, i))

            #print("decoder post:", decoder_features)

        return {'decoder_features': decoder_features,
                'encoder_features': encoder_outputs}


class _StitchingDecoder(snt.AbstractModule):
    """A module that completes the truncated U-Net decoder.

    Using the output of the HierarchicalCore this module fills in the missing
    decoder levels such that together the two form a symmetric U-Net.
    """

    def __init__(self, latent_dims, channels_per_block, num_classes,
                 down_channels_per_block=None, activation_fn=tf.nn.relu,
                 initializers=None, regularizers=None, convs_per_block=1,
                 blocks_per_level=2, name='StitchingDecoder'):

        """Initializes a StichtingDecoder.
        Args:
        latent_dims: List of integers specifying the dimensions of the latents at
          each scale. The length of the list indicates the number of U-Net
          decoder scales that have latents.
        channels_per_block: A list of integers specifying the number of output
          channels for each encoder block.
        num_classes: An integer specifying the number of segmentation classes.
        down_channels_per_block: A list of integers specifying the number of
          intermediate channels for each encoder block. If None, the
          intermediate channels are chosen equal to channels_per_block.
        activation_fn: A callable activation function.
        initializers: Optional dict containing ops to initialize the filters (with
          key 'w') or biases (with key 'b'). The default initializer for the
          weights is a truncated normal initializer, which is commonly used when
          the inputs are zero centered (see
          https://arxiv.org/pdf/1502.03167v3.pdf). The default initializer for the
          bias is a zero initializer.
        regularizers: Optional dict containing regularizers for the filters
          (with key 'w') and the biases (with key 'b'). As a default, no
          regularizers are used. A regularizer should be a function that takes a
          single `Tensor` as an input and returns a scalar `Tensor` output, e.g.
          the L1 and L2 regularizers in `tf.contrib.layers`.
        convs_per_block: An integer specifying the number of convolutional layers.
        blocks_per_level: An integer specifying the number of residual blocks per
          level.
        name: A string specifying the name of the module.
        """
        super(_StitchingDecoder, self).__init__(name=name)
        self._latent_dims = latent_dims
        self._channels_per_block = channels_per_block
        self._num_classes = num_classes
        self._activation_fn = activation_fn
        self._initializers = initializers
        self._regularizers = regularizers
        self._convs_per_block = convs_per_block
        self._blocks_per_level = blocks_per_level
        if down_channels_per_block is None:
            down_channels_per_block = channels_per_block
        self._down_channels_per_block = down_channels_per_block

    def _build(self, encoder_features, decoder_features):
        """Build-method that returns the segmentation logits.

        Args:
          encoder_features: A list of tensors of shape (b,h_i,w_i,c_i).
          decoder_features: A tensor of shape (b,h,w,c).
        Returns:
          Logits, i.e. a tensor of shape (b,h,w,num_classes).
        """

        self._num_base_filters = 4
        self._channels_per_block = (self._num_base_filters * 1, self._num_base_filters * 2, self._num_base_filters * 4,
                                    self._num_base_filters * 8, self._num_base_filters * 8, self._num_base_filters * 8,
                                    self._num_base_filters * 8, self._num_base_filters * 8)
        self._latent_dims = (1, 1, 1, 1)
        self._down_channels_per_block = (self._num_base_filters * 8, self._num_base_filters * 8, self._num_base_filters * 8,
                                         self._num_base_filters * 8, self._num_base_filters * 8, self._num_base_filters * 4,
                                         self._num_base_filters * 2, self._num_base_filters)
        self._blocks_per_level = 3
        self._convs_per_block = 3

        num_latents = len(self._latent_dims)
        start_level = num_latents + 1
        num_levels = len(self._channels_per_block)

        for level in range(start_level, num_levels, 1):

            decoder_features = unet_utils.resize_up2d(decoder_features, scale=2)
            decoder_features = tf.concat([decoder_features, encoder_features[::-1][level]], axis=-1)

            for _ in range(self._blocks_per_level):
                decoder_features = unet_utils.res_block2d(
                    input_features=decoder_features,
                    n_channels=self._channels_per_block[::-1][level],
                    n_down_channels=self._down_channels_per_block[::-1][level],
                    activation_fn=self._activation_fn,
                    initializers=self._initializers,
                    regularizers=self._regularizers,
                    convs_per_block=self._convs_per_block)

            #print("decoder:", decoder_features)

        a = snt.Conv2D(output_channels=self._num_classes,
                       kernel_shape=(1, 1),
                       padding='SAME',
                       initializers=self._initializers,
                       regularizers=self._regularizers,
                       name='logits')(decoder_features)

        return a


class HierarchicalProbUNet(snt.AbstractModule):
    """A Hierarchical Probabilistic U-Net."""

    def __init__(self,
                 latent_dims=(1, 1, 1, 1),
                 channels_per_block=None,
                 num_classes=1,
                 down_channels_per_block=None,
                 activation_fn=tf.nn.relu,
                 initializers=None,
                 regularizers=None,
                 convs_per_block=1,
                 blocks_per_level=1,
                 name='HPUNet'):
        """Initializes a HierarchicalProbUNet.

        The default values are set as for the LIDC-IDRI experiments in
        `A Hierarchical Probabilistic U-Net for Modeling Multi-Scale Ambiguities',
        see https://arxiv.org/abs/1905.13077.
        Args:
          latent_dims: List of integers specifying the dimensions of the latents at
            each scales. The length of the list indicates the number of U-Net
            decoder scales that have latents.
          channels_per_block: A list of integers specifying the number of output
            channels for each encoder block.
          num_classes: An integer specifying the number of segmentation classes.
          down_channels_per_block: A list of integers specifying the number of
            intermediate channels for each encoder block. If None, the
            intermediate channels are chosen equal to channels_per_block.
          activation_fn: A callable activation function.
          initializers: Optional dict containing ops to initialize the filters (with
            key 'w') or biases (with key 'b').
          regularizers: Optional dict containing regularizers for the filters
            (with key 'w') and the biases (with key 'b').
          convs_per_block: An integer specifying the number of convolutional layers.
          blocks_per_level: An integer specifying the number of residual blocks per
            level.
          loss_kwargs: None or dictionary specifying the loss setup.
          name: A string specifying the name of the module.
        """
        super(HierarchicalProbUNet, self).__init__(name=name)
        base_channels = 1
        default_channels_per_block = (
            base_channels, 1 * base_channels, 1 * base_channels, 1 * base_channels,
            1 * base_channels, 1 * base_channels, 1 * base_channels
        )
        if channels_per_block is None:
            channels_per_block = default_channels_per_block
        if down_channels_per_block is None:
            down_channels_per_block = \
                tuple([i / 2 for i in default_channels_per_block])
        if initializers is None:
            initializers = {
                'w': tf.orthogonal_initializer(gain=1.0, seed=None),
                'b': tf.truncated_normal_initializer(stddev=0.001)
            }
        if regularizers is None:
            regularizers = {
                'w': tf.keras.regularizers.l2(1e-5),
                'b': tf.keras.regularizers.l2(1e-5)
            }

        if down_channels_per_block is None:
            down_channels_per_block = channels_per_block

        with self._enter_variable_scope():

            self._posterior = _HierarchicalCore(
                latent_dims=latent_dims,
                channels_per_block=channels_per_block,
                down_channels_per_block=down_channels_per_block,
                activation_fn=activation_fn,
                initializers=initializers,
                regularizers=regularizers,
                convs_per_block=convs_per_block,
                blocks_per_level=blocks_per_level,
                name='posterior')

            self._f_comb = _StitchingDecoder(
                latent_dims=latent_dims,
                channels_per_block=channels_per_block,
                num_classes=num_classes,
                down_channels_per_block=down_channels_per_block,
                activation_fn=activation_fn,
                initializers=initializers,
                regularizers=regularizers,
                convs_per_block=convs_per_block,
                blocks_per_level=blocks_per_level,
                name='f_comb')

            self._cache = ()

    def optimizer(self, loss, lr, global_step):

        loss = loss['supervised_loss']
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            optimiser = tf.train.AdamOptimizer(learning_rate=lr)

            gradients = optimiser.compute_gradients(loss)
            solver = optimiser.apply_gradients(gradients, global_step=global_step)

            # to add to tensorboard
            l2_norm = lambda t: tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))
            for gradient, variable in gradients:
                tf.summary.histogram("gradients/" + variable.name, l2_norm(gradient))
                tf.summary.histogram("variables/" + variable.name, l2_norm(variable))

        return solver

    def _build(self, img1):
        """Inserts all ops used during training into the graph exactly once.

        The first time this method is called given the input pair (seg, img) all
        ops relevant for training are inserted into the graph. Calling this method
        more than once does not re-insert the modules into the graph (memoization),
        thus preventing multiple forward passes of submodules for the same inputs.
        The method is private and called when setting up the loss.
        Args:
          seg: A tensor of shape (b, h, w, num_classes).
          img: A tensor of shape (b, h, w, c)
        Returns: None
        """
        inputs = img1
        if self._cache == inputs:
            return
        else:
            self._q_sample = self._posterior(img1)
            self._cache = inputs
        return

    def reconstruct(self, img1):
        """Reconstruct a segmentation using the posterior.
        Args:
          seg: A tensor of shape (b, h, w, num_classes).
          img: A tensor of shape (b, h, w, c).
          mean: A boolean, specifying whether to sample from the full hierarchy of
           the posterior or use the posterior means at each scale of the hierarchy.
        Returns:
          A segmentation tensor of shape (b,h,w,num_classes).
        """
        self._build(img1)
        post_out = self._q_sample
        encoder_features = post_out['encoder_features']
        decoder_features = post_out['decoder_features']
        #return decoder_features
        return self._f_comb(encoder_features=encoder_features, decoder_features=decoder_features)

    def rec_loss(self, seg, img1, sample_weights):
        """Cross-entropy reconstruction loss employed in the ELBO-/ GECO-objective.

        Args:
          seg: A tensor of shape (b, h, w, num_classes).
          img: A tensor of shape (b, h, w, c).
          mask: A mask of shape (b, h, w) or None. If None no pixels are masked in
           the loss.
          top_k_percentage: None or a float in (0.,1.]. If None, a standard
            cross-entropy loss is calculated.
          deterministic: A Boolean indicating whether or not to produce the
            prospective top-k mask deterministically.
        Returns:
          A dictionary holding the mean and the pixelwise sum of the loss for the
          batch as well as the employed loss mask.
        """

        reconstruction = self.reconstruct(img1)
        return geco_utils_det.ce_loss(reconstruction, seg, sample_weights)

    def loss(self, batch, sample_weights, bs):
        """The full training objective, either ELBO or GECO.

        Args:
          seg: A tensor of shape (b, h, w, num_classes).
          img: A tensor of shape (b, h, w, c).
          mask: A mask of shape (b, h, w) or None. If None no pixels are masked in
           the loss.
        Returns:
          A dictionary holding the loss (with key 'loss') and the tensorboard
          summaries (with key 'summaries').
        """

        img1 = batch[0][..., :4]
        img1 = tf.reshape(img1, [bs, 192, 192, 4])

        seg = tf.cast(batch[1][..., -1] > 0, tf.float32)
        seg = tf.reshape(seg, [bs, 192, 192])

        summaries = {}
        rec_loss = self.rec_loss(seg, img1, sample_weights)

        summaries['rec_loss_mean'] = rec_loss['mean']
        summaries['rec_loss_sum'] = rec_loss['sum']
        summaries['weight'] = rec_loss['weight']

        return dict(supervised_loss=rec_loss['mean'], summaries=summaries)


if __name__ == '__main__':
    hpu_net = HierarchicalProbUNet()
