import tensorflow as tf
import numpy as np
from os.path import join
import glob
import tensorflow_probability as tfp


_LOC = [0.1, 0.7, 1.9, 3.4, 3.8]
_SCALE = [0.1, 0.2, 0.4, 0.2, 0.1]
_SKEWNESS = [-0.1, 1.5, 0.5, -1.5, 2.0]


class BrainDataProvider:
    """
    Data provider for multimodal brain MRI.
    Usage:
    :param config: dict to config options
    :param outdir: path to the tfrecord files
    """

    def __init__(self, outdir, config):
        self._dim = config.get('img_shape')
        self._mode = config.get('mode')
        self._batch_size = config.get('batch_size', 1)
        self._augment = config.get("data_augment", False)
        self._rotate = config.get("data_rotate", False)
        self._nb_class = config.get("nb_class", 5)
        self._nb_class = self._nb_class if self._nb_class >1 else self._nb_class + 1
        self._label_dist = config.get("label_dist", False)
        self._combined_labels = config.get("combined_labels", False)
        self._nb_samples = 1

        if self._mode == 'train':
            self._filenames_bravo = glob.glob(join(outdir, "bravo*train*.tfrecords"))
            self._filenames_ascend = glob.glob(join(outdir, "ascend_placebo*train*.tfrecords"))
            self._filename = self._filenames_bravo#self._filenames_bravo# + self._filenames_ascend
        elif self._mode == 'valid':
            self._filenames_bravo = glob.glob(join(outdir, 'bravo*valid*.tfrecords'))
            self._filenames_ascend = glob.glob(join(outdir, 'ascend_placebo*valid*.tfrecords'))
            self._filename = self._filenames_bravo#self._filenames_bravo# + self._filenames_ascend
        elif self._mode == 'test':
            self._filenames_bravo = glob.glob(join(outdir, 'bravo*test*.tfrecords'))
            self._filenames_ascend = glob.glob(join(outdir, 'ascend*test*.tfrecords'))
            self._filename = self._filenames_bravo

    def get_nb_samples(self):
        count = 0
        for fn in self._filename:
            for _ in tf.python_io.tf_record_iterator(fn):
                count += 1
        return count

    def combined_labels(self, nb_lesions):
        def f1(): return tf.constant(0)

        def f2(): return tf.constant(1)

        def f3(): return tf.constant(2)

        return tf.switch_case(nb_lesions, branch_fns={0: f1, 1: f2, 2: f2, 3: f2, 4: f2})

    def data_generator(self):

        def parser(serialized_example):
            """Parses a single tf.Example into image and label tensors."""
            features = tf.parse_single_example(
                serialized_example,
                features={
                    'tp1': tf.FixedLenFeature([], tf.string),
                    'tp2': tf.FixedLenFeature([], tf.string),
                    'id': tf.FixedLenFeature([], tf.string)
                    })

            input_images = tf.decode_raw(features['tp1'], tf.float64)
            input_images = tf.cast(tf.reshape(input_images, shape=(192, 192, 6)), tf.float32)
            output_images = tf.decode_raw(features['tp2'], tf.float64)
            output_images = tf.cast(tf.reshape(output_images, shape=(192, 192, 5)), tf.float32)
            labels = tf.reduce_sum(output_images[..., -1])
            labels = tf.cast(tf.greater(labels, 0), tf.float32)
            return input_images, output_images, labels

        def augmentation(input_image, output_image, labels):

            random_angles = tf.random_uniform(shape=(), minval=-np.pi / 4, maxval=np.pi / 4)
            image = tf.concat([input_image, output_image], axis=-1)

            image = tf.reshape(image, [192, 192, 11])
            image_r = tf.contrib.image.rotate(image, random_angles)
            image_r = tf.image.random_flip_left_right(image_r)
            image_r = tf.image.random_flip_up_down(image_r)

            image_r = tf.reshape(image_r, [192, 192, 11])

            input_image_r = image_r[..., :6]
            output_image_r = image_r[..., 6:]

            shift_value = tf.cast(tf.random_uniform((), -6, 6), tf.int32)
            input_image_r = tf.manip.roll(input_image_r, shift_value, axis=0)
            output_image_r = tf.manip.roll(output_image_r, shift_value, axis=0)

            shift_value = tf.cast(tf.random_uniform((), -6, 6), tf.int32)
            input_image_r = tf.manip.roll(input_image_r, shift_value, axis=1)
            output_image_r = tf.manip.roll(output_image_r, shift_value, axis=1)

            return input_image_r, output_image_r, labels

        dataset = tf.data.TFRecordDataset(self._filename)
        if self._mode == 'train':
            dataset = dataset.shuffle(buffer_size=10 * self._batch_size)
        dataset = dataset.map(parser, num_parallel_calls=8)
        if (self._mode == 'train') & self._augment:
            dataset = dataset.map(augmentation, num_parallel_calls=8)
        dataset = dataset.repeat().batch(self._batch_size).prefetch(1)
        return dataset
