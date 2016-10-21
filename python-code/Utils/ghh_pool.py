# ghh_pool.py ---
#
# Filename: ghh_pool.py
# Description:
# Author: Kwang
# Maintainer:
# Created: Wed Dec  2 18:03:59 2015 (+0100)
# Version:
# Package-Requires: ()
# Last-Updated: Thu Sep 15 23:01:03 2016 (+0200)
#           By: kwang
#     Update #: 61
# URL:
# Doc URL:
# Keywords:
# Compatibility:
#
#

# Commentary:
#
#
#
#

# Change Log:
#
#
#
#
# Copyright (C), EPFL Computer Vision Lab.
#
#

# Code:

from __future__ import print_function

import lasagne
import numpy as np
import six
import theano
import theano.tensor as T
from lasagne.layers import FeaturePoolLayer, Layer
from lasagne.random import get_rng
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

floatX = theano.config.floatX


class GHHFeaturePoolLayer(FeaturePoolLayer):
    """
    Just a wrapper on top of FeaturePoolLayer for easy use

    TODO: this layer does not support overlapping pooling regions...

    """

    def __init__(self, incoming, num_in_sum, num_in_max, axis=1, max_strength=-1, **kwargs):

        pool_size = num_in_sum * num_in_max
        if max_strength < 0:
            pool_function = make_ghh_pool_conv2d(num_in_sum, num_in_max)
        elif max_strength > 0:
            pool_function = make_ghh_soft_pool_conv2d(
                num_in_sum, num_in_max, max_strength)
        else:
            raise ValueError(
                'GHHFeaturePoolLayer: max_strength should not be zero!')

        super(GHHFeaturePoolLayer, self).__init__(
            incoming, pool_size, axis=1, pool_function=pool_function, **kwargs)


def make_ghh_pool_conv2d(num_in_sum, num_in_max):

    def pool_function(input, axis):

        input_shape = tuple(input.shape)
        num_feature_maps_out = input_shape[axis - 1]
        pool_size = input_shape[axis]

        pool_shape = (input_shape[:axis] + (num_in_sum,
                                            num_in_max) + input_shape[axis + 1:])
        # print("make_ghh_pool_conv2d: pool_shape is {}".format(pool_shape))
        input_reshaped = input.reshape(pool_shape)

        res_after_max = T.max(input_reshaped, axis=axis + 1)

        # Get deltas
        delta = np.cast[floatX](1.0) - np.cast[floatX](2.0) * \
            (T.arange(num_in_sum, dtype=floatX) % np.cast[floatX](2))
        target_dimshuffle = ('x',) * axis + (0,) + ('x',) * \
            (len(input_shape) - 1 - axis)
        # print("make_ghh_pool_conv2d: target_dimshuffle is {}".format(target_dimshuffle))
        delta = delta.flatten().dimshuffle(*target_dimshuffle)

        res_after_sum = T.sum(res_after_max * delta, axis=axis)

        return res_after_sum

    return pool_function


def make_ghh_soft_pool_conv2d(num_in_sum, num_in_max, max_strength):

    def pool_function(input, axis):

        input_shape = tuple(input.shape)
        num_feature_maps_out = input_shape[axis - 1]
        pool_size = input_shape[axis]

        pool_shape = (input_shape[:axis] + (num_in_sum,
                                            num_in_max) + input_shape[axis + 1:])
        # print("make_ghh_pool_conv2d: pool_shape is {}".format(pool_shape))
        input_reshaped = input.reshape(pool_shape)

        # raise NotImplementedError('TODO: use a soft max instead of T.max')
        # res_after_max = T.max(input_reshaped,axis=axis+1)

        # Soft max with strength of max_strength
        res_after_max = np.cast[floatX](1.0) / np.cast[floatX](max_strength) \
            * T.log(T.mean(T.exp(max_strength * (input_reshaped - T.max(input_reshaped, axis=axis + 1, keepdims=True))), axis=axis + 1)) \
            + T.max(input_reshaped, axis=axis + 1)

        # Get deltas
        delta = np.cast[floatX](1.0) - np.cast[floatX](2.0) * \
            (T.arange(num_in_sum, dtype=floatX) % np.cast[floatX](2))
        target_dimshuffle = ('x',) * axis + (0,) + ('x',) * \
            (len(input_shape) - 1 - axis)
        # print("make_ghh_pool_conv2d: target_dimshuffle is {}".format(target_dimshuffle))
        delta = delta.flatten().dimshuffle(*target_dimshuffle)

        res_after_sum = T.sum(res_after_max * delta, axis=axis)

        return res_after_sum

    # print('TODO: softmax needs debugging, produces NaNs!')

    return pool_function


class GHHActivationLayer(Layer):
    """
    TODO: write me

    adapted from the ParametricRectifierLayer code

    """

    def __init__(self, incoming, num_in_sum, num_in_max, max_strength=-1,
                 alpha=lasagne.init.Normal(0.05), beta=lasagne.init.Constant(0.),
                 noise_sigma=0.0, shared_axes='auto',
                 **kwargs):
        super(GHHActivationLayer, self).__init__(incoming, **kwargs)
        if shared_axes == 'auto':
            self.shared_axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif shared_axes == 'all':
            self.shared_axes = tuple(range(len(self.input_shape)))
        elif isinstance(shared_axes, int):
            self.shared_axes = (shared_axes,)
        else:
            self.shared_axes = shared_axes

        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        self.num_in_sum = num_in_sum
        self.num_in_max = num_in_max
        self.max_strength = max_strength
        self.noise_sigma = noise_sigma

        # Shape of a single parameter
        single_shape = [size for axis, size in enumerate(self.input_shape)
                        if axis not in self.shared_axes]
        if any(size is None for size in single_shape):
            raise ValueError("GHHActivationLayer needs input sizes for "
                             "all axes that alpha's are not shared over.")

        # Shape of entire alpha and beta
        # shape = single_shape + [self.num_in_sum,self.num_in_max-1] # we use
        # the original output in max to avoid diminishing grads
        shape = single_shape + [self.num_in_sum, self.num_in_max]

        # dimshuffle pattern for input
        self.input_pattern = ['x', 'x'] + range(len(self.input_shape))

        # dimshuffle pattern for alpha and beta
        axes = iter(range(len(single_shape)))
        single_pattern = ['x' if input_axis in self.shared_axes else next(
            axes) for input_axis in six.moves.xrange(len(self.input_shape))]
        self.param_pattern = [1, 2] + single_pattern

        self.alpha = self.add_param(alpha, shape, name="alpha",
                                    regularizable=True)  # we want alpha to be regularizable as it will affect output range
        self.beta = self.add_param(beta, shape, name="beta",
                                   regularizable=False)

    def get_output_for(self, input, deterministic=False, **kwargs):

        input = input.dimshuffle(self.input_pattern)
        alpha = self.alpha.dimshuffle(self.param_pattern)
        beta = self.beta.dimshuffle(self.param_pattern)

        # # we use the original output in max to avoid diminishing grads
        # res = T.concatenate([input * alpha + beta, T.tile(input, [self.num_in_sum] + [1]*(len(self.input_shape)+1))],axis=1)
        res = input * alpha + beta
        if not (deterministic or self.noise_sigma == 0):
            res = res + \
                self._srng.normal(res.shape, avg=0.0, std=self.noise_sigma)

        # Do the max
        if self.max_strength < 0:
            # Hard max
            res_after_max = T.max(res, axis=1)
        elif self.max_strength > 0:
            # Soft max with strength of max_strength
            res_after_max = np.cast[floatX](1.0) / np.cast[floatX](self.max_strength) \
                * T.log(T.mean(T.exp(self.max_strength * (res - T.max(res, axis=1, keepdims=True))), axis=1)) \
                + T.max(res, axis=1)

            # print('TODO: softmax needs debugging, produces NaNs!')

        else:
            raise ValueError(
                'GHHActivationLayer: max_strength should not be zero!')

        # Get deltas
        delta = np.cast[floatX](1.0) - np.cast[floatX](2.0) * \
            (T.arange(self.num_in_sum, dtype=floatX) % np.cast[floatX](2))
        target_dimshuffle = [0] + ['x'] * (res_after_max.ndim - 1)
        # print("make_ghh_pool_conv2d: target_dimshuffle is {}".format(target_dimshuffle))
        delta = delta.flatten().dimshuffle(target_dimshuffle)

        # Do the sum
        res_after_sum = T.sum(res_after_max * delta, axis=0)

        return res_after_sum


#
# ghh_pool.py ends here
