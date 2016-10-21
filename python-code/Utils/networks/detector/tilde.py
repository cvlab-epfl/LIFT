# tilde.py ---
#
# Filename: tilde.py
# Description:
# Author: Kwang
# Maintainer:
# Created: Fri Feb  5 19:25:14 2016 (+0100)
# Version:
# Package-Requires: ()
# Last-Updated:
#           By:
#     Update #: 27
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

import warnings

import numpy as np
import theano
import theano.tensor as T
from lasagne.init import Constant, HeNormal
from lasagne.layers import (DropoutLayer, ExpressionLayer, GaussianNoiseLayer,
                            NonlinearityLayer, get_output_shape)
from lasagne.layers.dnn import Conv3DDNNLayer as Conv3DLayer

from Utils.ghh_pool import GHHFeaturePoolLayer
from Utils.lasagne_tools import createXYZMapLayer

# Disable future warnings (caused by theano)
warnings.simplefilter(action="ignore", category=FutureWarning)

floatX = theano.config.floatX


def build(myNet, idxSiam, verbose=True):

    # # custom activations
    # relu = lambda x: x * (x > 0)  # from theano example
    # # stable softmax
    # softmax = lambda x: T.exp(x) \
    #     / (T.exp(x).sum(1, keepdims=True, dtype=floatX))
    # # log-soft-max
    # log_softmax = lambda x: (x - x.max(1, keepdims=True)) \
    #     - T.log(T.sum(
    #         T.exp(x - x.max(1, keepdims=True)),
    #         axis=1, keepdims=True, dtype=floatX
    #     ))

    # batch_size = myNet.config.batch_size
    # num_channel = myNet.config.num_channel
    # output_dim = myNet.config.out_dim

    INITIALIZATION_GAIN = 1.0
    # INITIALIZATION_GAIN = 0.0
    BIAS_RND = myNet.config.bias_rnd

    # ---------------------------------------------------------------------
    # Dropout on the input (no rescaling!)
    fInputDroprate = getattr(myNet.config, 'fInputDroprate', 0.0)
    myNet.layers[idxSiam]['kp-inputdrop'] = DropoutLayer(
        myNet.layers[idxSiam]['kp-input'],
        p=np.cast[floatX](fInputDroprate),
        rescale=False,
        name='kp-inputdrop')

    # ---------------------------------------------------------------------
    # convolution sharing weights
    # shape of fileter
    if 'nFilterScaleSize' in myNet.config.__dict__.keys():
        fs = [myNet.config.nFilterSize, myNet.config.nFilterSize,
              myNet.config.nFilterScaleSize]
    else:
        fs = [myNet.config.nFilterSize, myNet.config.nFilterSize, 1]
    ns = 4  # num in sum
    nm = 4  # num in max
    nu = 1  # num units after Feuture pooling
    if idxSiam == 0:
        W_init = HeNormal(gain=INITIALIZATION_GAIN)
        # W_init = Constant(0.0)
        b_init = Constant(0.0)
    else:
        W_init = myNet.layers[0]['kp-c0'].W
        b_init = myNet.layers[0]['kp-c0'].b
    # For testing 3D2D convolution
    if 'bTestConv3D2D' in myNet.config.__dict__.keys():
        raise RuntimeError('Deprecated!')
    myNet.layers[idxSiam]['kp-c0'] = Conv3DLayer(
        myNet.layers[idxSiam]['kp-inputdrop'],
        num_filters=nu * ns * nm,
        filter_size=fs,
        nonlinearity=None,
        W=W_init,
        b=b_init,
        name='kp-c0', )
    # noise layer
    myNet.layers[idxSiam]['kp-c0n'] = GaussianNoiseLayer(
        myNet.layers[idxSiam]['kp-c0'],
        sigma=BIAS_RND,
        name='kp-c0n', )
    # GHH pooling activation
    myNet.layers[idxSiam]['kp-c0a'] = GHHFeaturePoolLayer(
        myNet.layers[idxSiam]['kp-c0n'],
        num_in_sum=ns,
        num_in_max=nm,
        axis=1,
        max_strength=myNet.config.max_strength,
        name='kp-c0a', )

    # # -------------------------------------------------------------------
    # # Fully connected with sharing weights
    # if idxSiam == 0:
    #     W_init = HeNormal(gain=INITIALIZATION_GAIN)
    #     # W_init = Constant(0.0)
    #     b_init = Constant(0.0)
    # else:
    #     W_init = myNet.layers[0]['output'].W
    #     b_init = myNet.layers[0]['output'].b
    # myNet.layers[idxSiam]['output'] = DenseLayer(
    #     myNet.layers[idxSiam]['f3a'],
    #     num_units=10,
    #     nonlinearity=log_softmax,
    #     W=W_init, b=b_init, name='output'
    # )

    final_nonlinearity = getattr(myNet.config, 'sKpNonlinearity', 'None')
    if verbose and idxSiam == 0:
        print(' -- kp_info: nonlinearity == ' + final_nonlinearity)
    if final_nonlinearity == 'None':
        final_nonlinearity = None
    elif final_nonlinearity == 'tanh':
        final_nonlinearity = T.tanh
    else:
        raise ValueError('Unsupported nonlinearity!')

    myNet.layers[idxSiam]['kp-scoremap'] = NonlinearityLayer(
        myNet.layers[idxSiam]['kp-c0a'],
        nonlinearity=final_nonlinearity,
        name='kp-scoremap', )

    # ---------------------------------------------------------------------
    # Layer for cropping to keep desc part within boundary
    rf = np.cast[floatX](float(myNet.config.nPatchSizeKp) /
                         float(myNet.config.nPatchSize))
    input_shape = get_output_shape(myNet.layers[0]['kp-input'])
    uncut_shape = get_output_shape(myNet.layers[0]['kp-scoremap'])
    req_boundary = np.ceil(rf * np.sqrt(2) * myNet.config.nDescInputSize /
                           2.0).astype(int)
    cur_boundary = (input_shape[2] - uncut_shape[2]) // 2
    crop_size = req_boundary - cur_boundary

    if verbose and idxSiam == 0:
        resized_shape = get_output_shape(myNet.layers[0]['kp-input'])
        print(' -- kp_info: output score map shape {}'.format(uncut_shape))
        print(' -- kp_info: input size after resizing {}'.format(resized_shape[
            2]))
        print(' -- kp_info: output score map size {}'.format(uncut_shape[2]))
        print(' -- kp info: required boundary {}'.format(req_boundary))
        print(' -- kp info: current boundary {}'.format(cur_boundary))
        print(' -- kp_info: additional crop size {}'.format(crop_size))
        print(' -- kp_info: additional crop size {}'.format(crop_size))
        print(' -- kp_info: final cropped score map size {}'.format(
            uncut_shape[2] - 2 * crop_size))
        print(' -- kp_info: movement ratio will be {}'.format((
            float(uncut_shape[2] - 2.0 * crop_size) /
            float(myNet.config.nPatchSizeKp - 1))))

    def crop(out, crop_size):

        return out[:, :, crop_size:-crop_size, crop_size:-crop_size, :]

    def cropfunc(out):

        return crop(out, crop_size)

    myNet.layers[idxSiam]['kp-scoremap-cut'] = ExpressionLayer(
        myNet.layers[idxSiam]['kp-scoremap'],
        cropfunc,
        output_shape='auto',
        name='kp-scoremap-cut', )

    # ---------------------------------------------------------------------
    # Mapping layer to x,y,z
    myNet.layers[idxSiam]['kp-output'] = createXYZMapLayer(
        myNet.layers[idxSiam]['kp-scoremap-cut'],
        fScaleList=myNet.config.fScaleList,
        req_boundary=req_boundary,
        name='kp-output',
        fCoMStrength=10.0,
        eps=myNet.config.epsilon)

#
# tilde.py ends here
