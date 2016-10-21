# norm_patch.py ---
#
# Filename: norm_patch.py
# Description:
# Author: Eduard
# Maintainer:
# Created: Sun Feb  7 22:39:35 2016 (+0100)
# Version:
# Package-Requires: ()
# Last-Updated:
#           By:
#     Update #: 11
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
import six
import theano
from lasagne import nonlinearities
from lasagne.layers import (Conv2DLayer, ExpressionLayer, FlattenLayer,
                            MaxPool2DLayer, NonlinearityLayer)

import Utils.dump_tools as dt
from Utils.lasagne_tools import LPPool2DLayer, SubtractiveNormalization2DLayer

floatX = theano.config.floatX

# Disable future warnings (caused by theano)
warnings.simplefilter(action="ignore", category=FutureWarning)


def build(myNet, idxSiam, verbose=True):

    # Load model
    fn = '%s/%s' % (myNet.config.descriptor_export_folder,
                    myNet.config.descriptor_model)
    model_dict = dt.loadh5(fn)

    # Load training mean/std

    # if we have the normalization setup
    if myNet.config.bNormalizeInput:
        kwang_mean = np.cast[floatX](myNet.config.mean_x)
        kwang_std = np.cast[floatX](myNet.config.std_x)
    # else, simply divide with 255
    else:
        kwang_mean = np.cast[floatX](0.0)
        kwang_std = np.cast[floatX](255.0)

    if 'patch-mean' in model_dict.keys():
        desc_mean_x = np.cast[floatX](model_dict['patch-mean'][0])
        desc_std_x = np.cast[floatX](model_dict['patch-std'][0])
    else:
        print('Warning: no mean/std in the model file')
        desc_mean_x = kwang_mean
        desc_std_x = kwang_std

    # Layer indices
    indices = model_dict['layers']
    if verbose and idxSiam == 0:
        print('*** Loading descriptor "%s" ***' % fn)
        print('Number of elements: %d' % indices.size)

    # Add another layer that transforms the original input
    curr_name = 'desc-re-normalize'
    curr_input = myNet.config.descriptor_input
    myNet.layers[idxSiam][curr_name] = ExpressionLayer(
        myNet.layers[idxSiam][curr_input],
        lambda x: (x * kwang_std + kwang_mean - desc_mean_x) / desc_std_x,
        name=curr_name
    )
    curr_input = curr_name

    # Loop over layers
    for i in six.moves.xrange(indices.size):
        if indices[i] == 1:
            if verbose and idxSiam == 0:
                print('%d -> SpatialConvolution' % i)
            curr_name = 'desc-%d-conv' % (i + 1)
            # read actual value for siamese 0
            w = model_dict['l-%d-weights' % (i + 1)].astype(floatX)
            b = model_dict['l-%d-bias' % (i + 1)].astype(floatX)
            num_filters, num_input_channels, filter_size, filter_size = w.shape
            # assert num_input_channels == myNet.config.nDescInputChannels
            # assert filter_size == myNet.config.nDescInputSize
            if verbose and idxSiam == 0:
                print('  Number of filters: %d' % num_filters)
                print('  Filter size: %d' % filter_size)
            # Manually create shared variables
            if idxSiam == 0:
                w = theano.shared(w, name=curr_name + '.W')
                b = theano.shared(b, name=curr_name + '.b')
            else:
                w = myNet.layers[0][curr_name].W
                b = myNet.layers[0][curr_name].b
            myNet.layers[idxSiam][curr_name] = Conv2DLayer(
                myNet.layers[idxSiam][curr_input],
                num_filters,
                filter_size,
                W=w,
                b=b,
                nonlinearity=None,  # no activation
                flip_filters=False,
                name=curr_name
            )

        elif indices[i] == 2:
            if verbose and idxSiam == 0:
                print('%d -> Linear' % i)
            raise RuntimeError('Layer type %d TODO' % i)

        elif indices[i] == 3:
            if verbose and idxSiam == 0:
                print('%d -> SpatialMaxPooling' % i)
            curr_name = 'desc-%d-maxpool' % (i + 1)
            kw = model_dict['l-%d-kw' % (i + 1)].astype(np.int32)[0]
            kh = model_dict['l-%d-kh' % (i + 1)].astype(np.int32)[0]
            if verbose and idxSiam == 0:
                print('  Region size: %dx%d' % (kw, kh))
            assert kw == kh
            kw = int(kw)

            myNet.layers[idxSiam][curr_name] = MaxPool2DLayer(
                myNet.layers[idxSiam][curr_input],
                pool_size=kw,
                stride=None,
                name=curr_name
            )

        elif indices[i] == 4:
            if verbose and idxSiam == 0:
                print('%d -> SpatialLPPooling' % i)
            curr_name = 'desc-%d-lppool' % (i + 1)
            kw = model_dict['l-%d-kw' % (i + 1)].astype(np.int32)[0]
            kh = model_dict['l-%d-kh' % (i + 1)].astype(np.int32)[0]
            if verbose and idxSiam == 0:
                print('  Region size: %dx%d' % (kw, kh))
            assert kw == kh
            kw = int(kw)

            myNet.layers[idxSiam][curr_name] = LPPool2DLayer(
                myNet.layers[idxSiam][curr_input],
                pnorm=2,
                pool_size=kw,
                stride=None,
                name=curr_name
            )

        elif indices[i] == 5:
            if verbose and idxSiam == 0:
                print('%d -> Tanh' % i)
            curr_name = 'desc-%d-tanh' % (i + 1)
            myNet.layers[idxSiam][curr_name] = NonlinearityLayer(
                myNet.layers[idxSiam][curr_input],
                nonlinearity=nonlinearities.tanh,
                name=curr_name
            )

        elif indices[i] == 6:
            if verbose and idxSiam == 0:
                print('%d -> ReLU' % i)
            curr_name = 'desc-%d-relu' % (i + 1)
            myNet.layers[idxSiam][curr_name] = NonlinearityLayer(
                myNet.layers[idxSiam][curr_input],
                nonlinearity=nonlinearities.rectify,
                name=curr_name
            )

        elif indices[i] == 7:
            if verbose and idxSiam == 0:
                print('%d -> SpatialSubtractiveNormalization' % i)
            curr_name = 'desc-%d-subt-norm' % (i + 1)
            kernel = model_dict['l-%d-filter' % (i + 1)].astype(floatX)
            w_kernel, h_kernel = kernel.shape
            if verbose and idxSiam == 0:
                print('  Kernel size: %dx%d' % (w_kernel, h_kernel))

            myNet.layers[idxSiam][curr_name] = SubtractiveNormalization2DLayer(
                myNet.layers[idxSiam][curr_input],
                kernel=kernel,
                name=curr_name
            )

        else:
            raise RuntimeError('Layer type %d: unknown' % i)

        # Input to the next layer
        curr_input = curr_name

    # Flatten output and rename
    myNet.layers[idxSiam]['desc-output'] = FlattenLayer(
        myNet.layers[idxSiam][curr_input],
        outdim=2
    )
