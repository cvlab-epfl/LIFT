# cvpr16.py ---
#
# Filename: cvpr16.py
# Description:
# Author: Kwang
# Maintainer:
# Created: Sun Feb  7 22:37:14 2016 (+0100)
# Version:
# Package-Requires: ()
# Last-Updated: Mon Oct 17 19:38:29 2016 (+0200)
#           By: kwang
#     Update #: 30
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

import theano
from lasagne.init import Constant, HeNormal
from lasagne.layers import (Conv2DLayer, DenseLayer, DropoutLayer,
                            ExpressionLayer, MaxPool2DLayer, NonlinearityLayer)
from lasagne.nonlinearities import rectify as relu

import Utils.custom_theano as CT
from Utils.ghh_pool import GHHFeaturePoolLayer

# from lasagne.layers.dnn import Conv3DDNNLayer as Conv3DLayer

# Disable future warnings (caused by theano)
warnings.simplefilter(action="ignore", category=FutureWarning)

floatX = theano.config.floatX


def build(myNet, idxSiam, verbose=True):

    INITIALIZATION_GAIN = 1.0

    # -----------------------------------------------------------------------------
    # input layer (2d croped patch)
    # myNet.layers[idxSiam]['ori-input']

    # -----------------------------------------------------------------------------
    # 3x Convolution and Max Pooling layers

    # --------------
    # Conv 0
    if idxSiam == 0:
        W_init = HeNormal(gain=INITIALIZATION_GAIN)
        # W_init = Constant(0.0)
        b_init = Constant(0.0)
    else:
        W_init = myNet.layers[0]['ori-c0'].W
        b_init = myNet.layers[0]['ori-c0'].b
    myNet.layers[idxSiam]['ori-c0'] = Conv2DLayer(
        myNet.layers[idxSiam]['ori-input'],
        num_filters=10,
        filter_size=5,
        W=W_init,
        b=b_init,
        nonlinearity=None,
        flip_filters=False,
        name='ori-c0',
    )
    # Activation 0
    myNet.layers[idxSiam]['ori-c0a'] = NonlinearityLayer(
        myNet.layers[idxSiam]['ori-c0'],
        nonlinearity=relu,
        name='ori-c0a',
    )
    # Pool 0
    myNet.layers[idxSiam]['ori-c0p'] = MaxPool2DLayer(
        myNet.layers[idxSiam]['ori-c0a'],
        pool_size=2,
        name='ori-c0p',
    )

    # --------------
    # Conv 1
    if idxSiam == 0:
        W_init = HeNormal(gain=INITIALIZATION_GAIN)
        # W_init = Constant(0.0)
        b_init = Constant(0.0)
    else:
        W_init = myNet.layers[0]['ori-c1'].W
        b_init = myNet.layers[0]['ori-c1'].b
    myNet.layers[idxSiam]['ori-c1'] = Conv2DLayer(
        myNet.layers[idxSiam]['ori-c0p'],
        num_filters=20,
        filter_size=5,
        W=W_init,
        b=b_init,
        nonlinearity=None,
        flip_filters=False,
        name='ori-c1',
    )
    # Activation 1
    myNet.layers[idxSiam]['ori-c1a'] = NonlinearityLayer(
        myNet.layers[idxSiam]['ori-c1'],
        nonlinearity=relu,
        name='ori-c1a',
    )
    # Pool 1
    myNet.layers[idxSiam]['ori-c1p'] = MaxPool2DLayer(
        myNet.layers[idxSiam]['ori-c1a'],
        pool_size=2,
        name='ori-c1p',
    )

    # --------------
    # Conv 2
    if idxSiam == 0:
        W_init = HeNormal(gain=INITIALIZATION_GAIN)
        # W_init = Constant(0.0)
        b_init = Constant(0.0)
    else:
        W_init = myNet.layers[0]['ori-c2'].W
        b_init = myNet.layers[0]['ori-c2'].b
    myNet.layers[idxSiam]['ori-c2'] = Conv2DLayer(
        myNet.layers[idxSiam]['ori-c1p'],
        num_filters=50,
        filter_size=3,
        W=W_init,
        b=b_init,
        nonlinearity=None,
        flip_filters=False,
        name='ori-c2',
    )
    # Activation 2
    myNet.layers[idxSiam]['ori-c2a'] = NonlinearityLayer(
        myNet.layers[idxSiam]['ori-c2'],
        nonlinearity=relu,
        name='ori-c2a',
    )
    # Pool 2
    myNet.layers[idxSiam]['ori-c2p'] = MaxPool2DLayer(
        myNet.layers[idxSiam]['ori-c2a'],
        pool_size=2,
        name='ori-c2p',
    )

    # -----------------------------------------------------------------------------
    # Fully Connected Layers

    # --------------
    # FC 3
    nu = 100
    ns = 4
    nm = 4
    if idxSiam == 0:
        W_init = HeNormal(gain=INITIALIZATION_GAIN)
        # W_init = Constant(0.0)
        b_init = Constant(0.0)
    else:
        W_init = myNet.layers[0]['ori-f3'].W
        b_init = myNet.layers[0]['ori-f3'].b
    myNet.layers[idxSiam]['ori-f3'] = DenseLayer(
        myNet.layers[idxSiam]['ori-c2a'],
        num_units=nu * ns * nm,
        W=W_init,
        b=b_init,
        nonlinearity=None,
        name='ori-f3',
    )
    # Activation 3
    myNet.layers[idxSiam]['ori-f3a'] = GHHFeaturePoolLayer(
        myNet.layers[idxSiam]['ori-f3'],
        num_in_sum=ns,
        num_in_max=nm,
        max_strength=myNet.config.max_strength,
        name='ori-f3a',
    )
    # Dropout 3
    myNet.layers[idxSiam]['ori-f3d'] = DropoutLayer(
        myNet.layers[idxSiam]['ori-f3a'],
        p=0.3,
        name='ori-f3d',
    )

    # --------------
    # FC 4
    nu = 2
    ns = 4
    nm = 4
    if idxSiam == 0:
        W_init = HeNormal(gain=INITIALIZATION_GAIN)
        # W_init = Constant(0.0)
        b_init = Constant(0.0)
    else:
        W_init = myNet.layers[0]['ori-f4'].W
        b_init = myNet.layers[0]['ori-f4'].b
    myNet.layers[idxSiam]['ori-f4'] = DenseLayer(
        myNet.layers[idxSiam]['ori-f3d'],
        num_units=nu * ns * nm,
        W=W_init,
        b=b_init,
        nonlinearity=None,
        name='ori-f4',
    )
    # Activation 4
    myNet.layers[idxSiam]['ori-f4a'] = GHHFeaturePoolLayer(
        myNet.layers[idxSiam]['ori-f4'],
        num_in_sum=ns,
        num_in_max=nm,
        max_strength=myNet.config.max_strength,
        name='ori-f4a',
    )

    # -----------------------------------------------------------------------------
    # Arctan2 Layer
    myNet.layers[idxSiam]['ori-output'] = ExpressionLayer(
        myNet.layers[idxSiam]['ori-f4a'],
        lambda x: CT.custom_arctan2(
            x[:, 0], x[:, 1]).flatten().dimshuffle(0, 'x'),
        output_shape=(myNet.config.batch_size, 1),
        name='ori-output',
    )

#
# cvpr16.py ends here
