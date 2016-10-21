# bypass.py ---
#
# Filename: bypass.py
# Description:
# Author: Kwang
# Maintainer:
# Created: Fri Feb 12 17:24:52 2016 (+0100)
# Version:
# Package-Requires: ()
# Last-Updated: Tue Feb 16 14:13:17 2016 (+0100)
#           By: Kwang
#     Update #: 14
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

from lasagne.layers import ExpressionLayer, InputLayer, ReshapeLayer

# Disable future warnings (caused by theano)
warnings.simplefilter(action="ignore", category=FutureWarning)

floatX = theano.config.floatX


def build(myNet, idxSiam, verbose=True):

    # -------------------------------------------------------------------------
    # Bypass for score map
    myNet.layers[idxSiam]['kp-bypass-input-score'] = InputLayer(
        (myNet.config.batch_size, ),
        input_var=myNet.y[idxSiam],
        name='kp-bypass-input-score')

    myNet.layers[idxSiam]['kp-scoremap-cut'] = ExpressionLayer(
        myNet.layers[idxSiam]['kp-bypass-input-score'],
        lambda x: x.reshape([myNet.config.batch_size, 1]) * 2.0 - 1.0,
        output_shape=[myNet.config.batch_size, 1],
        name='kp-scoremap-cut')

    myNet.layers[idxSiam]['kp-scoremap'] = ExpressionLayer(
        myNet.layers[idxSiam]['kp-bypass-input-score'],
        lambda x: x.reshape([myNet.config.batch_size, 1]) * 2.0 - 1.0,
        output_shape=[myNet.config.batch_size, 1],
        name='kp-scoremap')

    # -------------------------------------------------------------------------
    # Bypass for xyz coordinates
    myNet.layers[idxSiam]['kp-bypass-input-xyz'] = InputLayer(
        (myNet.config.batch_size, 3),
        input_var=myNet.pos[idxSiam],
        name='kp-bypass-input-xyz')

    myNet.layers[idxSiam]['kp-output'] = ExpressionLayer(
        myNet.layers[idxSiam]['kp-bypass-input-xyz'],
        # lambda x: x + np.asarray([0.5, 0.5, 1],
        #                          dtype=floatX).reshape([1, 3]),
        lambda x: x,
        output_shape=[myNet.config.batch_size, 3],
        name='kp-output')
#
# bypass.py ends here
