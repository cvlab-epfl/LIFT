# bypass.py ---
#
# Filename: bypass.py
# Description:
# Author: Kwang
# Maintainer:
# Created: Thu Feb 18 20:26:56 2016 (+0100)
# Version:
# Package-Requires: ()
# URL:
# Doc URL:
# Keywords:
# Compatibility:
#
#

# Commentary: This one is just for the testing module to not really
# create complicated graphs.
#
#
#
#

# Change Log:
#
#
#
# Copyright (C), EPFL Computer Vision Lab.

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
    # Bypass for descriptor (just use the labels.... avoid computations)

    # overwrite the input as well
    myNet.layers[idxSiam]['desc-input'] = InputLayer(
        (myNet.config.batch_size, ),
        input_var=myNet.y[idxSiam],
        name='desc-input')

    myNet.layers[idxSiam]['desc-output'] = ReshapeLayer(
        myNet.layers[idxSiam]['desc-input'],
        (myNet.config.batch_size, 1),
        name='desc-output')
#
# bypass.py ends here
