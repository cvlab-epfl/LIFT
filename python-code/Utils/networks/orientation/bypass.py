# bypass.py ---
#
# Filename: bypass.py
# Description:
# Author: Kwang
# Maintainer:
# Created: Fri Feb 12 17:24:52 2016 (+0100)
# Version:
# Package-Requires: ()
# Last-Updated: Fri Feb 12 17:26:44 2016 (+0100)
#           By: Kwang
#     Update #: 6
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

from lasagne.layers import InputLayer, ReshapeLayer

# Disable future warnings (caused by theano)
warnings.simplefilter(action="ignore", category=FutureWarning)

floatX = theano.config.floatX


def build(myNet, idxSiam, verbose=True):

    myNet.layers[idxSiam]['ori-bypass-input'] = InputLayer(
        (myNet.config.batch_size, ),
        input_var=myNet.angle[idxSiam],
        name='ori-bypass-input')

    myNet.layers[idxSiam]['ori-output'] = ReshapeLayer(
        myNet.layers[idxSiam]['ori-bypass-input'],
        shape=[myNet.config.batch_size, 1],
        name='ori-output')

#
# bypass.py ends here
