# custom_theano.py ---
#
# Filename: custom_theano.py
# Description:
# Author: Kwang
# Maintainer:
# Created: Wed Aug 26 16:33:33 2015 (+0200)
# Version:
# Package-Requires: ()
# Last-Updated: Wed Oct 19 18:21:52 2016 (+0200)
#           By: kwang
#     Update #: 92
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

import numpy as np
import theano
from theano import scalar as scal
from theano import printing
from theano.printing import pprint
from theano.scalar import get_scalar_type, neg, sqr
from theano.tensor import elemwise

# ------------------------------------------------------------------------
# Types
int8 = get_scalar_type('int8')
int16 = get_scalar_type('int16')
int32 = get_scalar_type('int32')
int64 = get_scalar_type('int64')
uint8 = get_scalar_type('uint8')
uint16 = get_scalar_type('uint16')
uint32 = get_scalar_type('uint32')
uint64 = get_scalar_type('uint64')
float32 = get_scalar_type('float32')
float64 = get_scalar_type('float64')
complex64 = get_scalar_type('complex64')
complex128 = get_scalar_type('complex128')

int_types = int8, int16, int32, int64
uint_types = uint8, uint16, uint32, uint64
float_types = float32, float64
complex_types = complex64, complex128

discrete_types = int_types + uint_types
continuous_types = float_types + complex_types
all_types = discrete_types + continuous_types

# ------------------------------------------------------------------------
# Utils


def _scal_elemwise_with_nfunc(nfunc, nin, nout):
    """
    Replace a symbol definition with an elementwise version of the
    corresponding scalar Op.  If it is not None, the nfunc argument
    should be a string such that getattr(numpy, nfunc) implements
    a vectorized version of the elemwise operation. nin is the number
    of inputs expected by that function, and nout is the number of
    **destination** inputs it takes. That is, the function should
    take nin+nout inputs. nout == 0 means that the numpy function
    does not take a numpy array argument to put its result in.
    """

    def construct(symbol):
        symbolname = symbol.__name__
        inplace = symbolname.endswith('_inplace')
        if inplace:
            msg = "inplace"
        else:
            msg = "no_inplace"

        n = "Elemwise{%s,%s}" % (symbolname, msg)

        if inplace:
            scalar_op = getattr(scal, symbolname[:-len('_inplace')])
            inplace_scalar_op = scalar_op.__class__(scal.transfer_type(0))
            rval = elemwise.Elemwise(inplace_scalar_op,
                                     {0: 0},
                                     name=n,
                                     nfunc_spec=(nfunc and (nfunc, nin, nout)))

        else:
            scalar_op = getattr(scal, symbolname)
            rval = elemwise.Elemwise(scalar_op,
                                     name=n,
                                     nfunc_spec=(nfunc and (nfunc, nin, nout)))

        if getattr(symbol, '__doc__', False):
            rval.__doc__ = symbol.__doc__ + '\n' + rval.__doc__

        # for the meaning of this see the ./epydoc script
        # it makes epydoc display rval as if it were a function, not an object
        rval.__epydoc_asRoutine = symbol
        rval.__module__ = 'tensor'

        pprint.assign(rval, printing.FunctionPrinter(symbolname))

        return rval

    return construct


# ------------------------------------------------------------------------
# Custom Ops
class CustomArcTan2(theano.scalar.basic.BinaryScalarOp):

    def impl(self, y, x):
        # If x and y are int8 or uint8, numpy.arctan2 will compute the result
        # in half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, 'dtype', ''))
        if x_dtype in ('int8', 'uint8'):
            y_dtype = str(getattr(x, 'dtype', ''))
            if y_dtype in ('int8', 'uint8'):
                return np.arctan2(y, x, sig='f')
        return np.arctan2(y, x)

    def grad(self, inputs, gout):
        (y, x) = inputs
        (gz, ) = gout
        if gz.type in complex_types:
            raise NotImplementedError()
        else:
            if self(x, y).type in discrete_types:
                if x.type in discrete_types:
                    gx = x.zeros_like(dtype=theano.config.floatX)
                else:
                    gx = x.zeros_like()
                if y.type in discrete_types:
                    gy = y.zeros_like(dtype=theano.config.floatX)
                else:
                    gy = y.zeros_like()
                return [gx, gy]

            eps = np.cast[x.type](1e-15)
            # If the output is float, the gradient should flow,
            # even if the inputs are ints
            return [gz * x / (sqr(x) + sqr(y) + eps),
                    gz * neg(y) / (sqr(x) + sqr(y) + eps)]

    def c_code(self, node, name, inputs, outputs, sub):
        (y, x) = inputs
        (z, ) = outputs
        if (node.inputs[0].type in complex_types or
                node.inputs[1].type in complex_types):
            raise NotImplementedError('type not supported', type)
        return "%(z)s = atan2(%(y)s, %(x)s);" % locals()


custom_arctan2 = CustomArcTan2(theano.scalar.basic.upgrade_to_float,
                               name='custom_arctan2')

# replace the function at theano scalar module
setattr(theano.scalar, 'custom_arctan2', custom_arctan2)


@_scal_elemwise_with_nfunc(None, 1, 1)
def custom_arctan2(a, b):
    """custom arctangent of a / b"""

# replace the function at theano tensor module
setattr(theano.tensor, 'custom_arctan2', custom_arctan2)


#
# custom_theano.py ends here
