# math_tools.py ---
#
# Filename: math_tools.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Fri Feb 19 18:04:58 2016 (+0100)
# Version:
# Package-Requires: ()
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
# Copyright (C), EPFL Computer Vision Lab.

# Code:


import numpy as np


def softmax(val, axis, softmax_strength):
    ''' Soft max function used for cost function '''

    import theano
    import theano.tensor as T
    floatX = theano.config.floatX

    if softmax_strength < 0:
        res_after_max = T.max(val, axis=axis)
    else:
        res_after_max = np.cast[floatX](1.0) / softmax_strength \
            * T.log(T.mean(T.exp(
                softmax_strength * (val - T.max(val,
                                                axis=axis,
                                                keepdims=True))
            ), axis=axis)) \
            + T.max(val, axis=axis)
    return res_after_max


def softargmax(val, axis, softargmax_strength):
    ''' Soft argmax function used for cost function '''

    import theano
    import theano.tensor as T
    floatX = theano.config.floatX

    # The implmentation only works for single axis
    assert isinstance(axis, int)

    if softargmax_strength < 0:
        res = T.argmax(val, axis=axis)
    else:
        safe_exp = T.exp(softargmax_strength * (
            val - T.max(val, axis=axis, keepdims=True)))
        prob = safe_exp / T.sum(safe_exp, axis=axis, keepdims=True)
        ind = T.arange(val.shape[axis], dtype=floatX)
        res = T.sum(prob * ind, axis=axis) / T.sum(prob, axis=axis)

    return res


def _subtractive_norm_make_coef(norm_kernel, input_hw):
    """Creates the coef matrix for accounting borders when applying
    subtractive normalization.

    Parameters
    ----------
    norm_kernel : np.ndarray, 2d
        Normalized kernel applied for subtractive normalization.

    input_hw : tuple
        Height and width of the input image (patch) for the
        SubtractiveNormalizationLayer.

    """

    assert np.isclose(np.sum(norm_kernel), 1.0)

    # This allows our mean computation to compensate for the border area,
    # where you have less terms adding up. Torch used convolution with a
    # ``one'' image, but since we do not want the library to depend on
    # other libraries with convolutions, we do it manually here.
    coef = np.ones(input_hw, dtype='float32')
    pad_x = norm_kernel.shape[1] // 2
    pad_y = norm_kernel.shape[0] // 2

    # Corners
    # for the top-left corner
    tl_cumsum_coef = np.cumsum(np.cumsum(
        norm_kernel[::-1, ::-1], axis=0), axis=1)[::1, ::1]
    coef[:pad_y + 1, :pad_x + 1] = tl_cumsum_coef[pad_y:, pad_x:]
    # for the top-right corner
    tr_cumsum_coef = np.cumsum(np.cumsum(
        norm_kernel[::-1, ::1], axis=0), axis=1)[::1, ::-1]
    coef[:pad_y + 1, -pad_x - 1:] = tr_cumsum_coef[pad_y:, :pad_x + 1]
    # for the bottom-left corner
    bl_cumsum_coef = np.cumsum(np.cumsum(
        norm_kernel[::1, ::-1], axis=0), axis=1)[::-1, ::1]
    coef[-pad_y - 1:, :pad_x + 1] = bl_cumsum_coef[:pad_y + 1, pad_x:]
    # for the bottom-right corner
    br_cumsum_coef = np.cumsum(np.cumsum(
        norm_kernel[::1, ::1], axis=0), axis=1)[::-1, ::-1]
    coef[-pad_y - 1:, -pad_x - 1:] = br_cumsum_coef[:pad_y + 1, :pad_x + 1]

    # Sides
    tb_slice = slice(pad_y + 1, -pad_y - 1)
    # for the left side
    fill_value = tl_cumsum_coef[-1, pad_x:]
    coef[tb_slice, :pad_x + 1] = fill_value.reshape([1, -1])
    # for the right side
    fill_value = br_cumsum_coef[0, :pad_x + 1]
    coef[tb_slice, -pad_x - 1:] = fill_value.reshape([1, -1])
    lr_slice = slice(pad_x + 1, -pad_x - 1)
    # for the top side
    fill_value = tl_cumsum_coef[pad_y:, -1]
    coef[:pad_y + 1, lr_slice] = fill_value.reshape([-1, 1])
    # for the right side
    fill_value = br_cumsum_coef[:pad_y + 1, 0]
    coef[-pad_y - 1:, lr_slice] = fill_value.reshape([-1, 1])

    # # code for validation of above
    # img = np.ones_like(input, dtype='float32')
    # import cv2
    # coef_cv2 = cv2.filter2D(img, -1, norm_kernel,
    #                         borderType=cv2.BORDER_CONSTANT)

    return coef

_lcn_make_coef = _subtractive_norm_make_coef


#
# math_tools.py ends here
