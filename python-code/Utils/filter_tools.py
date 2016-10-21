# filter_tools.py ---
#
# Filename: filter_tools.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Sat Feb 20 10:11:48 2016 (+0100)
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

import os

import cv2
import numpy as np
import six

from Utils.dump_tools import loadh5, loadpklz, saveh5
from Utils.math_tools import _lcn_make_coef, _subtractive_norm_make_coef

floatX = 'float32'


def apply_ghh_filter(img, W, b, num_in_sum, num_in_max, nonlinearity):
    """Applies GHH Filtering.

    This function can be used to compute the GHH layer **without** the nead of
    theano.

    Parameters
    ----------
    img: np.ndarray, 3D or 2D
        Image to be filtered. If preprocessing is required, it should be done
        beforehand. Last dimension is considered to be the channel dimension.

    W: np.ndarray, 5D
        Filter Weights. The dimension order should be (filters, input_channel,
        h, w, scale). Note that although we do not use scale space, we have the
        final dimension. This should later on be extended to use 3D filters.

    b: np.ndarray
        Filter biases.

    num_in_sum: int
        Number of elements inside a summation.

    num_in_max: int
        Number of elementes inside a max.

    nonlinearity: str
        Nonlinearity to apply to the output.

    """

    # Make sure we have the 5D filter
    if len(W.shape) != 5:
        raise ValueError('Shape of W is {}, '
                         'but I am expecting a 5D tensor!'
                         ''.format(W.shape))

    num_kernel = W.shape[0]
    num_inputC = W.shape[1]
    # num_outC = num_kernel / num_in_sum / num_in_max
    ky = W.shape[2]
    kx = W.shape[3]

    if W.shape[4] != 1:
        raise ValueError('Scale space filtering is not yet supported!')

    # Make sure the image and filter have the proper input channels
    if len(img.shape) == 2:
        # np.expand_dims(img.shape, axis=1)
        img = img[..., None]
    assert img.shape[2] == num_inputC
    h, w = img.shape[:2]

    # valid slice of filtering output
    valid_y = slice(ky // 2, h - ((ky - 1) // 2))
    valid_x = slice(kx // 2, w - ((kx - 1) // 2))

    # save response for each kernel as a list
    resp = [None] * num_kernel
    for idx_k in six.moves.xrange(num_kernel):
        # for each input channel, apply filtering
        cur_res = np.zeros(img.shape[:2], dtype=floatX)
        cur_res = cur_res[valid_y, valid_x]
        for idx_C in six.moves.xrange(num_inputC):
            # apply filter
            filter_res = cv2.filter2D(img[:, :, idx_C], -1,
                                      W[idx_k, idx_C, :, :, 0])
            # add to cur_res
            cur_res += filter_res[valid_y, valid_x]
        # add bias
        cur_res += b[idx_k]
        # save to list (in bc01 order to match theano ops)
        resp[idx_k] = np.expand_dims(cur_res, axis=0)

    resp = np.concatenate(resp)
    resp = resp.reshape([-1, num_in_sum, num_in_max,
                         resp.shape[-2], resp.shape[-1]])

    # max pool
    resp = np.max(resp, axis=2)

    # sum pool
    delta = np.cast[floatX](
        1.0 - 2.0 * (np.arange(num_in_sum, dtype=floatX) % 2.0)
    )
    resp = resp * delta.reshape([1, num_in_sum, 1, 1])
    resp = np.sum(resp, axis=1)

    # change resp's dimenstion order to 01c
    resp = resp.transpose([1, 2, 0])

    # Apply nonlinearity
    if nonlinearity == 'None':
        resp = resp
    elif nonlinearity == 'tanh':
        resp = np.tanh(resp)
    else:
        raise ValueError('Unsupported nonlinearity!')

    return resp


def apply_learned_filter_2_image_no_theano(image, save_dir,
                                           bNormalizeInput,
                                           sKpNonlinearity,
                                           verbose=True,
                                           network_weights=None):
    """Apply the learned keypoint detector to image.

    Parameters
    ----------
    image: np.ndarray
        Raw image, without any pre-processing.

    save_dir: str
        Full path to the model directory.

    bNormalizeInput: bool
        Normalization parameter used for training.

    sKpNonlinearity: str
        Nonlinearity applied to the keypoint scoremap.

    """

    if len(image.shape) > 2:
        raise NotImplementedError("This function is only meant for "
                                  "grayscale images!")

    num_in_sum = 4
    num_in_max = 4

    if network_weights is None:

        # Load model
        if os.path.exists(save_dir + 'model.h5'):
            model = loadh5(save_dir + 'model.h5')
            W = model['kp-c0']['kp-c0.W'].astype(floatX)
            b = model['kp-c0']['kp-c0.b'].astype(floatX)
        elif os.path.exists(save_dir + 'model.pklz'):
            model = loadpklz(save_dir + 'model.pklz')
            W = model[0].astype(floatX)
            b = model[1].astype(floatX)
        else:
            raise RuntimeError('Learned model does not exist!')

    else:
        model = loadh5(network_weights)
        W = model['kp-c0']['kp-c0.W'].astype(floatX)
        b = model['kp-c0']['kp-c0.b'].astype(floatX)

    # add a new axis at the end if the filter is 4D
    if W.ndim == 4:
        W = W[..., np.newaxis]

    if bNormalizeInput:
        print("Using trained mean and Std for normalization")
        mean_std_dict = loadh5(save_dir + 'mean_std.h5')
        image_prep = ((image - mean_std_dict['mean_x']) /
                      mean_std_dict['std_x'])
    else:
        print("Just dividing with 255!")
        image_prep = image / np.cast[floatX](255.0)

    # Do subtracive normalization if it is there
    bSubtractiveNormalization = False
    if 'kp-subnorm' in model.keys():
        norm_kern = model['kp-subnorm']['kp-subnorm.kernel']
        assert np.isclose(np.sum(norm_kern), 1.0)
        bSubtractiveNormalization = True

    # prepare image
    if bSubtractiveNormalization:
        print("Performing subtractive normalization!")
        # compute the coeffs to make adjust at borders
        coef = _subtractive_norm_make_coef(norm_kern, image.shape[:2])
        # run the filter to get means
        conv_mean = cv2.filter2D(image_prep, -1, norm_kern,
                                 borderType=cv2.BORDER_CONSTANT)
        # adjust means with precomputed coef
        adj_mean = conv_mean / coef
        # subtract the mean
        image_prep -= adj_mean

    # Do LCN if it is there
    bLCN = False
    if 'kp-lcn' in model.keys():
        norm_kern = model['kp-lcn']['kp-lcn.kernel']
        assert np.isclose(np.sum(norm_kern), 1.0)
        bLCN = True

    if bLCN:
        # compute the coeffs to make adjust at borders
        coef = _lcn_make_coef(norm_kern, image.shape[:2])
        # run the filter to get means and adjust
        conv_mean = cv2.filter2D(image_prep, -1, norm_kern,
                                 borderType=cv2.BORDER_CONSTANT)
        adj_mean = conv_mean / coef
        # subtract the mean
        sub_norm = image_prep - adj_mean
        # run the filter to get std and adjust
        conv_std = np.sqrt(
            cv2.filter2D(sub_norm**2.0, -1, norm_kern,
                         borderType=cv2.BORDER_CONSTANT)
        )
        adj_std = np.sqrt(conv_std / coef)
        # run the filter once more to get std mean and adjust it
        conv_mean_std = np.sqrt(
            cv2.filter2D(adj_std, -1, norm_kern,
                         borderType=cv2.BORDER_CONSTANT)
        )
        adj_mean_std = conv_mean_std / coef
        # now divide
        image_prep = sub_norm / np.maximum(adj_mean_std, adj_std)

    h, w = image_prep.shape

    # NEW implementation
    nonlinearity = sKpNonlinearity
    resp = apply_ghh_filter(
        image_prep, W, b,
        num_in_sum, num_in_max, nonlinearity)

    # The final return of this function should be a single channel image
    if len(resp.shape) == 3:
        assert resp.shape[2] == 1
        resp = np.squeeze(resp)

    return resp


#
# filter_tools.py ends here
