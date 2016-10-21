# sift_tools.py ---
#
# Filename: sift_tools.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Mon Feb 29 14:36:38 2016 (+0100)
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

from copy import deepcopy
from ctypes import c_int, c_void_p, cdll

import cv2
import numpy
import six

from Utils.kp_tools import (IDX_ANGLE, IDX_OCTAVE, IDX_RESPONSE, IDX_SIZE,
                            IDX_X, IDX_Y, update_affine)

SIFT_ORI_HIST_BINS = 36
SIFT_INT_DESCR_FCTR = 512.      # SIFT descriptor multiplication factor

import platform
if platform.system() == "Darwin":
    libSIFT = cdll.LoadLibrary("../c-code/libSIFT.dylib")
else:
    libSIFT = cdll.LoadLibrary('../c-code/libSIFT.so')

def recomputeOrientation(gray_img, kp_list, bSingleOrientation=False):
    """Wrapper for the C Library. """

    # prepare to send image to c library
    indata = numpy.uint8(gray_img.reshape(1, -1))
    h, w = gray_img.shape

    # prepare the keypoints
    x = numpy.zeros(len(kp_list), dtype=numpy.double)
    y = numpy.zeros(len(kp_list), dtype=numpy.double)
    octave = numpy.zeros(len(kp_list), dtype=numpy.int32)  # may be wrong
    response = numpy.zeros(len(kp_list), dtype=numpy.double)
    size = numpy.zeros(len(kp_list), dtype=numpy.double)
    angle = numpy.zeros(len(kp_list), dtype=numpy.double)
    numKey = len(kp_list)
    out_angle = numpy.zeros(len(kp_list), dtype=numpy.double)
    out_histogram = numpy.zeros(
        len(kp_list) * SIFT_ORI_HIST_BINS, dtype=numpy.double)

    for idxKp in six.moves.xrange(len(kp_list)):
        x[idxKp] = deepcopy(kp_list[idxKp][IDX_X])
        y[idxKp] = deepcopy(kp_list[idxKp][IDX_Y])
        octave[idxKp] = deepcopy(kp_list[idxKp][IDX_OCTAVE])
        response[idxKp] = deepcopy(kp_list[idxKp][IDX_RESPONSE])
        size[idxKp] = deepcopy(kp_list[idxKp][IDX_SIZE] * 2.0)
        angle[idxKp] = deepcopy(kp_list[idxKp][IDX_ANGLE])

    # Run the library
    libSIFT.recomputeOrientation(c_void_p(indata.ctypes.data),
                                 c_int(h), c_int(w),
                                 c_void_p(x.ctypes.data),
                                 c_void_p(y.ctypes.data),
                                 c_void_p(octave.ctypes.data),
                                 c_void_p(response.ctypes.data),
                                 c_void_p(size.ctypes.data),
                                 c_void_p(angle.ctypes.data),
                                 c_int(numKey),
                                 c_void_p(out_angle.ctypes.data),
                                 c_void_p(out_histogram.ctypes.data),
                                 c_int(bSingleOrientation))

    # Put into our keypoint object
    new_kp_list = deepcopy(kp_list)
    for kp, kp_old, new_angle in zip(new_kp_list, kp_list, out_angle):
        kp[IDX_ANGLE] = deepcopy(new_angle)
        # # copy other stuff....(due to opencv sucking at python)
        # kp[CLASS_ID] = kp_old.class_id
        # kp.octave = kp_old.octave
        # kp.pt = kp_old.pt
        # kp.response = kp_old.response
        # kp.size = kp_old.size

    # Update the affines
    affine_kp_list = [None] * len(new_kp_list)
    for idxKp in six.moves.xrange(len(new_kp_list)):
        affine_kp_list[idxKp] = update_affine(new_kp_list[idxKp])

    # Unroll SIFT orientation data
    out_histogram.shape = [len(kp_list), SIFT_ORI_HIST_BINS]
    # Change that to a list
    out_hist_list = [out_histogram[i, :].flatten()
                     for i in six.moves.xrange(len(kp_list))]

    return affine_kp_list, out_hist_list

#
# sift_tools.py ends here
