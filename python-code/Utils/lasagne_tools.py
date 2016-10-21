# lasagne_tools.py ---
#
# Filename: lasagne_tools.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Fri Feb 19 16:38:39 2016 (+0100)
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


from __future__ import print_function

import os

import h5py
import numpy as np
import theano
import theano.tensor as T
from lasagne.layers import (ConcatLayer, DimshuffleLayer, ExpressionLayer,
                            InputLayer, Layer, MergeLayer, get_output_shape,
                            set_all_param_values)
from lasagne.layers.pool import pool_output_length
from lasagne.utils import as_tuple
from theano.tensor.signal.pool import pool_2d

from Utils.custom_types import paramStruct, pathConfig
from Utils.dump_tools import loadpklz
from Utils.math_tools import (_lcn_make_coef, _subtractive_norm_make_coef,
                              softargmax)

floatX = theano.config.floatX


def createXYZMapLayer(scoremap_cut_layer,
                      fScaleList,
                      req_boundary,
                      name=None,
                      fCoMStrength=10.0,
                      eps=1e-8):
    """Creates the mapping layer for transforming the cut scores to xyz.

    Parameters
    ----------
    scoremap_cut_layer: lasange layer
        The input layer to the mapping. Typically
        myNet.layers[idxSiam]['kp-scoremap-cut']

    fScaleList: ndarray, flaot
        Array of scales that the layer operates with. Given by the
        configuration.

    req_boundary: int
        The number of pixels at each boundary that was cut from the original
        input patch size to get the valid scormap.

    name: str
        Name of the mapping layer.

    fCoMStrength: float
        Strength of the soft CoM style argmax. Negative value means hard
        argmax.

    eps: float
        The epsilon that is added to the denominator of CoM to prevent
        numerical problemsu

    """

    fCoMStrength = np.cast[floatX](fCoMStrength)
    eps = np.cast[floatX](eps)
    scoremap_shape = get_output_shape(scoremap_cut_layer)
    num_scale = len(fScaleList)
    # scale_space_min changes according to scoremap_shape
    num_scale_after = scoremap_shape[4]
    new_min_idx = (num_scale - num_scale_after) / 2
    scale_space_min = fScaleList[new_min_idx]

    if num_scale >= 2:
        scale_space_step = (fScaleList[num_scale - 1] /
                            fScaleList[num_scale // 2])**(
                                1.0 / float(num_scale // 2))
    else:
        scale_space_step = np.cast[floatX](1)

    # mapping from score map to x,y,z
    def map2xyz(out, fCoMStrength, scoremap_shape, eps, scale_space_min,
                scale_space_step, req_boundary):

        # In case of soft argmax
        if fCoMStrength > 0:
            # x = softargmax(T.sum(out, axis=[1, 2, 4]), axis=1,
            #                softargmax_strength=fCoMStrength)
            # y = softargmax(T.sum(out, axis=[1, 3, 4]), axis=1,
            #                softargmax_strength=fCoMStrength)
            # z = softargmax(T.sum(out, axis=[1, 2, 3]), axis=1,
            #                softargmax_strength=fCoMStrength)
            od = len(scoremap_shape)
            # CoM to get the coordinates
            pos_array_x = T.arange(scoremap_shape[3], dtype=floatX)
            pos_array_y = T.arange(scoremap_shape[2], dtype=floatX)
            pos_array_z = T.arange(scoremap_shape[4], dtype=floatX)

            # max_out = T.max(T.maximum(out, 0),
            #                 axis=list(range(1, od)), keepdims=True)
            max_out = T.max(out, axis=list(range(1, od)), keepdims=True)
            o = T.exp(fCoMStrength * (out - max_out))
            # o = T.exp(fCoMStrength * T.maximum(out, 0) + np.cast[floatX](
            #     1.0)) - np.cast[floatX](1.0)
            x = T.sum(
                o * pos_array_x.dimshuffle(['x', 'x', 'x', 0, 'x']),
                axis=list(range(1, od))
            ) / (T.sum(o, axis=list(range(1, od))))
            y = T.sum(
                o * pos_array_y.dimshuffle(['x', 'x', 0, 'x', 'x']),
                axis=list(range(1, od))
            ) / (T.sum(o, axis=list(range(1, od))))
            z = T.sum(
                o * pos_array_z.dimshuffle(['x', 'x', 'x', 'x', 0]),
                axis=list(range(1, od))
            ) / (T.sum(o, axis=list(range(1, od))))

            # --------------
            # Turn x, and y into range -1 to 1, where the patch size is
            # mapped to -1 and 1
            orig_patch_width = (
                scoremap_shape[3] + np.cast[floatX](req_boundary * 2.0))
            orig_patch_height = (
                scoremap_shape[2] + np.cast[floatX](req_boundary * 2.0))

            x = ((x + np.cast[floatX](req_boundary)) / np.cast[floatX](
                (orig_patch_width - 1.0) * 0.5) -
                np.cast[floatX](1.0)).dimshuffle([0, 'x'])
            y = ((y + np.cast[floatX](req_boundary)) / np.cast[floatX](
                (orig_patch_height - 1.0) * 0.5) -
                np.cast[floatX](1.0)).dimshuffle([0, 'x'])

            # --------------
            #  Turn z into log2 scale, where z==0 is the center
            # scale. e.g. z == -1 would mean that it is actuall scale
            # of 0.5 x center scale

            # z = np.cast[floatX](scale_space_min) * (
            #     np.cast[floatX](scale_space_step)**z)
            z = np.cast[floatX](np.log2(scale_space_min)) + \
                np.cast[floatX](np.log2(scale_space_step)) * z
            z = z.dimshuffle([0, 'x'])

        # In case of hard argmax
        else:
            raise RuntimeError('The hard argmax does not have derivatives!')
            # x = T.cast(
            #     T.argmax(T.sum(out, axis=[1, 2, 4]), axis=1), dtype=floatX)
            # y = T.cast(
            #     T.argmax(T.sum(out, axis=[1, 3, 4]), axis=1), dtype=floatX)
            # z = T.cast(
            #     T.argmax(T.sum(out, axis=[1, 2, 3]), axis=1), dtype=floatX)
            x = softargmax(T.sum(out, axis=[1, 2, 4]), axis=1,
                           softargmax_strength=-fCoMStrength)
            y = softargmax(T.sum(out, axis=[1, 3, 4]), axis=1,
                           softargmax_strength=-fCoMStrength)
            z = softargmax(T.sum(out, axis=[1, 2, 3]), axis=1,
                           softargmax_strength=-fCoMStrength)

            # --------------
            # Turn x, and y into range -1 to 1, where the patch size is
            # mapped to -1 and 1
            orig_patch_width = (
                scoremap_shape[3] + np.cast[floatX](req_boundary * 2.0))
            orig_patch_height = (
                scoremap_shape[2] + np.cast[floatX](req_boundary * 2.0))

            x = ((x + np.cast[floatX](req_boundary)) / np.cast[floatX](
                (orig_patch_width - 1.0) * 0.5) -
                np.cast[floatX](1.0)).dimshuffle([0, 'x'])
            y = ((y + np.cast[floatX](req_boundary)) / np.cast[floatX](
                (orig_patch_height - 1.0) * 0.5) -
                np.cast[floatX](1.0)).dimshuffle([0, 'x'])

            # --------------
            #  Turn z into log2 scale, where z==0 is the center
            # scale. e.g. z == -1 would mean that it is actuall scale
            # of 0.5 x center scale

            # z = np.cast[floatX](scale_space_min) * (
            #     np.cast[floatX](scale_space_step)**z)
            z = np.cast[floatX](np.log2(scale_space_min)) + \
                np.cast[floatX](np.log2(scale_space_step)) * z
            z = z.dimshuffle([0, 'x'])

        return T.concatenate([x, y, z], axis=1)

    def mapfunc(out):
        return map2xyz(out, fCoMStrength, scoremap_shape, eps, scale_space_min,
                       scale_space_step, req_boundary)

    return ExpressionLayer(
        scoremap_cut_layer,
        mapfunc,
        output_shape='auto',
        name=name)


# create a sub-network of layer that crops the input resizes to given
# patch size
def createXYZTCropLayer(input_layer_4d,
                        xyz_layer,
                        theta_layer,
                        max_scale,
                        out_width,
                        name=None):

    input_layer_shape = get_output_shape(input_layer_4d)
    batch_size = input_layer_shape[0]

    new_width = out_width
    new_height = out_width

    # ratio to reduce to patch size from original
    reduc_ratio = (np.cast[floatX](out_width) /
                   np.cast[floatX](input_layer_shape[3]))

    # merge xyz and t layers together to form xyzt
    xyzt_layer = ConcatLayer([xyz_layer, theta_layer])

    # create a param layer from xyz layer
    def xyzt_2_param(xyzt):
        # get individual xyz
        dx = xyzt[:, 0]  # x and y are already between -1 and 1
        dy = xyzt[:, 1]  # x and y are already between -1 and 1
        z = xyzt[:, 2]
        t = xyzt[:, 3]
        # compute the resize from the largest scale image
        dr = (np.cast[floatX](reduc_ratio) * np.cast[floatX]
              (2.0)**z / np.cast[floatX](max_scale))

        # dimshuffle before concatenate
        params = [dr * T.cos(t), -dr * T.sin(t), dx, dr * T.sin(t),
                  dr * T.cos(t), dy]
        params = [_p.flatten().dimshuffle(0, 'x') for _p in params]

        # concatenate to have (1 0 0 0 1 0) when identity transform
        return T.concatenate(params, axis=1)

    param_layer = ExpressionLayer(xyzt_layer,
                                  xyzt_2_param,
                                  output_shape=(batch_size, 6))

    resize_layer = TransformerLayer(input_layer_4d,
                                    param_layer,
                                    new_height,
                                    new_width,
                                    name=name)

    return resize_layer


# create a sub-network of layer that crops the input resizes to given
# patch size
def createXYZCropLayer(input_layer_4d,
                       xyz_layer,
                       max_scale,
                       out_width,
                       name=None):

    input_layer_shape = get_output_shape(input_layer_4d)
    batch_size = input_layer_shape[0]

    new_width = out_width
    new_height = out_width

    # ratio to reduce to patch size from original
    reduc_ratio = (np.cast[floatX](out_width) /
                   np.cast[floatX](input_layer_shape[3]))

    # create a param layer from xyz layer
    def xyz_2_param(xyz):
        # get individual xyz
        dx = xyz[:, 0]  # x and y are already between -1 and 1
        dy = xyz[:, 1]  # x and y are already between -1 and 1
        z = xyz[:, 2]
        # compute the resize from the largest scale image
        dr = (np.cast[floatX](reduc_ratio) * np.cast[floatX]
              (2.0)**z / np.cast[floatX](max_scale))

        zl = T.zeros_like(dx)

        # dimshuffle before concatenate
        params = [dr, zl, dx, zl, dr, dy]
        params = [_p.flatten().dimshuffle(0, 'x') for _p in params]

        # concatenate to have (1 0 0 0 1 0) when identity transform
        return T.concatenate(params, axis=1)

    param_layer = ExpressionLayer(xyz_layer,
                                  xyz_2_param,
                                  output_shape=(batch_size, 6))

    resize_layer = TransformerLayer(input_layer_4d,
                                    param_layer,
                                    new_height,
                                    new_width,
                                    name=name)

    return resize_layer


# create a sub-network of layer that resizes the input (using full image)
def createResizeLayer(input_layer_4d, resize_ratio, name=None):

    input_layer_shape = get_output_shape(input_layer_4d)
    batch_size = input_layer_shape[0]

    new_height = int(np.round(resize_ratio * input_layer_shape[2]))
    new_width = int(np.round(resize_ratio * input_layer_shape[3]))

    # ds = 1.0 / resize_ratio
    # rf = resize_ratio * ds

    rf = 1.0  # since we basically down sample with new_height, new_width
    rescaleA = np.tile(
        np.asarray([[rf, 0], [0, rf], [0, 0]],
                   dtype=floatX).T.reshape([1, 2, 3]),
        [batch_size, 1, 1]).reshape([-1, 6])
    param_layer = InputLayer(
        (batch_size, 6),
        input_var=theano.shared(rescaleA))
    resize_layer = TransformerLayer(input_layer_4d,
                                    param_layer,
                                    new_height,
                                    new_width,
                                    name=name)

    return resize_layer


# create a sub-network of layers that performs scale space conversion
def createConvtScaleSpaceLayer(input_layer_4d, resize_ratio_list, name=None):

    input_layer_shape = get_output_shape(input_layer_4d)
    batch_size = input_layer_shape[0]
    # patch_size = input_layer_shape[2]

    # orig_c = (float(input_layer_shape[2]) - 1.0) * 0.5

    scale_space_layer_list = []

    for resize_ratio in resize_ratio_list:

        rf = resize_ratio
        c = 0
        # the implementation already works on 0-center coordinate system

        rescaleA = np.tile(
            np.asarray([[rf, 0], [0, rf], [c, c]],
                       dtype=floatX).T.reshape([1, 2, 3]),
            [batch_size, 1, 1]).reshape([-1, 6])
        param_layer = InputLayer(
            (batch_size, 6),
            input_var=theano.shared(rescaleA))
        resize_layer = TransformerLayer(input_layer_4d, param_layer,
                                        input_layer_shape[2],
                                        input_layer_shape[3])
        scale_space_layer_list += [
            # ReshapeLayer(resize_layer,
            #              tuple([v for v in input_layer_shape] + [1]))
            DimshuffleLayer(resize_layer, (0, 1, 2, 3, 'x'))
        ]

    scale_space_layer = ConcatLayer(scale_space_layer_list, axis=4, name=name)

    return scale_space_layer


def loadNetwork(dict_of_layers, dump_file_full_name_no_ext, prefix=''):
    ''' Loads the network parameter from saved file.

    dict_of_layers: dictionay containing lasagne layers type
    dump_file_full...: full path of the dump file, WITHOUT extensions
    prefix: only load layers with name starting with given prefix (Optional)

    '''

    # Check h5 file, load that if it exists, else use the old one
    if os.path.exists(dump_file_full_name_no_ext + '.h5'):
        with h5py.File(dump_file_full_name_no_ext + '.h5', 'r') as h5file:
            # Find the layer parameters
            for _key in dict_of_layers.keys():
                # Only load when startswith given prefix
                if not _key.startswith(prefix):
                    continue

                if _key not in h5file.keys():
                    # If there are actual parameters to load
                    if len(dict_of_layers[_key].get_params()) > 0:
                        raise ValueError(
                            'The saved file does not have parameters stored '
                            'for layer {}!'.format(_key))
                    # If it is just an expression layer, then probably it's
                    # safe to ignore as it indicates an update in
                    # implementation. However, we print something out, just in
                    # case.
                    else:
                        print("WARNING: We don't have the pure-expression "
                              "layer {} saved in the model file! This might "
                              "be okay, just due to implementation changes, "
                              "but you should still be aware.".format(_key))

                else:
                    # special case for subtractive normalization layer
                    if isinstance(dict_of_layers[_key],
                                  SubtractiveNormalization2DLayer):
                        if _key + '.kernel' in h5file[_key].keys():
                            dict_of_layers[_key].kernel = np.asarray(
                                h5file[_key][_key + '.kernel'].value
                            )
                        else:
                            print("WARNING: We don't have the kernel value "
                                  "saved for the subtractive normalization "
                                  "layer. This might be dues to implementation"
                                  " changes.")
                    # special case for local contrast normalization layer
                    if isinstance(dict_of_layers[_key],
                                  LocalContrastNormalization2DLayer):
                        if _key + '.kernel' in h5file[_key].keys():
                            dict_of_layers[_key].kernel = np.asarray(
                                h5file[_key][_key + '.kernel'].value
                            )
                        else:
                            print("WARNING: We don't have the kernel value "
                                  "saved for the local contrast normalization "
                                  "layer. This might be dues to implementation"
                                  " changes.")
                    # For each parameter in the layer
                    cur_grp = h5file[_key]
                    for param in dict_of_layers[_key].get_params():
                        if param.name not in cur_grp.keys():
                            raise ValueError(
                                'The saved file does not have parameter {}'
                                'for layer {}!'.format(param.name,
                                                       dict_of_layers[_key]))
                        else:
                            # special treatment fot the weight matrix
                            val = np.asarray(cur_grp[param.name], dtype=floatX)
                            if param.name.endswith('.W'):
                                # If param is 5D tensor and val is 4D but the
                                # last dimension in param is only 1 then these
                                # two are compatible. we should reshape the
                                # read value to 5D and set value.
                                if (param.get_value().ndim == 5 and
                                        val.ndim == 4 and
                                        param.get_value().shape[-1] == 1):
                                    val = val[..., np.newaxis]
                                # If param is 4D tensor and val is 5D but the
                                # last dimension is only of size 1, these two
                                # are compatible. We should reshape the read
                                # value to 4D and set value.
                                if (param.get_value().ndim == 4 and
                                        val.ndim == 5 and
                                        val.shape[-1] == 1):
                                    val = val[..., 0]
                            param.set_value(np.asarray(
                                cur_grp[param.name], dtype=floatX))

    # The old loading
    elif os.path.exists(dump_file_full_name_no_ext + '.pklz'):
        values = loadpklz(dump_file_full_name_no_ext + '.pklz')
        set_all_param_values(dict_of_layers['output'], values)

    else:
        raise RuntimeError('Dump file for {} does not exist!'.format(
            dump_file_full_name_no_ext))


def loadNetworkWithConfig(dict_of_layers, config_file_name, prefix=''):
    ''' Loads the network parameter from saved file given config file

    dict_of_layers: dictionay containing lasagne layers type
    config_file_name: config file name
    prefix: only load layers with name starting with given prefix (Optional)

    '''

    save_dir_pretrain = get_save_dir_from_config(config_file_name)

    # Load pretrained network weights
    loadNetwork(dict_of_layers, save_dir_pretrain + 'model', prefix=prefix)


def load_layer_weights(layer_name, saved_model_name):
    """Loads the weights of a designated layer within the network model (hdf5)

    """

    with h5py.File(saved_model_name, "r") as h5file:
        if layer_name in h5file.keys():
            W = h5file[layer_name][layer_name + ".W"].value
            b = h5file[layer_name][layer_name + ".b"].value
        else:
            raise RuntimeError(
                "No such layer {}".format(layer_name)
            )

    return W, b


def get_save_dir_from_config(config_file_name):
    """ Returns the configuration's save_dir
    """

    param_pretrain = paramStruct()
    param_pretrain.loadParam(config_file_name,
                             verbose=False)
    pathconf_pretrain = pathConfig()
    # check the results directory to see if we are working on this
    # configuration already
    pathconf_pretrain.setupTrain(param_pretrain, 0)
    save_dir_pretrain = pathconf_pretrain.result

    return save_dir_pretrain


def saveNetwork(dict_of_layers, dump_file_full_name_no_ext):
    ''' Saves the network parameter to a file '''

    with h5py.File(dump_file_full_name_no_ext + '.h5', 'w') as h5file:
        for _key in dict_of_layers.keys():
            # create group for the current
            h5file.create_group(_key)
            # special case for subtractive normalization layer
            if isinstance(dict_of_layers[_key],
                          SubtractiveNormalization2DLayer):
                h5file[_key][_key + '.kernel'] = dict_of_layers[_key].kernel
            # special case for local contrast normalization layer
            if isinstance(dict_of_layers[_key],
                          LocalContrastNormalization2DLayer):
                h5file[_key][_key + '.kernel'] = dict_of_layers[_key].kernel
            # for each parameter of the layer
            for param in dict_of_layers[_key].get_params():
                if param.name is None:
                    raise ValueError(
                        'Layer {} has a parameter with no name!'.format(
                            dict_of_layers[_key]))
                else:
                    h5file[_key][param.name] = param.get_value()


class TransformerLayer(MergeLayer):
    """
    Spatial transformer layer (Custom version to avoid ds problem)

    The layer applies an affine transformation on the input. The affine
    transformation is parameterized with six learned parameters [1]_.
    The output is interpolated with a bilinear transformation.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape. The
        output of this layer should be a 4D tensor, with shape
        ``(batch_size, num_input_channels, input_rows, input_columns)``.
        or, it can also be a 3D tensor of shape
        ``(num_input_channels, input_rows, input_columns)``.

    localization_network : a :class:`Layer` instance
        The network that calculates the parameters of the affine
        transformation. See the example for how to initialize to the identity
        transform.

    downsample_factor is replaced with desired output shape

    downsample_factor : float or iterable of float
        A float or a 2-element tuple specifying the downsample factor for the
        output image (in both spatial dimensions). A value of 1 will keep the
        original size of the input. Values larger than 1 will downsample the
        input. Values below 1 will upsample the input.

    References
    ----------
    .. [1]  Spatial Transformer Networks
            Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
            Submitted on 5 Jun 2015

    Examples
    --------
    Here we set up the layer to initially do the identity transform, similarly
    to [1]_. Note that you will want to use a localization with linear output.
    If the output from the localization networks is [t1, t2, t3, t4, t5, t6]
    then t1 and t5 determines zoom, t2 and t4 determines skewness, and t3 and
    t6 move the center position.

    >>> import numpy as np
    >>> import lasagne
    >>> b = np.zeros((2, 3), dtype='float32')
    >>> b[0, 0] = 1
    >>> b[1, 1] = 1
    >>> b = b.flatten()  # identity transform
    >>> W = lasagne.init.Constant(0.0)
    >>> l_in = lasagne.layers.InputLayer((None, 3, 28, 28))
    >>> l_loc = lasagne.layers.DenseLayer(l_in, num_units=6, W=W, b=b,
    ... nonlinearity=None)
    >>> l_trans = lasagne.layers.TransformerLayer(l_in, l_loc)
    """

    def __init__(self, incoming, localization_network, out_height, out_width,
                 **kwargs):
        super(TransformerLayer, self).__init__(
            [incoming, localization_network], **kwargs)
        self.out_height = out_height
        self.out_width = out_width

        input_shp, loc_shp = self.input_shapes

        if loc_shp[-1] != 6 or (len(loc_shp) != 2 and len(loc_shp) != 3):
            raise ValueError("The localization network must have "
                             "output shape: (batch_size, 6)")
        if len(input_shp) != 4:
            raise ValueError("The input network must have a 4-dimensional "
                             "output shape: (batch_size, num_input_channels, "
                             "input_rows, input_columns).")

    def get_output_shape_for(self, input_shapes):
        shape = input_shapes[0]
        return (shape[:2] + (self.out_height, self.out_width))

    def get_output_for(self, inputs, **kwargs):
        # see eq. (1) and sec 3.1 in [1]
        input, theta = inputs
        return _transform(theta, input, self.out_height, self.out_width)


def _transform(theta, input, out_height, out_width):
    num_b = theta.shape[0]

    _, num_channels, height, width = input.shape
    num_batch = num_b
    theta = T.reshape(theta, (-1, 2, 3))

    # grid of (x_t, y_t, 1), eq (1) in ref [1]
    # out_height = T.cast(height / downsample_factor[0], 'int64')
    # out_width = T.cast(width / downsample_factor[1], 'int64')
    grid = _meshgrid(out_height, out_width)

    # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
    T_g = T.dot(theta, grid)
    x_s = T_g[:, 0]
    y_s = T_g[:, 1]
    x_s_flat = x_s.flatten()
    y_s_flat = y_s.flatten()

    # dimshuffle input to  (bs, height, width, channels)
    input_dim = input.dimshuffle(0, 2, 3, 1)

    # index and interpolate
    input_transformed = _interpolate(input_dim, x_s_flat, y_s_flat, out_height,
                                     out_width, num_b)

    output = T.reshape(input_transformed,
                       (num_batch, out_height, out_width, num_channels))
    output = output.dimshuffle(0, 3, 1, 2)  # dimshuffle to conv format
    return output


def _interpolate(im, x, y, out_height, out_width, num_b):
    _, height, width, channels = im.shape
    # *_f are floats
    height_f = T.cast(height, theano.config.floatX)
    width_f = T.cast(width, theano.config.floatX)

    # clip coordinates to [-1, 1]
    x = T.clip(x, -1, 1)
    y = T.clip(y, -1, 1)

    # scale coordinates from [-1, 1] to [0, width/height - 1]
    x = (x + 1) / 2 * (width_f - 1)
    y = (y + 1) / 2 * (height_f - 1)

    # obtain indices of the 2x2 pixel neighborhood surrounding the coordinates;
    # we need those in floatX for interpolation and in int64 for indexing. for
    # indexing, we need to take care they do not extend past the image.
    x0_f = T.floor(x)
    y0_f = T.floor(y)
    x1_f = x0_f + 1
    y1_f = y0_f + 1

    # KMYI: we cast only at the end to maximize GPU usage
    x0 = T.floor(x0_f)
    y0 = T.floor(y0_f)
    x1 = T.floor(T.minimum(x1_f, width_f - 1))
    y1 = T.floor(T.minimum(y1_f, height_f - 1))

    dim2 = width_f
    dim1 = width_f * height_f
    base = T.repeat(
        T.arange(num_b,
                 dtype=theano.config.floatX) * dim1,
        out_height * out_width)
    base_y0 = base + y0 * dim2
    base_y1 = base + y1 * dim2
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1

    # use indices to lookup pixels for all samples
    im_flat = im.reshape((-1, channels))
    Ia = im_flat[T.cast(idx_a, 'int64')]
    Ib = im_flat[T.cast(idx_b, 'int64')]
    Ic = im_flat[T.cast(idx_c, 'int64')]
    Id = im_flat[T.cast(idx_d, 'int64')]

    # calculate interpolated values
    wa = ((x1_f - x) * (y1_f - y)).dimshuffle(0, 'x')
    wb = ((x1_f - x) * (y - y0_f)).dimshuffle(0, 'x')
    wc = ((x - x0_f) * (y1_f - y)).dimshuffle(0, 'x')
    wd = ((x - x0_f) * (y - y0_f)).dimshuffle(0, 'x')
    output = T.sum([wa * Ia, wb * Ib, wc * Ic, wd * Id], axis=0)
    return output


def _linspace(start, stop, num):
    # Theano linspace. Behaves similar to np.linspace
    start = T.cast(start, theano.config.floatX)
    stop = T.cast(stop, theano.config.floatX)
    num = T.cast(num, theano.config.floatX)
    step = (stop - start) / (num - 1)
    return T.arange(num, dtype=theano.config.floatX) * step + start


def _meshgrid(height, width):
    # This function is the grid generator from eq. (1) in reference [1].
    # It is equivalent to the following numpy code:
    #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
    #                         np.linspace(-1, 1, height))
    #  ones = np.ones(np.prod(x_t.shape))
    #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
    # It is implemented in Theano instead to support symbolic grid sizes.
    # Note: If the image size is known at layer construction time, we could
    # compute the meshgrid offline in numpy instead of doing it dynamically
    # in Theano. However, it hardly affected performance when we tried.
    x_t = T.dot(
        T.ones((height, 1)), _linspace(-1.0, 1.0, width).dimshuffle('x', 0))
    y_t = T.dot(
        _linspace(-1.0, 1.0, height).dimshuffle(0, 'x'), T.ones((1, width)))

    x_t_flat = x_t.reshape((1, -1))
    y_t_flat = y_t.reshape((1, -1))
    ones = T.ones_like(x_t_flat)
    grid = T.concatenate([x_t_flat, y_t_flat, ones], axis=0)
    return grid


class LPPool2DLayer(Layer):
    """
    2D LP Pooling layer

    Performs 2D LP pooling over the two trailing axes of a 4D input tensor.
    Implemented from looking at the torch source code, available at
    https://github.com/torch/nn/blob/master/SpatialLPPooling.lua, and modifying
    the Pool2DDNNLayer

    Parameters
    ----------
    incoming : a :class:`Layer` instance or tuple
        The layer feeding into this layer, or the expected input shape.

    pnorm : integer
        The degree of the LP-norm part.

    pool_size : integer or iterable
        The length of the pooling region in each dimension.  If an integer, it
        is promoted to a square pooling region. If an iterable, it should have
        two elements.

    stride : integer, iterable or ``None``
        The strides between sucessive pooling regions in each dimension.
        If ``None`` then ``stride = pool_size``.

    pad : integer or iterable
        Number of elements to be added on each side of the input
        in each dimension. Each value must be less than
        the corresponding stride.

    ignore_border : bool
        If ``True``, partial pooling regions will be ignored.
        Must be ``True`` if ``pad != (0, 0)``.

    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.
    """

    def __init__(self, incoming, pnorm, pool_size, stride=None, pad=(0, 0),
                 ignore_border=True, mode='average_inc_pad', **kwargs):
        super(LPPool2DLayer, self).__init__(incoming, **kwargs)
        if len(self.input_shape) != 4:
            raise ValueError("Tried to create a 2D pooling layer with "
                             "input shape %r. Expected 4 input dimensions "
                             "(batchsize, channels, 2 spatial dimensions)."
                             % (self.input_shape,))
        self.pnorm = T.cast(pnorm, dtype=theano.config.floatX)
        self.pool_size = as_tuple(pool_size, 2)
        if stride is None:
            self.stride = self.pool_size
        else:
            self.stride = as_tuple(stride, 2)
        self.pad = as_tuple(pad, 2)
        self.ignore_border = ignore_border
        # The ignore_border argument is for compatibility with MaxPool2DLayer.
        # ignore_border=False is not supported. Borders are always ignored.
        # if not ignore_border:
        #     raise NotImplementedError("LPPool2DLayer is based on "
        #                               "Pool2DDNNLayer that does not support "
        #                               "ignore_border=False.")
        if mode != 'average_inc_pad' and mode != 'average_exc_pad':
            raise ValueError("LPPool2DLayer requires mode=average_inc_pad"
                             " or mode=average_exc_pad, but received "
                             "mode={} instead.".format(mode))
        self.mode = mode

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)  # copy / convert to mutable list

        output_shape[2] = pool_output_length(input_shape[2],
                                             pool_size=self.pool_size[0],
                                             stride=self.stride[0],
                                             pad=self.pad[0],
                                             ignore_border=self.ignore_border,
                                             )

        output_shape[3] = pool_output_length(input_shape[3],
                                             pool_size=self.pool_size[1],
                                             stride=self.stride[1],
                                             pad=self.pad[1],
                                             ignore_border=self.ignore_border,
                                             )

        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        # Power
        input_powered = input**self.pnorm
        # Average pool
        avg_pooled = pool_2d(input_powered,
                             ds=self.pool_size,
                             st=self.stride,
                             ignore_border=self.ignore_border,
                             padding=self.pad,
                             mode=self.mode,
                             )
        # Scale with pooling kernel since we want the sum, not average
        scaler = T.cast(np.prod(self.pool_size), dtype=theano.config.floatX)
        scaled = avg_pooled * scaler
        # De-power
        depowered = scaled**(T.cast(1.0, dtype=theano.config.floatX) /
                             self.pnorm)

        return depowered


class SubtractiveNormalization2DLayer(Layer):
    """
    Subtractive normalization that works in 2D.

    WRITEME

    Parameters
    ----------
    incoming : a :class:`Layer` instance
        The layer feeding into this layer.

    kernel : numpy ndarray, optional
        The kernel to be used to compute the so-called ``mean''. If not given,
        the layer take the mean of 9 x 9 region. This is in-line with the torch
        implementation.

    convolution : callable, optional
        The convolution implementation to use. Usually it should be fine to
        leave this at the default value.
    """

    def __init__(self, incoming, kernel=None, convolution=T.nnet.conv2d,
                 **kwargs):
        super(SubtractiveNormalization2DLayer, self).__init__(incoming,
                                                              **kwargs)
        if kernel is None:
            kernel = np.ones((9, 9), dtype=theano.config.floatX)
        if kernel.ndim == 1:
            raise NotImplementedError(
                "Seperable filters are not yet supported!")
        elif kernel.ndim != 2:
            raise ValueError(
                "Kernel dimension should either be 1 or 2!")
        if any(s % 2 == 0 for s in kernel.shape):
            raise ValueError(
                "Only odd kernel sizes are supported!")
        self.kernel = np.cast[theano.config.floatX](kernel)
        self.convolution = convolution

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        # Get input shape
        input_shape = self.input_shape
        orig_input_shape = input_shape
        if any(s is None for s in input_shape):
            input_shape = input.shape
        # Make sure that we at least know the input_shape in the dimensions we
        # convolve.
        if any(s is None for s in input_shape[2:]):
            raise RuntimeError(
                "Input shape for SubtractiveNormalization2DLayer is not known!"
                " At least the convolving dimensions should be known for "
                "this layer to work properly.")
        # Make sure that input shape is in bc01 or bc01 + 1 extra dimension
        if len(input_shape) != 4:
            if input_shape[4] != 1:
                raise RuntimeError(
                    "Input shape should be in bc01 format or bc01 with one "
                    "extra dimension that is of length one.")
            else:
                # safely drop the last dim
                input_shape = input_shape[:4]
                input = T.reshape(input, input_shape)

        # ----------
        # Normalize kernel.
        # Note that unlike Torch, we don't divide the kernel here. We divide
        # when it is fed to the convolution, since we use it to generate the
        # coefficient map.
        kernel = self.kernel
        norm_kernel = (kernel / np.sum(kernel))

        # ----------
        # Compute the adjustment coef.
        # This allows our mean computation to compensate for the border area,
        # where you have less terms adding up. Torch used convolution with a
        # ``one'' image, but since we do not want the library to depend on
        # other libraries with convolutions, we do it manually here.
        coef = _subtractive_norm_make_coef(norm_kernel, input_shape[2:4])

        # ----------
        # Extract convolutional mean
        # Make filter a c01 filter by repeating. Note that we normalized above
        # with the number of repetitions we are going to do.
        norm_kernel = np.tile(norm_kernel, [input_shape[1], 1, 1])
        # Re-normlize the kernel so that the sum is one.
        norm_kernel /= np.sum(norm_kernel)
        # add another axis in from to make oc01 filter, where o is the number
        # of output dimensions (in our case, 1!)
        norm_kernel = norm_kernel[np.newaxis, ...]
        # To pad with zeros, half the size of the kernel (only for 01 dims)
        border_mode = tuple(s // 2 for s in norm_kernel.shape[2:])
        # Convolve the mean filter. Results in shape of (batch_size,
        # 1, input_shape[2], input_shape[3]).
        conv_mean = self.convolution(input=input,
                                     filters=T.as_tensor(norm_kernel),
                                     filter_shape=norm_kernel.shape,
                                     border_mode=border_mode,
                                     filter_flip=False,
                                     )
        # ----------
        # Adjust convolutional mean with precomputed coef
        # This is to prevent border values being too small.
        adj_mean = conv_mean / T.as_tensor(coef).dimshuffle(['x', 'x', 0, 1])
        # Make second dimension broadcastable as we are going to
        # subtract for all channels.
        adj_mean = T.addbroadcast(adj_mean, 1)

        # ----------
        # Subtract mean and reshape back just in case
        sub_normalized = input - adj_mean
        sub_normalized = T.reshape(sub_normalized, orig_input_shape)

        # # line for debugging
        # test = theano.function(inputs=[input], outputs=[sub_normalized])

        return sub_normalized


class LocalContrastNormalization2DLayer(Layer):
    """
    Local contrast normalization that works in 2D.

    WRITEME

    Parameters
    ----------
    incoming : a :class:`Layer` instance
        The layer feeding into this layer.

    kernel : numpy ndarray, optional
        The kernel to be used to compute the so-called ``mean''. If not given,
        the layer take the mean of 9 x 9 region. This is in-line with the torch
        implementation.

    eps : float
        Small number to be added to the divisor when dividing to avoid
        numerical problems.

    convolution : callable, optional
        The convolution implementation to use. Usually it should be fine to
        leave this at the default value.
    """

    def __init__(self, incoming, kernel=None, eps=1e-8,
                 convolution=T.nnet.conv2d, **kwargs):
        super(LocalContrastNormalization2DLayer, self).__init__(incoming,
                                                                **kwargs)
        if kernel is None:
            kernel = np.ones((9, 9), dtype=theano.config.floatX)
        if kernel.ndim == 1:
            raise NotImplementedError(
                "Seperable filters are not yet supported!")
        elif kernel.ndim != 2:
            raise ValueError(
                "Kernel dimension should either be 1 or 2!")
        if any(s % 2 == 0 for s in kernel.shape):
            raise ValueError(
                "Only odd kernel sizes are supported!")
        self.kernel = np.cast[theano.config.floatX](kernel)
        self.eps = eps
        self.convolution = convolution

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        # Get input shape
        input_shape = self.input_shape
        orig_input_shape = input_shape
        if any(s is None for s in input_shape):
            input_shape = input.shape
        # Make sure that we at least know the input_shape in the dimensions we
        # convolve.
        if any(s is None for s in input_shape[2:]):
            raise RuntimeError(
                "Input shape for SubtractiveNormalization2DLayer is not known!"
                " At least the convolving dimensions should be known for "
                "this layer to work properly.")
        # Make sure that input shape is in bc01 or bc01 + 1 extra dimension
        if len(input_shape) != 4:
            if input_shape[4] != 1:
                raise RuntimeError(
                    "Input shape should be in bc01 format or bc01 with one "
                    "extra dimension that is of length one.")
            else:
                # safely drop the last dim
                input_shape = input_shape[:4]
                input = T.reshape(input, input_shape)

        # ----------
        # Normalize kernel.
        # Note that unlike Torch, we don't divide the kernel here. We divide
        # when it is fed to the convolution, since we use it to generate the
        # coefficient map.
        kernel = self.kernel
        norm_kernel = (kernel / np.sum(kernel))

        # ----------
        # Compute the adjustment coef.
        # This allows our mean computation to compensate for the border area,
        # where you have less terms adding up. Torch used convolution with a
        # ``one'' image, but since we do not want the library to depend on
        # other libraries with convolutions, we do it manually here.
        coef = _lcn_make_coef(norm_kernel, input_shape[2:])

        # ----------
        # Extract convolutional mean per each channel
        # Note that this is different from the subtractive normalization which
        # gets the mean by taking the average over all channels. We first make
        # oc01 filter by adding two new dimensions at the front
        norm_kernel = norm_kernel[np.newaxis, np.newaxis, ...]
        # To pad with zeros, half the size of the kernel (only for 01 dims)
        border_mode = tuple(s // 2 for s in norm_kernel.shape[2:])
        # Reshape the input so that it is (batch_size x channels, 1,
        # input_shape[2], input_shape[3])
        input_reshape = T.reshape(input, (
            input_shape[0] * input_shape[1], 1) + input_shape[2:]
        )
        # Convolve the mean filter. Results in shape of (batch_size x channels,
        # 1, input_shape[2], input_shape[3]).
        conv_mean = self.convolution(input=input_reshape,
                                     filters=T.as_tensor(norm_kernel),
                                     filter_shape=norm_kernel.shape,
                                     border_mode=border_mode,
                                     filter_flip=False,
                                     )

        # ----------
        # Adjust convolutional mean with precomputed coef
        # This is to prevent border values being too small.
        adj_mean = conv_mean / T.as_tensor(coef).dimshuffle(['x', 'x', 0, 1])
        # Note that adj_mean is now in shape of (batch_size x channels, 1,
        # input_shape[2], input_shape[3]).

        # ----------
        # Subtract mean
        sub_normalized = input_reshape - adj_mean

        # ----------
        # Extract convolutional std and adjust it
        conv_std = self.convolution(input=T.sqr(sub_normalized),
                                    filters=T.as_tensor(norm_kernel),
                                    filter_shape=norm_kernel.shape,
                                    border_mode=border_mode,
                                    filter_flip=False,
                                    )
        adj_std = conv_std / T.as_tensor(coef).dimshuffle(['x', 'x', 0, 1])
        adj_std = T.sqrt(adj_std)
        # We are still in (b x c, 1, h, w) shape

        # ----------
        # Another Local mean on the std to get minimum division
        conv_mean_std = self.convolution(input=adj_std,
                                         filters=T.as_tensor(norm_kernel),
                                         filter_shape=norm_kernel.shape,
                                         border_mode=border_mode,
                                         filter_flip=False,
                                         )
        adj_mean_std = (
            conv_mean_std / T.as_tensor(coef).dimshuffle(['x', 'x', 0, 1])
        )

        # ----------
        # Now divide and reshape back
        lcn_normalized = sub_normalized / (
            T.maximum(adj_mean_std, adj_std) + self.eps
        )
        lcn_normalized = T.reshape(lcn_normalized, orig_input_shape)

        # # line for debugging
        # test = theano.function(inputs=[input], outputs=[sub_normalized])

        return lcn_normalized

#
# lasagne_tools.py ends here
