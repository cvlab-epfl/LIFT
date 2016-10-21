# solvers.py ---
#
# Filename: solvers.py
# Description:
# Author: Kwang
# Maintainer:
# Created: Thu Feb 18 14:40:50 2016 (+0100)
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


import importlib
import time


def CreateNetwork(pathconf, param, train_data_in, valid_data_in, test_data_in):

    # ------------------------------------------------------------------------
    # Setup and load correct module for the network configuration

    # Load proper network and set proper num_siamese (in the param!)
    # ------------------------------------------------------------------------
    network_module = importlib.import_module(
        'Utils.networks.' + param.dataset.dataType.lower(
        ) + '_' + param.model.modelType.lower())

    # ------------------------------------------------------------------------
    # Initialize network Config
    myNetConfig = network_module.NetworkConfig()

    # ------------------------------------------------------------------------
    # Copy over all other attributes to Config, destroying the group
    for _group in param.__dict__.keys():
        for _key in getattr(param, _group).__dict__.keys():
            setattr(myNetConfig, _key, getattr(getattr(param, _group), _key))

    # ------------------------------------------------------------------------
    # Config fields which need individual attention

    # directories
    myNetConfig.save_dir = pathconf.result

    # dataset info (let's say these should be given in the data structure?)
    myNetConfig.num_channel = train_data_in.num_channel
    myNetConfig.patch_height = train_data_in.patch_height
    myNetConfig.patch_width = train_data_in.patch_width
    myNetConfig.out_dim = train_data_in.out_dim

    # ------------------------------------------------------------------------
    # Actual instantiation and setup
    myNet = network_module.Network(myNetConfig)

    # myNet.setupSGD()  # Setup SGD
    # myNet.setup4Train()  # Setup Train
    # myNet.compile4Train()  # Compile Train
    myNet.setup4Test()  # Setup Test
    myNet.compile4Test()  # Compile Test
    myNet.setupSpecific()  # Setup specific to runType
    myNet.compileSpecific()  # Compile specific to runType

    return myNet


def CreateNetwork4Image(pathconf, param, image, verbose=True):

    # ------------------------------------------------------------------------
    # Setup and load correct module for the network configuration

    # Load proper network and set proper num_siamese (in the param!)
    # ------------------------------------------------------------------------
    network_module = importlib.import_module(
        'Utils.networks.' + param.dataset.dataType.lower(
        ) + '_' + param.model.modelType.lower())

    # ------------------------------------------------------------------------
    # Initialize network Config
    myNetConfig = network_module.NetworkConfig()

    # ------------------------------------------------------------------------
    # Copy over all other attributes to Config, destroying the group
    for _group in param.__dict__.keys():
        for _key in getattr(param, _group).__dict__.keys():
            setattr(myNetConfig, _key, getattr(getattr(param, _group), _key))

    # ------------------------------------------------------------------------
    # Config fields which need individual attention

    # directories
    myNetConfig.save_dir = pathconf.result

    # # dataset info (let's say these should be given in the data structure?)
    # myNetConfig.num_channel = train_data_in.num_channel
    # myNetConfig.patch_size = train_data_in.patch_size
    # myNetConfig.out_dim = train_data_in.out_dim

    # image info
    myNetConfig.num_channel = 1  # now we use grayscale
    myNetConfig.patch_height = image.shape[0]
    myNetConfig.patch_width = image.shape[1]
    myNetConfig.out_dim = 1

    # Override batch_size
    myNetConfig.batch_size = 1

    # ------------------------------------------------------------------------
    # Actual instantiation and setup
    myNet = network_module.Network(myNetConfig,
                                   rng=None,
                                   bTestWholeImage=True,
                                   verbose=verbose)

    # myNet.setupSGD()  # Setup SGD
    # myNet.setup4Train()         # Setup Train
    # myNet.compile4Train()        # Compile Train
    myNet.setup4Test()  # Setup Test
    # mynet.compile4Test()        # Compile Test
    # myNet.setupSpecific()       # Setup specific to runType
    myNet.compileSpecific(verbose=verbose)  # Compile specific to runType

    return myNet


def TestImage(pathconf, param, image, verbose=True, network_weights=None):

    start_time = time.clock()
    # ------------------------------------------------------------------------
    # Create the Network
    myNet = CreateNetwork4Image(pathconf, param, image, verbose=verbose)
    end_time = time.clock()
    compile_time = (end_time - start_time) * 1000.0

    # # -----------------------------------------------------------------------
    # # Disable Garbage Collection so that it does not interfere with theano
    # gc.disable()
    # gc.collect()                # collect once before running experiments

    # ------------------------------------------------------------------------
    # Run Train
    test_res = myNet.runTest4Image(
        image, verbose=verbose, network_weights=network_weights)

    # # -----------------------------------------------------------------------
    # # Re-enable GC
    # gc.enable()

    return test_res, compile_time


def Test(pathconf,
         param,
         test_data_in,
         test_mode=None,
         network_weights=None):

    # ------------------------------------------------------------------------
    # Create the Network
    start_time = time.clock()
    myNet = CreateNetwork(pathconf, param, test_data_in, test_data_in,
                          test_data_in)
    end_time = time.clock()
    compile_time = (end_time - start_time) * 1000.0

    # ------------------------------------------------------------------------
    # Run Test
    return myNet.runTest(test_data_in,
                         test_mode=test_mode,
                         deterministic=True,
                         model_epoch="",
                         verbose=True,
                         network_weights=network_weights) + (compile_time,)


#
# solvers.py ends here
