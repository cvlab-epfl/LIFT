# custom_types.py ---
#
# Filename: custom_types.py
# Description: Python Module for custom types
# Author: Kwang
# Maintainer:
# Created: Fri Jan 16 12:01:52 2015 (+0100)
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
#
# Copyright (C), EPFL Computer Vision Lab.
#
#

# Code:

# -------------------------------------------
# Imports
from __future__ import print_function

import hashlib
import os
import shutil
from copy import deepcopy
from datetime import timedelta

import numpy as np
from flufl.lock import Lock
from parse import parse


class pathConfig:
    """ Path config """

    dataset = None
    temp = None
    result = None
    debug = None

    train_data = None
    train_mask = None

    def prefix_dataset(self, param, do_sort=True):
        """ Prefix stuff that will affect keypoint generation and pairing """

        prefixstr = ""

        group_list = ['dataset']

        exclude_list = {}
        exclude_list['dataset'] = ['trainSetList']

        for group_name in group_list:
            key_list = deepcopy(
                list(getattr(param, group_name).__dict__.keys()))
            if do_sort:
                key_list.sort()
            for key_name in key_list:
                if key_name not in exclude_list[group_name]:
                    prefixstr += str(getattr(
                        getattr(param, group_name),
                        key_name))

        prefixstr = hashlib.md5(prefixstr.encode()).hexdigest()
        prefixstr += "/"

        return prefixstr

    def prefix_patch(self, param, do_sort=True):
        """ Prefix stuff that will affect patch extraction """

        prefixstr = ""

        group_list = ['patch']

        exclude_list = {}
        exclude_list['patch'] = []

        for group_name in group_list:
            key_list = deepcopy(
                list(getattr(param, group_name).__dict__.keys()))
            if do_sort:
                key_list.sort()
            for key_name in key_list:
                if key_name not in exclude_list[group_name]:
                    prefixstr += str(getattr(getattr(param,
                                                     group_name), key_name))

        prefixstr = hashlib.md5(prefixstr.encode()).hexdigest()
        prefixstr += "/"

        return prefixstr

    def prefix_learning(self, param, do_sort=True):
        """ Prefix stuff that will affect learning outcome """

        prefixstr = ""

        group_list = ['model', 'learning']

        for group_name in group_list:
            key_list = deepcopy(
                list(getattr(param, group_name).__dict__.keys()))
            if do_sort:
                key_list.sort()
            for key_name in key_list:
                prefixstr += str(getattr(getattr(param, group_name), key_name))

        prefixstr = hashlib.md5(prefixstr.encode()).hexdigest()
        prefixstr += "/"

        return prefixstr

    def setupTrain(self, param, setID):

        # Lock to prevent race condition
        lock_file = ".locks/setup.lock"
        lock = Lock(lock_file)
        lock.lifetime = timedelta(days=2)
        lock.lock()

        # ---------------------------------------------------------------------
        # Base path
        # for dataset
        self.dataset = os.getenv('PROJ_DATA_DIR', '')
        if self.dataset == '':
            self.dataset = os.path.expanduser("~/Datasets")
        self.dataset += "/" + param.dataset.trainSetList[setID]
        # for temp
        self.temp = os.getenv('PROJ_TEMP_DIR', '')
        if self.temp == '':
            self.temp = os.path.expanduser("~/Temp")
        self.temp += "/" + param.dataset.trainSetList[setID]
        # for volatile temp
        self.volatile_temp = os.getenv('PROJ_VOLTEMP_DIR', '')
        if self.volatile_temp == '':
            self.volatile_temp = "/scratch/" + os.getenv('USER') + "/Temp"
        self.volatile_temp += "/" + param.dataset.trainSetList[setID]

        # ---------------------------------------------------------------------
        # Path for the model learning
        resdir = os.getenv('PROJ_RES_DIR', '')
        if resdir == '':
            resdir = os.path.expanduser("~/Results")
        self.result = (resdir + "/" +
                       self.getResPrefix(param) +
                       self.prefix_dataset(param) +
                       self.prefix_patch(param) +
                       self.prefix_learning(param))
        if not os.path.exists(self.result):
            # Check if the un-sorted prefix exists
            unsorted_hash_path = (resdir + "/" +
                                  self.getResPrefix(param, do_sort=False) +
                                  self.prefix_dataset(param, do_sort=False) +
                                  self.prefix_patch(param, do_sort=False) +
                                  self.prefix_learning(param, do_sort=False))

            if os.path.exists(unsorted_hash_path):
                shutil.copytree(unsorted_hash_path, self.result)
                shutil.rmtree(unsorted_hash_path)

        lock.unlock()

    def setupTest(self, param, testDataName):

        # Lock to prevent race condition
        lock_file = ".locks/setup.lock"
        lock = Lock(lock_file)
        lock.lifetime = timedelta(days=2)
        lock.lock()

        # ---------------------------------------------------------------------
        # Base path
        # for dataset
        self.dataset = os.getenv('PROJ_DATA_DIR', '')
        if self.dataset == '':
            self.dataset = os.path.expanduser("~/Datasets")
        self.dataset += "/" + testDataName
        # for temp
        self.temp = os.getenv('PROJ_TEMP_DIR', '')
        if self.temp == '':
            self.temp = os.path.expanduser("~/Temp")
        self.temp += "/" + testDataName
        # for volatile temp
        self.volatile_temp = os.getenv('PROJ_VOLTEMP_DIR', '')
        if self.volatile_temp == '':
            self.volatile_temp = "/scratch/" + os.getenv('USER') + "/Temp"
        self.volatile_temp += "/" + testDataName

        # ---------------------------------------------------------------------
        # Path for data loading
        self.train_data = None  # block these as they should not be used
        self.train_mask = None  # block these as they should not be used
        self.debug = self.dataset + "/debug/" + self.prefix_dataset(param)

        # ---------------------------------------------------------------------
        # Path for the model learning
        resdir = os.getenv('PROJ_RES_DIR', '')
        if resdir == '':
            resdir = os.path.expanduser("~/Results")
        self.result = (resdir + "/" +
                       self.getResPrefix(param) +
                       self.prefix_dataset(param) +
                       self.prefix_patch(param) +
                       self.prefix_learning(param))
        # Check if the un-sorted prefix exists
        unsorted_hash_path = (resdir + "/" +
                              self.getResPrefix(param, do_sort=False) +
                              self.prefix_dataset(param, do_sort=False) +
                              self.prefix_patch(param, do_sort=False) +
                              self.prefix_learning(param, do_sort=False))

        if os.path.exists(unsorted_hash_path):
            shutil.copytree(unsorted_hash_path, self.result)
            shutil.rmtree(unsorted_hash_path)

        lock.unlock()

    def getResPrefix(self, param, do_sort=True):
        trainSetList = deepcopy(list(param.dataset.trainSetList))
        if do_sort:
            trainSetList.sort()
        res_prefix = param.dataset.dataType + '/' + \
            hashlib.md5(
                "".join(trainSetList).encode()).hexdigest()

        # this is probably deprecated
        if 'prefixStr' in param.__dict__.keys():
            print('I am not deprecated!!!')
            res_prefix = param.prefixStr + "/" + res_prefix

        return res_prefix + '/'


class paramGroup:
    """ Parameter group """

    def __init__(self):
        # do nothing
        return


class paramStruct:
    """ Parameter structure """

    def __init__(self):

        # ---------------------------------------------------------------------
        # NOTICE: everything set to None is to make it crash without config

        # ---------------------------------------------------------------------
        # Paramters for patch extraction
        self.patch = paramGroup()

        # ---------------------------------------------------------------------
        # Parameters for synthetic images
        self.synth = paramGroup()

        # ---------------------------------------------------------------------
        # Model parameters
        self.model = paramGroup()

        # ---------------------------------------------------------------------
        # Optimization parameters
        self.learning = paramGroup()

        # ---------------------------------------------------------------------
        # Dataset parameters
        self.dataset = paramGroup()
        self.dataset.dataType = None

        # ---------------------------------------------------------------------
        # Validation parameters (Not used in prefix generation)
        self.validation = paramGroup()

    def loadParam(self, file_name, verbose=True):

        config_file = open(file_name, 'rb')
        if verbose:
            print("Parameters")

        # ------------------------------------------
        # Read the configuration file line by line
        while True:
            line2parse = config_file.readline().decode("utf-8")
            if verbose:
                print(line2parse, end='')

            # Quit parsing if we reach the end
            if not line2parse:
                break

            # Parse
            parse_res = parse(
                '{parse_type}: {group}.{field_name} = {read_value};{trash}',
                line2parse)

            # Skip if it is something we cannot parse
            if parse_res is not None:
                if parse_res['parse_type'] == 'ss':
                    setattr(getattr(self, parse_res['group']), parse_res[
                            'field_name'], parse_res['read_value'].split(','))
                elif parse_res['parse_type'] == 's':
                    setattr(getattr(self, parse_res['group']), parse_res[
                            'field_name'], parse_res['read_value'])
                elif parse_res['parse_type'] == 'd':
                    eval_res = eval(parse_res['read_value'])
                    if isinstance(eval_res, np.ndarray):
                        setattr(getattr(self, parse_res['group']), parse_res[
                                'field_name'], eval_res.astype(int))
                    elif isinstance(eval_res, list):
                        setattr(getattr(self, parse_res['group']), parse_res[
                                'field_name'], [int(v) for v in eval_res])
                    else:
                        setattr(getattr(self, parse_res['group']), parse_res[
                                'field_name'], int(eval_res))
                elif parse_res['parse_type'] == 'f':
                    eval_res = eval(parse_res['read_value'])
                    if isinstance(eval_res, np.ndarray):
                        setattr(getattr(self, parse_res['group']), parse_res[
                                'field_name'], eval_res.astype(float))
                    elif isinstance(eval_res, list):
                        setattr(getattr(self, parse_res['group']), parse_res[
                                'field_name'], [float(v) for v in eval_res])
                    else:
                        setattr(getattr(self, parse_res['group']), parse_res[
                                'field_name'], float(eval_res))
                elif parse_res['parse_type'] == 'sf':
                    setattr(getattr(self, parse_res['group']),
                            parse_res['field_name'],
                            [float(eval(s)) for s in
                             parse_res['read_value'].split(',')])
                elif parse_res['parse_type'] == 'b':
                    setattr(getattr(self, parse_res['group']), parse_res[
                            'field_name'], bool(int(parse_res['read_value'])))
                else:
                    if verbose:
                        print('  L-> skipped')


#
# custom_types.py ends here
