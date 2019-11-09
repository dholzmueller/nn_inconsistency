# Copyright 2019 The nn_inconsistency Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import os
import os.path
import numpy as np
import glob
import gzip
import matplotlib.pyplot as plt

import pickle

def existsDir(directory):
    if directory != '':
        if not os.path.exists(directory):
            return False
    return True

def existsFile(file_path):
    return os.path.isfile(file_path)

def ensureDir(file_path):
    directory = os.path.dirname(file_path)
    if directory != '':
        if not os.path.exists(directory):
            os.makedirs(directory)

def matchFiles(file_matcher):
    return glob.glob(file_matcher)

def newDirname(prefix):
    i = 0
    name = prefix
    if existsDir(prefix):
        while existsDir(prefix + "_" + str(i)):
            i += 1
        name = prefix + "_" + str(i)
    os.makedirs(name)
    return name

def getSubfolderNames(folder):
    return [os.path.basename(name)
    for name in os.listdir(folder)
        if os.path.isdir(os.path.join(folder, name))]

def getSubfolders(folder):
    return [os.path.join(folder, name)
    for name in os.listdir(folder)
        if os.path.isdir(os.path.join(folder, name))]

def getFilesInFolder(folder):
    return [os.path.join(folder, name)
    for name in os.listdir(folder)
        if not os.path.isdir(os.path.join(folder, name))]


def writeToFile(filename, content):
    ensureDir(filename)
    file = open(filename, 'w')
    file.truncate()
    file.write(content)
    file.close()


def readFromFile(filename):
    if not os.path.isfile(filename):
        return ''

    file = open(filename, 'r')
    result = file.read()
    file.close()
    return result


def serialize(filename, obj, compressed=False):
    ensureDir(filename)
    if compressed:
        file = gzip.open(filename, 'wb')
    else:
        file = open(filename, 'wb')
    pickle.dump(obj, file, protocol=3)
    file.close()

def deserialize(filename, compressed=False):
    if compressed:
        file = gzip.open(filename, 'rb')
    else:
        file = open(filename, 'rb')
    result = pickle.load(file)
    file.close()
    return result

def plot_cdf(x):
    y_space = np.linspace(0, 1, len(x)+1)
    y_matrix = np.array([y_space[:-1], y_space[1:]])
    x_matrix = np.sort(x)
    x_matrix = np.array([x_matrix, x_matrix])
    plt.plot(np.transpose(x_matrix).flatten(), np.transpose(y_matrix).flatten())
