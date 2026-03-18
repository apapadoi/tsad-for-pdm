# Copyright 2026 Anastasios Papadopoulos, Apostolos Giannoulidis
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

from scipy.spatial.distance import cdist
from dtaidistance import dtw
from sklearn.metrics.pairwise import rbf_kernel
#from tslearn.metrics import cycc
import numpy as np


# calculate all pair distances between two dataframes a and b
def calculate_distance_many_to_many( a, b, metric):
    distances = []
    if metric == 'dtw':
        for s1 in a.values:
            distances.append([])
            for s2 in b.values:
                ddist = dtw.distance_fast(s1, s2)
                distances[-1].append(ddist)
    # elif metric == "cc":
    #     for s1 in a.values:
    #         distances.append([])
    #         for s2 in b.values:
    #             ddist = cross_dists(s1, s2)
    #             distances[-1].append(ddist)
    elif 'rbf_kernel' in metric:
        if metric == "rbf_kernel":
            gamma = 0.5
        else:
            gamma = float(metric.split("kernel")[-1])
        distances = 1 - rbf_kernel(a.values, b.values, gamma=gamma)
        distances = distances
    else:
        distances = cdist(a.values, b.values, metric)
    return distances


# calculate distance from point to all profile points
def calculate_distance_many_to_one(a, b, metric):
    distances = []
    if metric == 'dtw':
        for s1 in a.values:
            ddist = dtw.distance_fast(s1, b)
            distances.append(ddist)
    # elif metric == "cc":
    #     for s1 in a.values:
    #         ddist = cross_dists(s1, b)
    #         distances.append(ddist)
    elif 'rbf_kernel' in metric:
        if metric == "rbf_kernel":
            gamma = 0.5
        else:
            gamma = float(metric.split("kernel")[-1])
        distances = 1 - rbf_kernel(b.reshape(1, -1), a.values, gamma=gamma)
    else:
        distances = cdist(b.reshape(1, -1), a, metric)[0]
    return distances
from tslearn.metrics import cdist_normalized_cc, y_shifted_sbd_vec

def cross_dists( s1, s2):
    return 1. - cdist_normalized_cc(np.expand_dims(s1, axis=0), np.expand_dims(s2, axis=0)).max()