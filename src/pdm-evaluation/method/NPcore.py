import logging
import math

import numpy as np
from sklearn.utils import check_random_state, random
from scipy.spatial import distance
from sklearn.preprocessing import scale as sk_scale
import pandas as pd
from numpy import inf


class NeighborProfileAllFeatures():

    def __init__(self,
                 n_nnballs=100,
                 max_sample=8,
                 random_state=None,
                 scale="zscore",
                 sub_sequence_length=10,
                 aggregation_strategy: str = 'avg'
                 ):
        self.n_nnballs = n_nnballs
        self.max_sample = max_sample
        self.random_state = random_state
        self.scale = scale
        self.sub_sequence_length = sub_sequence_length
        self.aggregation_strategy = aggregation_strategy

        self.modelPerColumn= {}


    def fit(self,data :pd.DataFrame):
        for col in data.columns:
            self.modelPerColumn[col]=NeighborProfile(n_nnballs=self.n_nnballs,max_sample=self.max_sample
                                                     ,random_state=self.random_state
                                                     ,scale=self.scale)
            series=data[col].values
            self.modelPerColumn[col].fit(series,self.sub_sequence_length)

    def predict(self,data :pd.DataFrame):
        scores_for_all_dimensions=[]
        for col in data.columns:
            series = data[col].values
            nprofile=self.modelPerColumn[col].estimate_for_time_series(series)
            # if sum([1 if math.isinf(npr) else 0 for npr in nprofile])>0:
            #     print("inf")
            #
            # #nprofile=[0 if math.isinf(npr) and npr<0 else npr for npr in nprofile]
            # #nprofile=[1 if math.isinf(npr) and npr>0 else npr for npr in nprofile]

            scores_for_all_dimensions.append(nprofile)
        if self.aggregation_strategy == 'avg':
            num_dimensions = len(scores_for_all_dimensions)
            return [sum(current_time_step_values) / num_dimensions for current_time_step_values in
                    zip(*scores_for_all_dimensions)]
        else:
            return [max(current_time_step_values) for current_time_step_values in zip(*scores_for_all_dimensions)]
class NeighborProfile():

    def __init__(self,
                 n_nnballs=100,
                 max_sample=8,
                 random_state=None,
                 scale="zscore"):
        self.n_nnballs = n_nnballs
        self.max_sample = max_sample
        self.random_state = random_state
        self.scale = scale

    def fit(self, T, d):
        rnd = check_random_state(self.random_state)
        l = len(T)

        if self.max_sample > l - d + 1:
            logging.warning(f'NPCore.py:fit max_sample is greater than len(T) - sub_sequence_length + 1. Duplication will be applied until this condition is reversed.')

        while self.max_sample > len(T) - d + 1:
            T = np.concatenate((T, T))

        l = len(T)
        self.d = d

        # construct nn balls
        self.list_of_nn_ball = []
        for _ in range(self.n_nnballs):
            seq_idx = random.sample_without_replacement(l - d + 1, self.max_sample, random_state=rnd)
            X = _construct_array(T, seq_idx, d)
            X = _scale(X, scale=self.scale)
            triu = distance.pdist(X)

            distance_matrix = np.zeros((self.max_sample, self.max_sample))
            distance_matrix[np.triu_indices(self.max_sample, 1)] = triu
            distance_matrix += distance_matrix.T
            distance_matrix += np.max(triu) * np.eye(self.max_sample)

            nn_distance = np.min(distance_matrix, axis=0)
            nn_ball = (X, nn_distance)
            self.list_of_nn_ball.append(nn_ball)

        return self

    def estimate_for_time_series(self, T):
        profile = []

        Y = _subsequences_from_series(T, range(0, len(T) - self.d), self.d)
        profile += list(self.estimate_for_subsequences(Y))
        return profile

    def estimate_for_subsequences(self, Y):
        Y = _scale(Y, self.scale)
        r_list = []
        for _, nn_ball in enumerate(self.list_of_nn_ball):
            nnball_c, nnball_r = nn_ball[0], nn_ball[1]
            cdist = distance.cdist(Y, nnball_c)
            nn_d_idx = cdist.argmin(axis=1)
            nn_d = cdist.min(axis=1)

            invalid_idx = nnball_r[nn_d_idx] < nn_d
            nn_r = nnball_r[nn_d_idx]
            nn_r[invalid_idx] = nn_d[invalid_idx]
            temp=np.log(nn_r)

            # This is added to deal with -inf
            #temp = [-max(temp) if npr<- else npr for npr in temp]
            #
            #if sum([1 if math.isinf(sm) else 0 for sm in temp]) >= 1:
            #    ok = 'ok'
            r_list.append(temp)
        profile = np.mean(r_list, axis=0)

        return profile


def _subsequences_from_series(ts, idx, d):
    X = np.empty((len(idx), d))
    for i in range(len(idx)):
        X[i, :] = ts[idx[i]:idx[i] + d]
    return X


def _construct_array(T, idx, d):
    X = np.empty((len(idx), d))
    for i, s in enumerate(idx):
        X[i, :] = T[s:s + d]
    return X


def _scale(X, scale):
    if scale == "auto":
        return X

    if scale == "demean":
        return sk_scale(X, axis=1, with_std=False)

    if scale == "zscore":
        #prex = X
        afterX = sk_scale(X, axis=1)
        # COnstant remain the same
        # for sampleold,sample in zip(prex,afterX):
        #     if max(sampleold)-min(sampleold)<0.00000000000001:
        #         print("cinst")
        return afterX
