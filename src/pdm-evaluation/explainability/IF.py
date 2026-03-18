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

from numpy._typing import NDArray
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class IForest:
    def __init__(self, n_estimators=200, max_samples=400, contamination='auto', max_features=0.6,
                 bootstrap=False, n_jobs=None, random_state=None, verbose=0,binary=False):
        self.binary=binary
        self.clf = IsolationForest(n_estimators=n_estimators, max_samples=max_samples,
                                   contamination=contamination, max_features=max_features,
                                   bootstrap=bootstrap, n_jobs=n_jobs,
                                   random_state=random_state, verbose=verbose)


    def fit(self, X: NDArray, y=None):
        if self.binary:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            self.clf.fit(X_scaled)
        else:
            self.clf.fit(X)
    def predict(self, X: NDArray):
        # print([-1*sc for sc in self.clf.score_samples(X).tolist()])
        # print("---------")
        if self.binary:
            return [-1*sc for sc in self.clf.score_samples(self.scaler.transform(X)).tolist()]
        return [-1*sc for sc in self.clf.score_samples(X).tolist()]

    def score_samples(self, X: NDArray):
        return self.clf.score_samples(X)