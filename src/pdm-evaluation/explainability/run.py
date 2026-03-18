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

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tensorflow import timestamp
from tqdm import tqdm

import utils
from IF import IForest
from XAI_method import c_kshap, c_lime


def transform_(x):
    x_data_tensor = torch.tensor(np.asarray(x), dtype=torch.float32)
    return x_data_tensor

class make_callable():
    def __init__(self, method):
        self.method = method

    def forward(self, x):
        y = self.method.predict(x.detach().numpy())[0]
        return torch.tensor(y, dtype=torch.float32)



def run_M(Mname, method, xaimethod, episodes,profile,predictive_horizon,pivot):
    explainations = {
        "model": Mname,
        "name": f"{xaimethod.__name__}",
        "label": [],
        "timestamp": [],
        "source": [],
        "important_features": [],
        "score": []
    }
    count=0
    for episode in episodes:

        sources=[s for s in episode["source"].values]
        episode.drop(columns=["source"],inplace=True)
        episode.index=pd.to_datetime(episode["Timestamp"])
        episode.drop(columns=["Timestamp"],inplace=True)
        episode.drop(columns=["RUL"],inplace=True)
        count+=1
        print(f"Processing episode {count}/{len(episodes)}")
        fit_data=episode.iloc[:profile].values
        method.fit(fit_data,None)
        callmethod = make_callable(method)
        predict_data=episode
        sources=sources
        last_timestamp=predict_data.index[-1]
        phtm=last_timestamp - pd.Timedelta(predictive_horizon)

        triplets=[]
        for x, ind, source,qi in zip(predict_data.values, predict_data.index, sources,[q for q in range(len(sources))]):
            triplets.append((x, ind, source,qi))
        countinner=0
        for trpl in tqdm(triplets):
            countinner+=1
            x, ind, source,qi = trpl
            if qi%pivot!=0:
                continue
            lab=0
            if ind > phtm and countinner>5:
                lab=1
            important_features=xaimethod(callmethod.forward,transform_(x))
            explainations["label"].append(lab)
            explainations["important_features"].append(important_features)
            explainations["timestamp"].append(ind)
            explainations["source"].append(source)
            explainations["score"].append(method.predict([x])[0])
    with open(f"Pickles/EDP/{Mname}_{xaimethod.__name__}_piv_{pivot}_impo.pkl", 'wb') as file:
        pickle.dump(explainations, file)


if __name__ == "__main__":
    pickle_file = "EDP/episodes_rtf.pkl"
    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as file:
            episodes, rtf = pickle.load(file)
    else:
        episodes, rtf = utils.prepare()
        with open(pickle_file, 'wb') as file:
            pickle.dump((episodes, rtf), file)
    print(sum([len(ep) for ep in episodes]))
    # run_M("IF", IForest(), c_lime, episodes, 900, "86400 minutes",pivot=1000)
    run_M("IF", IForest(random_state=42), c_kshap, episodes, 900, "86400 minutes",pivot=50)