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

from method.semi_supervised_method import SemiSupervisedMethodInterface
from pdm_evaluation_types.types import EventPreferences
import pandas as pd
from method.USAD.usadCore import usadCore
from exceptions.exception import NotFitForSourceException

class usad(SemiSupervisedMethodInterface):
    def __init__(self, event_preferences: EventPreferences, window_size=10,lr=0.001, num_epochs=15,BATCH_SIZE=4,hidden_size=16,train_val=0.9,minmax=True, *args,
                 **kwargs):
        super().__init__(event_preferences=event_preferences)
        self.window_size = window_size
        self.BATCH_SIZE = BATCH_SIZE
        self.hidden_size = hidden_size
        self.N_EPOCHS = num_epochs
        self.train_val = train_val
        self.minmax=minmax
        self.lr = lr
        self.models={}


    def fit(self, historic_data: list[pd.DataFrame], historic_sources: list[str], event_data: pd.DataFrame) -> None:

        for source,data in zip(historic_sources,historic_data):

            self.models[source]=usadCore( window_size=min(self.window_size, data.shape[0] - 1), num_epochs=self.N_EPOCHS,lr=self.lr,
                                          BATCH_SIZE=self.BATCH_SIZE,hidden_size=self.hidden_size,train_val=self.train_val,
                                          minmax=self.minmax)
            self.models[source].fit(data.copy())


    def predict(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> list[float]:
        if source in self.models.keys():
            return self.models[source].predict(target_data)
        else:
            raise NotFitForSourceException()

    # TODO: predict one
    def predict_one(self, new_sample: pd.Series, source: str, is_event: bool) -> float:
        pass

    def get_library(self) -> str:
        return 'no_save'


    def get_params(self) -> dict:
        params = {
            "window_size":self.window_size,
            "BATCH_SIZE":self.BATCH_SIZE,
            "hidden_size":self.hidden_size,
            "train_val":self.train_val,
            "num_epochs":self.N_EPOCHS,
            "lr":self.lr,
        }

        return params


    def __str__(self) -> str:
        return 'USAD'


    def get_all_models(self):
        pass
