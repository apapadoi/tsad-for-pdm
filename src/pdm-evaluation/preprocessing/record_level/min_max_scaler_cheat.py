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

import pandas as pd
from sklearn.preprocessing import MinMaxScaler as SKLearnMinMaxScaler

from preprocessing.record_level.record_level_pre_processor import RecordLevelPreProcessorInterface
from pdm_evaluation_types.types import EventPreferences


class MinMaxScalerCheat(RecordLevelPreProcessorInterface):
    def __init__(self, event_preferences: EventPreferences):
        super().__init__(event_preferences=event_preferences)
        self.scaler_per_source = {}


    def fit(self, historic_data: list[pd.DataFrame], historic_sources: list[str], event_data: pd.DataFrame) -> None:
        pass
        

    def transform(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> pd.DataFrame:
        self.scaler_per_source[source] = SKLearnMinMaxScaler().fit(target_data)

        if source in self.scaler_per_source:
            return pd.DataFrame(self.scaler_per_source[source].transform(target_data), columns=target_data.columns, index=target_data.index)
        
        return target_data 


    def transform_one(self, new_sample: pd.Series, source: str, is_event: bool) -> pd.Series:
        return self.scaler_per_source[source].transform_one(pd.DataFrame([], columns=new_sample.index).append(new_sample, ignore_index=True)).iloc[0]
    

    def get_params(self):
        return {}
    

    def __str__(self) -> str:
        return 'MinMaxScaler'