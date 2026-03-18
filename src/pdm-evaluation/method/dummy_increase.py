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
from method.semi_supervised_method import SemiSupervisedMethodInterface
from pdm_evaluation_types.types import EventPreferences


class DummyIncrease(SemiSupervisedMethodInterface):
    def __init__(self,
                 event_preferences: EventPreferences,
                 **kwargs
                 ):
        super().__init__(event_preferences=event_preferences)

    def fit(self, historic_data: list[pd.DataFrame], historic_sources: list[str], event_data: pd.DataFrame):
        pass

    def predict(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> list[float]:
        scores=[i for i in range(len(target_data.index))]
        scores[0]=0
        return scores



    def predict_one(self, new_sample: pd.Series, source: str, is_event: bool) -> float:
        # TODO need to keep buffer until profile size are encountered and then start predicting
        return 0

    def get_library(self) -> str:
        return 'no_save'

    def __str__(self) -> str:
        return 'DummyAddAlarms'

    def get_params(self) -> dict:
        return {
        }

    def get_all_models(self):
        pass