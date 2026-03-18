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

import abc

import pandas as pd

from pdm_evaluation_types.types import EventPreferences


class ThresholderInterface(abc.ABC):
    def __init__(self, event_preferences: EventPreferences):
        self.event_preferences = event_preferences
        

    @abc.abstractmethod
    def infer_threshold(self, scores: list[float], source: str, event_data: pd.DataFrame, scores_dates: list[pd.Timestamp]) -> list[float]:
        """
            Returns a threshold value for each score in the 'scores' list parameter
        """
        pass


    @abc.abstractmethod
    def infer_threshold_one(self, score: float, source: str, event_data: pd.DataFrame) -> float:
        pass


    @abc.abstractmethod
    def get_params(self):
        pass
    
    
    @abc.abstractmethod
    def __str__(self) -> str:
        """
            Returns a string representation of the corresponding thresholder
        """
        pass