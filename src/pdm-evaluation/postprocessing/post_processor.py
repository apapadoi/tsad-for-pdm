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


class PostProcessorInterface(abc.ABC):
    def __init__(self, event_preferences: EventPreferences):
        self.event_preferences = event_preferences


    @abc.abstractmethod
    def transform(self, scores: list[float], source: str, event_data: pd.DataFrame) -> list[float]:
        pass


    @abc.abstractmethod
    def transform_one(self, score_point: float, source: str, is_event: bool) -> float:
        pass


    @abc.abstractmethod
    def get_params(self):
        pass
    

    @abc.abstractmethod
    def __str__(self) -> str:
        pass