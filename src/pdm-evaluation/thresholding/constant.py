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

from thresholding.thresholder import ThresholderInterface
from pdm_evaluation_types.types import EventPreferences


class ConstantThresholder(ThresholderInterface):
    def __init__(self, event_preferences: EventPreferences, threshold_value: float = 0.5):
        super().__init__(event_preferences=event_preferences)
        self.threshold_value = threshold_value


    def infer_threshold(self, scores: list[float], source: str, event_data: pd.DataFrame, scores_dates: list[pd.Timestamp]) -> list[float]:
        return [self.threshold_value for i in range(len(scores))]
    

    def infer_threshold_one(self, score: float, source: str, event_data: pd.DataFrame) -> float:
        return self.threshold_value


    def get_params(self):
        return {
            'threshold_value': self.threshold_value
        }
    

    def __str__(self) -> str:
        return 'ConstantThresholder'