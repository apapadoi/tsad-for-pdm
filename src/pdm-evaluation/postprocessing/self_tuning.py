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

import statistics

import numpy as np
import pandas as pd

from postprocessing.post_processor import PostProcessorInterface
from pdm_evaluation_types.types import EventPreferences


class SelfTuningPostProcessor(PostProcessorInterface):
    def __init__(self, event_preferences: EventPreferences, window_length: int):
        super().__init__(event_preferences=event_preferences)
        self.window_length = window_length
        self.scores_buffer_per_source = {}


    def transform(self, scores: list[float], source: str, event_data: pd.DataFrame) -> list[float]:
        if self.window_length == 0:
            return scores

        scores_for_calculating_metrics_init = scores[:self.window_length]
        scores_for_calculating_metrics=[]
        for sc in scores_for_calculating_metrics_init:
            if len(scores_for_calculating_metrics)==0:
                scores_for_calculating_metrics.append(sc)
            elif sc==scores_for_calculating_metrics[-1]:
                continue
            else:
                scores_for_calculating_metrics.append(sc)
        if len(scores_for_calculating_metrics)>1:
            mean, std = statistics.mean(scores_for_calculating_metrics), np.std(scores_for_calculating_metrics)
        else:
            mean = statistics.mean(scores_for_calculating_metrics)
            std=0
            
        if std == 0.0:
            return [sc - mean for sc in scores]

        return list(map(lambda score: (score - mean) / std, scores))
    

    def transform_one(self, score_point: float, source: str, is_event: bool) -> float:
        if self.window_length == 0:
            return score_point

        if source not in self.scores_buffer_per_source:
            self.scores_buffer_per_source[source] = []

        self.scores_buffer_per_source[source].append(score_point)

        if len(self.scores_buffer_per_source[source]) < self.window_length:
            return score_point
        else:
            self.scores_buffer_per_source[source] = self.scores_buffer_per_source[source][:self.window_length]

            assert len(self.scores_buffer_per_source[source]) == self.window_length

            mean, std = statistics.mean(self.scores_buffer_per_source[source]), statistics.stdev(self.scores_buffer_per_source[source])

            return (score_point - mean) / std
    

    def get_params(self):
        return {
            'window_length': self.window_length
        }


    def __str__(self) -> str:
        return f'Self_Tuning'