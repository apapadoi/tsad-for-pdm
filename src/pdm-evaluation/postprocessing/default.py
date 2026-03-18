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

from postprocessing.post_processor import PostProcessorInterface


class DefaultPostProcessor(PostProcessorInterface):
    def transform(self, scores: list[float], source: str, event_data: pd.DataFrame) -> list[float]:
        return scores


    def transform_one(self, score_point: float, source: str, is_event: bool) -> float:
        return score_point
    

    def get_params(self):
        return {}


    def __str__(self) -> str:
        return 'Default'