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

from preprocessing.record_level.record_level_pre_processor import RecordLevelPreProcessorInterface


class DefaultPreProcessor(RecordLevelPreProcessorInterface):
    def fit(self, historic_data: list[pd.DataFrame], historic_sources: list[str], event_data: pd.DataFrame) -> None:
        pass
        

    def transform(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> pd.DataFrame:
        return target_data


    def transform_one(self, new_sample: pd.Series, source: str, is_event: bool) -> pd.Series:
        return new_sample
    

    def get_params(self):
        return {}
    
    
    def __str__(self) -> str:
        return 'Default'