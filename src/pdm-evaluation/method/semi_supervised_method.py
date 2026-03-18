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
from method.method import MethodInterface

class SemiSupervisedMethodInterface(MethodInterface):
    @abc.abstractmethod
    def fit(self, historic_data: list[pd.DataFrame], historic_sources: list[str], event_data: pd.DataFrame) -> None:
        """
        This method is used to fit a anomaly detection model in (relative) normal data, where the data are passed in form
        of Dataframes along with their respected source.

        :param historic_data: a list of Dataframes (used to fit a semi-supervised model). The `historic_data` list parameter elements should be copied if a corresponding method needs to store them for future processing
        :param historic_sources: a list with strings (names) of the different sources
        :param event_data: event data that are produced from the different sources
        :return: None.
        """

        pass