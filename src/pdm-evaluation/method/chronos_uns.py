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

import logging

import pandas as pd
import numpy as np
import mlflow
import torch
from chronos import ChronosPipeline

from method.unsupervised_method import UnsupervisedMethodInterface
from exceptions.exception import NotFitForSourceException
from pdm_evaluation_types.types import EventPreferences


class ChronosUns(UnsupervisedMethodInterface):
    def __init__(self, 
                 event_preferences: EventPreferences, 
                 num_samples: int, 
                 context_length: int,
                 slide: int = 1, 
                 model: str = "amazon/chronos-t5-small", 
                 device_type: str = "cuda:1",
                 torch_dtype: torch.Tensor = torch.bfloat16,
                 forecasting_distance: str = 'euclidean',
                 *args, **kwargs):
        super().__init__(event_preferences=event_preferences)
        self.initial_args = args
        self.initial_kwargs = kwargs

        self.num_samples = num_samples
        self.context_length = context_length
        self.prediction_length = slide

        self.model = model
        self.device_type = device_type
        self.torch_dtype = torch_dtype
        self.forecasting_distance = forecasting_distance


    def predict(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> list[float]:
        initial_target_data = target_data.copy()
        for i in range(self.context_length):
            point_to_pre_append = target_data.iloc[0]
            target_data = pd.concat([
                pd.DataFrame([point_to_pre_append], columns=target_data.columns),
                target_data
            ])

        current_pipeline = ChronosPipeline.from_pretrained(
                self.model,
                device_map=self.device_type,
                torch_dtype=self.torch_dtype
        )
        result_per_column = []

        for column_index, column in enumerate(target_data):
            print(f'{column_index+1} / {len(target_data.columns)}')
            batch_size = 1024 # 280 for 15, 32 slide
            
            while True:    
                logging.info(f'Trying batch size {batch_size}')
                try:
                    internal_batch_result_for_current_column = []
                    context_list = []
                    for current_prediction_index in range(self.context_length, target_data.shape[0], self.prediction_length):
                        start_index = max(0, current_prediction_index - self.context_length)
                        current_context = target_data.iloc[start_index:current_prediction_index]
                        context_list.append(torch.tensor(current_context[column]))

                        if len(context_list) == batch_size or current_prediction_index == target_data.shape[0] - 1 or current_prediction_index + self.prediction_length >= target_data.shape[0]:
                            forecast = current_pipeline.predict(
                                            context=context_list,
                                            prediction_length=self.prediction_length,
                                            num_samples=self.num_samples
                                        )

                            for forecast_tensor in forecast:
                                low, median, high = np.quantile(forecast_tensor.numpy(), [0.1, 0.5, 0.9], axis=0)
                                internal_batch_result_for_current_column.extend(median.tolist())

                            context_list = []

                    result_per_column.append(internal_batch_result_for_current_column[:initial_target_data.shape[0]])
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    
                    break
                except RuntimeError as e:
                    logging.warning(f'Batch size {batch_size} failed. Applying halving...')
                    batch_size = batch_size // 2


        result_per_column = np.array(result_per_column)
        if self.forecasting_distance == 'euclidean':
            scores = np.linalg.norm(result_per_column.T - initial_target_data.values, axis=1)
        else:
            raise RuntimeError('Other forecasting distance options are not implemented yet')
        
        return scores.tolist()
        

    def predict_one(self, new_sample: pd.Series, source: str, is_event: bool) -> float:
         # TODO need to keep buffer until profile size are encountered and then start predicting
        return 0.0
    

    def get_library(self) -> str:
        return 'no_save'
    

    def __str__(self) -> str:
        return 'ChronosUnsupervised'
    

    def get_params(self) -> dict:
        return {
            'context_length': self.context_length,
            'slide': self.prediction_length,
            'num_samples': self.num_samples,
            'model': self.model,
            'device_type': self.device_type,
            'torch_dtype': self.torch_dtype,
            'forecasting_distance': self.forecasting_distance
        }
    

    def get_all_models(self):
        pass