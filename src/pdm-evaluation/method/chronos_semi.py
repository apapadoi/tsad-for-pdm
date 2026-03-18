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

import os
from pathlib import Path
import logging

from sklearn.ensemble import IsolationForest as isolation_forest
import pandas as pd
import numpy as np
import mlflow
import subprocess
import torch
from chronos import ChronosPipeline
import uuid

from method.semi_supervised_method import SemiSupervisedMethodInterface
from exceptions.exception import NotFitForSourceException
from pdm_evaluation_types.types import EventPreferences
from method.chronos_train.train import main as chronos_train_main
from utils.utils import convert_to_arrow


class ChronosSemi(SemiSupervisedMethodInterface):
    def __init__(self, 
                 event_preferences: EventPreferences, 
                 num_samples: int, 
                 context_length: int,
                 slide: int = 1, 
                 min_past: int = 1,
                 seed: int = 42,
                 model: str = "amazon/chronos-t5-small", 
                 max_steps: int = 1,
                 learning_rate: float = 1e-3,
                 device_type: str = "cuda",
                 torch_dtype: torch.Tensor = torch.bfloat16,
                 forecasting_distance: str = 'euclidean',
                 rolling_mean: bool = False,
                 *args, **kwargs):
        super().__init__(event_preferences=event_preferences)

        self.ftraining_batch=None
        
        self.initial_args = args
        self.initial_kwargs = kwargs

        self.num_samples = num_samples
        self.context_length = context_length
        self.prediction_length = slide
        self.min_past = min_past
        self.seed = seed

        self.model = model
        self.device_type = device_type
        self.torch_dtype = torch_dtype
        self.forecasting_distance = forecasting_distance
        self.rolling_mean = rolling_mean

        self.max_steps = max_steps
        self.learning_rate = learning_rate

        self.historic_data_per_source = {}

        self.own_uuid = str(uuid.uuid4())

        self.input_file_template = f'input_file_chronos_{self.own_uuid}_'
        self.output_dir_template = f'./chronos_semi_output_{self.own_uuid}/'


    def fit(self, historic_data: list[pd.DataFrame], historic_sources: list[str], event_data: pd.DataFrame) -> None:
        for current_historic_data, current_historic_source in zip(historic_data, historic_sources):
            self.historic_data_per_source[current_historic_source] = current_historic_data.copy()


    def predict(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> list[float]:
        if source not in self.historic_data_per_source.keys():
            raise NotFitForSourceException()

        result_per_column = []
        current_historic_data = self.historic_data_per_source[source]
        initial_target_data = target_data.copy()

        while current_historic_data.shape[0] < self.context_length + self.prediction_length + 1:
            point_to_pre_append = current_historic_data.iloc[0]
            current_historic_data = pd.concat([
                pd.DataFrame([point_to_pre_append], columns=current_historic_data.columns),
                current_historic_data
            ])

        for i in range(self.context_length):
            point_to_pre_append = target_data.iloc[0]
            target_data = pd.concat([
                pd.DataFrame([point_to_pre_append], columns=target_data.columns),
                target_data
            ])

        for column_index, column in enumerate(target_data):
            print(f'{column_index + 1} / {len(target_data.columns)}')

            current_input_file = self.input_file_template + f'{column}.arrow'
        
            convert_to_arrow(
                path=current_input_file,
                time_series=[current_historic_data[column].to_numpy()],
                start_times=[current_historic_data.index[0].to_numpy()],
            )
            if self.ftraining_batch is None:
                training_batch_size = 1024
            else:
                training_batch_size=self.ftraining_batch
            current_fine_tuned_model_dir = None

            while True: 
                try:   
                    logging.info(f'Trying training batch size {training_batch_size}')

                    current_fine_tuned_model_dir = chronos_train_main(
                        training_data_paths=[current_input_file],
                        column=column,
                        output_dir=self.output_dir_template,
                        context_length=self.context_length,
                        prediction_length=self.prediction_length,
                        num_samples=self.num_samples,
                        min_past=self.min_past,
                        max_steps=self.max_steps,
                        learning_rate=self.learning_rate,
                        model_id=self.model,
                        seed=self.seed,
                        per_device_train_batch_size=training_batch_size,
                        dataloader_num_workers=0  # Disable multiprocessing in DataLoader
                    )

                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

                    break
                except RuntimeError as e:
                    print(e)
                    logging.warning(f'Training batch size {training_batch_size} failed. Applying halving...')
                    training_batch_size = training_batch_size // 2
            
            self.ftraining_batch=training_batch_size
            assert current_fine_tuned_model_dir != None

            current_pipeline = ChronosPipeline.from_pretrained(
                current_fine_tuned_model_dir,
                device_map=self.device_type,
                torch_dtype=self.torch_dtype
            )

            os.remove(current_input_file)

            batch_size = 1024
            
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

                    for root, dirs, files in os.walk(self.output_dir_template):
                        for file in files:
                            file_path = os.path.join(root, file)
                            
                            os.remove(file_path)
                    
                    for root, dirs, files in os.walk(self.output_dir_template, topdown=False):
                        for dir in dirs:
                            dir_path = os.path.join(root, dir)
                            
                            os.rmdir(dir_path)

                    if os.path.exists(self.output_dir_template):
                        os.rmdir(self.output_dir_template)                                                                      
                    
                    break
                except RuntimeError as e:
                    logging.warning(f'Batch size {batch_size} failed. Applying halving...')
                    batch_size = batch_size // 2


        result_per_column = np.array(result_per_column)
        if self.forecasting_distance == 'euclidean':
            scores = np.linalg.norm(result_per_column.T - initial_target_data.values, axis=1)
        else:
            raise RuntimeError('Other forecasting distance options are not implemented yet')
        
        if self.rolling_mean:
            def rolling_mean_with_slide(arr, slide):
                n = len(arr)
                rolling_means = np.zeros(n)

                for i in range(0, n, slide):
                    window = arr[i:i+slide]
                    mean_val = np.mean(window)
                    rolling_means[i:i+slide] = mean_val

                if i-slide<n:
                        rolling_means[ i-slide:] = np.mean(arr[i-slide:])

                return rolling_means.tolist()

            return rolling_mean_with_slide(scores, self.prediction_length)


        return scores.tolist()
        

    def predict_one(self, new_sample: pd.Series, source: str, is_event: bool) -> float:
         # TODO need to keep buffer until profile size are encountered and then start predicting
        return 0.0
    

    def get_library(self) -> str:
        return 'no_save'
    

    def __str__(self) -> str:
        return 'ChronosSemi'
    

    def get_params(self) -> dict:
        return {
            'context_length': self.context_length,
            'slide': self.prediction_length,
            'num_samples': self.num_samples,
            'model': self.model,
            'device_type': self.device_type,
            'torch_dtype': self.torch_dtype,
            'forecasting_distance': self.forecasting_distance,
            'min_past': self.min_past,
            'seed': self.seed,
            'max_steps': self.max_steps,
            'learning_rate': self.learning_rate,
            'rolling_mean': self.rolling_mean,
        }
    

    def get_all_models(self):
        pass