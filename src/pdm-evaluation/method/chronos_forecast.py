import re

from sklearn.ensemble import IsolationForest as isolation_forest
import pandas as pd
import numpy as np
import mlflow
import subprocess
import torch
from chronos import ChronosPipeline

from method.semi_supervised_method import SemiSupervisedMethodInterface
from exceptions.exception import NotFitForSourceException
from pdm_evaluation_types.types import EventPreferences
from method.unsupervised_method import UnsupervisedMethodInterface


class ForecastingAnomalyPredictionMethod(SemiSupervisedMethodInterface):
    def __init__(self, 
                 event_preferences: EventPreferences, 
                 num_samples: int, 
                 context_length: int,
                 prediction_length: int, 
                 anomaly_detector: UnsupervisedMethodInterface,
                 forecasting_model: str = "amazon/chronos-t5-small", 
                 device_type: str = "cuda:1",
                 torch_dtype: torch.Tensor = torch.bfloat16,
                 aggregation_strategy: str = 'avg',
                 *args, **kwargs):
        super().__init__(event_preferences=event_preferences)
        self.initial_args = args
        self.initial_kwargs = kwargs

        self.num_samples = num_samples
        self.context_length = context_length
        self.prediction_length = prediction_length

        self.forecasting_model = forecasting_model
        self.device_type = device_type
        self.torch_dtype = torch_dtype
        self.aggregation_strategy = aggregation_strategy
        
        self.anomaly_detector = anomaly_detector
        self.anomaly_detector_params = {re.sub('anomaly_detector_', '', k): v for k, v in kwargs.items() if 'anomaly_detector' in k}

        self.pipeline_per_source = {}
        self.historic_data_per_source = {}
        self.anomaly_detector_per_source = {}


    def fit(self, historic_data: list[pd.DataFrame], historic_sources: list[str], event_data: pd.DataFrame) -> None:
        for current_historic_data, current_historic_source in zip(historic_data, historic_sources):
            self.historic_data_per_source[current_historic_source] = current_historic_data.copy()

            self.anomaly_detector_per_source[current_historic_source] = self.anomaly_detector(self.event_preferences, **self.anomaly_detector_params)
            self.anomaly_detector_per_source[current_historic_source].fit([current_historic_data], [current_historic_source], event_data)


    def predict(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> list[float]:
        self.pipeline_per_source[source] = ChronosPipeline.from_pretrained(
                                                                self.forecasting_model,
                                                                device_map=self.device_type,
                                                                torch_dtype=self.torch_dtype
                                                            )
        print(f'source: {source}')
        if source not in self.pipeline_per_source.keys():
            raise NotFitForSourceException()
        
        current_pipeline = self.pipeline_per_source[source]
        result_per_column = []

        whole_scenario = pd.concat([
            self.historic_data_per_source[source],
            target_data
        ])

        for column_index, column in enumerate(target_data):
            print(f'{column_index+1} / {len(target_data.columns)}')
            internal_batch_result_for_current_column = []
            context_list = []
            for current_prediction_index in range(self.historic_data_per_source[source].shape[0], whole_scenario.shape[0]):
                # TODO return 0 if current_prediction_index - self.prediction_length < 0
                start_index = max(0, current_prediction_index - self.context_length - self.prediction_length)
                current_context = whole_scenario.iloc[start_index:current_prediction_index - self.prediction_length]
                context_list.append(torch.tensor(current_context[column]))

                if len(context_list) == 1000 or current_prediction_index == whole_scenario.shape[0] - 1:
                    forecast = current_pipeline.predict(
                                    context=context_list,
                                    prediction_length=self.prediction_length,
                                    num_samples=self.num_samples
                                )

                    for forecast_tensor in forecast:
                        low, median, high = np.quantile(forecast_tensor.numpy(), [0.1, 0.5, 0.9], axis=0)
                        internal_batch_result_for_current_column.append(median.tolist()[-1])

                    context_list = []

            result_per_column.append(internal_batch_result_for_current_column)
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        scores_per_column = []
        for result_column_index, result_for_current_column in enumerate(result_per_column):
            # scores_per_column.append([])
        # for detection_target_data_index, detection_target_data_list in enumerate(result_for_current_column):
            current_detector = self.anomaly_detector_per_source[source]
            # TODO need handling for unsupervised anomaly detector
            scores_for_detection_target_data = current_detector.predict(
                pd.DataFrame(result_for_current_column, columns=['y']),
                source,
                event_data
            )

            scores_per_column.append(scores_for_detection_target_data)


        if self.aggregation_strategy == 'avg':
            num_dimensions = len(scores_per_column)
            return [sum(current_time_step_values) / num_dimensions for current_time_step_values in
                    zip(*scores_per_column)]
        else:
            return [max(current_time_step_values) for current_time_step_values in zip(*scores_per_column)]
        

    def predict_one(self, new_sample: pd.Series, source: str, is_event: bool) -> float:
         # TODO need to keep buffer until profile size are encountered and then start predicting
        return 0.0
    

    def get_library(self) -> str:
        return 'no_save'
    

    def __str__(self) -> str:
        return 'ForecastingAnomalyPrediction'
    

    def get_params(self) -> dict:
        return {
            'num_samples': self.num_samples,
            'context_length': self.context_length,
            'prediction_length': self.prediction_length,
            'forecasting_model': self.forecasting_model,
            'device_type': self.device_type,
            'torch_dtype': self.torch_dtype,
            'aggregation_strategy': self.aggregation_strategy,
            'anomaly_detector': str(self.anomaly_detector(self.event_preferences, **self.anomaly_detector_params)),
            **{'anomaly_detector_' + k: v for k, v in self.anomaly_detector_params.items()}
        }
    

    def get_all_models(self):
        pass