import os
import argparse
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import uuid

from method.semi_supervised_method import SemiSupervisedMethodInterface
from pdm_evaluation_types.types import EventPreferences
from method.timemixerpp.TimeMixer.exp.exp_basic import Exp_Basic 
from method.timemixerpp.TimeMixer.exp.exp_long_term_forecasting  import Exp_Long_Term_Forecast


class TimeMixerPP(SemiSupervisedMethodInterface):
    def __init__(
            self,
            event_preferences: EventPreferences,
            seq_len: int,
            moving_avg: int,
            train_epochs: int,
            task_name: str = 'long_term_forecast',
            features: str = 'M',
            label_len: int = 0,
            pred_len: int = 1,
            down_sampling_layers: int = 2,
            down_sampling_window: int = 2,
            channel_independence: int = 0,
            e_layers: int = 2,
            use_future_temporal_feature: int = 0,
            d_model: int = 16,
            embed: str = 'timeF',
            freq: str = 'h',
            dropout: float = 0.1,
            use_norm: int = 1,
            down_sampling_method: str = 'avg',
            decomp_method: str = 'moving_avg',
            d_ff: int = 32,
            itr: int = 1,
            learning_rate: float = 0.001,
            patience: int = 3,
            loss: str = 'MSE',
            pct_start: float = 0.2,
            lradj: str = 'TST',
            batch_size: int = 16,
            seasonal_patterns: str = 'Monthly',
            num_workers: int = 10,
            n_vali: float = 0.1,
            use_amp: bool = False,
            inverse: bool = False,
            output_attention: bool = False,
            forecasting_distance: str = 'euclidean',
            model: str = 'TimeMixer',
            use_gpu: bool = True,
            gpu: int = 1,
            use_multi_gpu: bool = False,
            devices: str = '0,1',
            checkpoints: str = './checkpoints_timemixerpp',
            *args,
            **kwargs
        ):
        super().__init__(event_preferences=event_preferences)
        self.task_name = task_name
        self.features = features
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.down_sampling_layers = down_sampling_layers
        self.down_sampling_window = down_sampling_window
        self.channel_independence = channel_independence
        self.e_layers = e_layers
        self.moving_avg = moving_avg
        self.use_future_temporal_feature = use_future_temporal_feature
        self.d_model = d_model
        self.embed = embed
        self.freq = freq
        self.dropout = dropout
        self.use_norm = use_norm
        self.down_sampling_method = down_sampling_method
        self.decomp_method = decomp_method
        self.d_ff = d_ff
        self.itr = itr
        self.learning_rate = learning_rate
        self.patience = patience
        self.loss = loss
        self.pct_start = pct_start
        self.train_epochs = train_epochs
        self.lradj = lradj
        self.batch_size = batch_size
        self.seasonal_patterns = seasonal_patterns
        self.num_workers = num_workers
        self.n_vali = n_vali
        self.use_amp = use_amp
        self.inverse = inverse
        self.output_attention = output_attention
        self.forecasting_distance = forecasting_distance
        self.model = model

        self.use_gpu = use_gpu
        self.gpu = gpu
        self.use_multi_gpu = use_multi_gpu
        self.devices = devices

        self.use_gpu = True if torch.cuda.is_available() and self.use_gpu else False
        
        if self.use_gpu and self.use_multi_gpu:
            self.devices = self.devices.replace(' ', '')
            device_ids = self.devices.split(',')
            self.device_ids = [int(id) for id in device_ids]
            self.gpu = self.device_ids[0]

        self.experiment_per_source = {}

        self.checkpoints = f'{checkpoints}/{uuid.uuid4()}/'
        Path(self.checkpoints).mkdir(parents=True, exist_ok=True)


    def fit(self, historic_data: list[pd.DataFrame], historic_sources: list[str], event_data: pd.DataFrame) -> None:
        for current_historic_data, current_historic_source in zip(historic_data, historic_sources):
            while math.floor((1 - self.n_vali) * len(current_historic_data)) < self.seq_len + self.pred_len + 1 + self.batch_size:
                point_to_pre_append = current_historic_data.iloc[0]
                current_historic_data = pd.concat([
                    pd.DataFrame([point_to_pre_append], columns=current_historic_data.columns),
                    current_historic_data
                ])


            while math.floor(self.n_vali * len(current_historic_data)) < self.seq_len + self.pred_len + 1 + self.batch_size:
                point_to_pre_append = current_historic_data.iloc[0]
                current_historic_data = pd.concat([
                    pd.DataFrame([point_to_pre_append], columns=current_historic_data.columns),
                    current_historic_data
                ])
            
            self.experiment_per_source[current_historic_source] = Exp_Long_Term_Forecast(argparse.Namespace(**{
                    'task_name': self.task_name,
                    'features': self.features,
                    'seq_len': self.seq_len,
                    'label_len': self.label_len,
                    'pred_len': self.pred_len,
                    'down_sampling_layers': self.down_sampling_layers,
                    'down_sampling_window': self.down_sampling_window,
                    'channel_independence': self.channel_independence,
                    'e_layers': self.e_layers,
                    'moving_avg': self.moving_avg,
                    'enc_in': current_historic_data.shape[1],
                    'c_out': current_historic_data.shape[1],
                    'use_future_temporal_feature': self.use_future_temporal_feature,
                    'd_model': self.d_model,
                    'embed': self.embed,
                    'freq': self.freq,   
                    'dropout': self.dropout, 
                    'use_norm': self.use_norm,
                    'down_sampling_method': self.down_sampling_method,
                    'decomp_method': self.decomp_method,
                    'd_ff': self.d_ff,
                    'itr': self.itr,
                    'learning_rate': self.learning_rate,
                    'checkpoints': self.checkpoints,
                    'patience': self.patience,
                    'loss': self.loss,
                    'pct_start': self.pct_start,
                    'train_epochs': self.train_epochs,
                    'lradj': self.lradj,
                    'data': 'custom',
                    'batch_size': self.batch_size,
                    'target': current_historic_data.columns[-1],
                    'seasonal_patterns': self.seasonal_patterns,
                    'num_workers': self.num_workers,
                    'use_amp': self.use_amp,
                    'inverse': self.inverse,
                    'output_attention': self.output_attention,
                    'forecasting_distance': self.forecasting_distance,
                    'model': self.model,
                    'use_gpu': self.use_gpu,
                    'gpu': self.gpu,
                    'use_multi_gpu': self.use_multi_gpu,
                    'devices': self.devices
                }), current_historic_data.copy(), self.n_vali)
            
            self.experiment_per_source[current_historic_source].train(f'{current_historic_source}')

    def predict(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> list[float]:
        if len(target_data) < self.seq_len + self.pred_len + 1:
            return [0 for i in range(len(target_data))]

        test_experiment = Exp_Long_Term_Forecast(argparse.Namespace(**{
            'task_name': self.task_name,
            'features': self.features,
            'seq_len': self.seq_len,
            'label_len': self.label_len,
            'pred_len': self.pred_len,
            'down_sampling_layers': self.down_sampling_layers,
            'down_sampling_window': self.down_sampling_window,
            'channel_independence': self.channel_independence,
            'e_layers': self.e_layers,
            'moving_avg': self.moving_avg,
            'enc_in': target_data.shape[1],
            'c_out': target_data.shape[1],
            'use_future_temporal_feature': self.use_future_temporal_feature,
            'd_model': self.d_model,
            'embed': self.embed,
            'freq': self.freq,
            'dropout': self.dropout,
            'use_norm': self.use_norm,
            'down_sampling_method': self.down_sampling_method,
            'decomp_method': self.decomp_method,
            'd_ff': self.d_ff,
            'itr': self.itr,
            'learning_rate': self.learning_rate,
            'checkpoints': self.checkpoints,
            'patience': self.patience,
            'loss': self.loss,
            'pct_start': self.pct_start,
            'train_epochs': self.train_epochs,
            'lradj': self.lradj,
            'data': 'custom',
            'batch_size': self.batch_size,
            'target': target_data.columns[-1],
            'seasonal_patterns': self.seasonal_patterns,
            'num_workers': self.num_workers,
            'use_amp': self.use_amp,
            'inverse': self.inverse,
            'output_attention': self.output_attention,
            'forecasting_distance': self.forecasting_distance,
            'model': self.model,
            'use_gpu': self.use_gpu,
            'gpu': self.gpu,
            'use_multi_gpu': self.use_multi_gpu,
            'devices': self.devices
        }), target_data.copy(), self.n_vali)

        scores = test_experiment.test(f'{source}', test=1)

        torch.cuda.empty_cache()

        return [scores[0] for i in range(len(target_data) - len(scores))] + scores


    def predict_one(self, new_sample: pd.Series, source: str, is_event: bool) -> float:
        return 0 # TODO we need to keep the last seq_len points for a source and then start predicting, when source changes keep another buffer


    def get_library(self) -> str:
        return 'no_save'


    def get_params(self) -> dict:
        return {
            'task_name': self.task_name,
            'features': self.features,
            'seq_len': self.seq_len,
            'label_len': self.label_len,
            'pred_len': self.pred_len,
            'down_sampling_layers': self.down_sampling_layers,
            'down_sampling_window': self.down_sampling_window,
            'channel_independence': self.channel_independence,
            'e_layers': self.e_layers,
            'moving_avg': self.moving_avg,
            'use_future_temporal_feature': self.use_future_temporal_feature,
            'd_model': self.d_model,
            'embed': self.embed,
            'freq': self.freq,
            'dropout': self.dropout,
            'use_norm': self.use_norm,
            'down_sampling_method': self.down_sampling_method,
            'decomp_method': self.decomp_method,
            'd_ff': self.d_ff,
            'itr': self.itr,
            'learning_rate': self.learning_rate,
            'patience': self.patience,
            'loss': self.loss,
            'pct_start': self.pct_start,
            'train_epochs': self.train_epochs,
            'lradj': self.lradj,
            'batch_size': self.batch_size,
            'seasonal_patterns': self.seasonal_patterns,
            'num_workers': self.num_workers,
            'n_vali': self.n_vali,
            'use_amp': self.use_amp,
            'inverse': self.inverse,
            'output_attention': self.output_attention,
            'forecasting_distance': self.forecasting_distance,
            'model': self.model,
            'use_gpu': self.use_gpu,
            'gpu': self.gpu,
            'use_multi_gpu': self.use_multi_gpu,
            'devices': self.devices,
            'checkpoints': self.checkpoints
        }


    def __str__(self) -> str:
        return 'TIMEMIXERPP'


    def get_all_models(self):
        pass


    def destruct(self):
        for root, dirs, files in os.walk(self.checkpoints):
            for file in files:
                file_path = os.path.join(root, file)
                
                os.remove(file_path)
        
        for root, dirs, files in os.walk(self.checkpoints, topdown=False):
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                
                os.rmdir(dir_path)

        if os.path.exists(self.checkpoints):
            os.rmdir(self.checkpoints)
