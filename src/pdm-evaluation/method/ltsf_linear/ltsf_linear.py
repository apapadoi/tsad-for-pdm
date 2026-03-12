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
from method.ltsf_linear.LTSF_Linear.models import Linear, DLinear, NLinear
from method.ltsf_linear.LTSF_Linear.data_provider.data_factory import data_provider
from method.ltsf_linear.LTSF_Linear.utils.tools import EarlyStopping, adjust_learning_rate 

model_dict = {
    'Linear': Linear,
    'DLinear': DLinear,
    'NLinear': NLinear,
}


class LTSFLinear(SemiSupervisedMethodInterface):
    def __init__(
            self, 
            event_preferences: EventPreferences,
            n_vali: float = 0.1, # percentage of points to take for validation from profile,
            ltsf_type: str = 'DLinear', # options: [Linear, DLinear, NLinear]
            features: str = 'M', # options: [M, S, MS]
            target: str = 'OT', # target feature when using S or MS options in 'features' parameter
            freq: str = 'h', # freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h'
            seq_len: int = 10, # input sequence length
            label_len: int = 1, # start token length
            pred_len: int = 1, # prediction sequence length
            individual: bool = False, # DLinear: a linear layer for each variate(channel) individually
            embed: str = 'timeF', # time features encoding, options:[timeF, fixed, learned]
            num_workers: int = 10, # data loader num workers
            train_epochs: int = 10, # train epochs
            batch_size: int = 16, # batch size of train input data - NOTE: should be < n_train * len(target_data) and < n_vali * len(target_data)
            patience: int = 3, # early stopping patience
            learning_rate: int = 0.0001, # optimizer learning rate
            lr_adjustment: str = 'type1', # adjust learning rate
            use_amp: bool = False, # use automatic mixed precision training
            use_gpu: bool = True,
            gpu: int = 0,
            use_multi_gpu: bool = False,
            devices: str = '0,1',
            forecasting_distance: str = 'euclidean',
            path: str = './checkpoints_ltsf',
            *args, 
            **kwargs
    ):
        super().__init__(event_preferences=event_preferences)
        
        self.n_vali = n_vali
        self.ltsf_type = ltsf_type
        self.features = features
        self.target = target
        self.freq = freq
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.individual = individual
        self.embed = embed
        self.num_workers = num_workers
        self.train_epochs = train_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.learning_rate = learning_rate
        self.lr_adjustment = lr_adjustment
        self.use_amp = use_amp
        self.gpu = gpu
        self.use_gpu = use_gpu
        self.use_multi_gpu = use_multi_gpu
        self.devices = devices
        self.forecasting_distance = forecasting_distance

        self.model_per_source = {}
        
        self.use_gpu = True if torch.cuda.is_available() and self.use_gpu else False
        
        if self.use_gpu and self.use_multi_gpu:
            self.devices = self.devices.replace(' ', '')
            device_ids = self.devices.split(',')
            self.device_ids = [int(id) for id in device_ids]
            self.gpu = self.device_ids[0]


        if self.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu) if not self.use_multi_gpu else self.devices
            device = torch.device('cuda:{}'.format(self.gpu))
        else:
            device = torch.device('cpu')
        self.device = device

        self.path = f'{path}/{uuid.uuid4()}'
        Path(self.path).mkdir(parents=True, exist_ok=True)


    def fit(self, historic_data: list[pd.DataFrame], historic_sources: list[str], event_data: pd.DataFrame) -> None:
        for current_historic_data, current_historic_source in zip(historic_data, historic_sources):
            self.model_per_source[current_historic_source] = model_dict[self.ltsf_type].Model(argparse.Namespace(**{
                'seq_len': self.seq_len,
                'pred_len': self.pred_len,
                'enc_in': current_historic_data.shape[1], # auto detect enc_in parameter - timestamp column is dropped from the experiment subclasses
                'individual': self.individual,
            })).float()

            if self.use_multi_gpu and self.use_gpu:
                self.model_per_source[current_historic_source] = torch.nn.DataParallel(self.model_per_source[current_historic_source], device_ids=self.device_ids)
            
            self.model_per_source[current_historic_source] = self.model_per_source[current_historic_source].to(self.device)

            current_model = self.model_per_source[current_historic_source]

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

            _, train_data_loader = data_provider(argparse.Namespace(**{
                'embed': self.embed,
                'batch_size': min(self.batch_size, current_historic_data.shape[0]),
                'freq': self.freq,
                'seq_len': self.seq_len,
                'label_len': self.label_len,
                'pred_len': self.pred_len,
                'features': self.features,
                'target': self.target,
                'num_workers': self.num_workers,
            }), 'train', current_historic_data.copy(), self.n_vali)

            _, vali_data_loader = data_provider(argparse.Namespace(**{
                'embed': self.embed,
                'batch_size': min(self.batch_size, int(current_historic_data.shape[0] * self.n_vali)),
                'freq': self.freq,
                'seq_len': self.seq_len,
                'label_len': self.label_len,
                'pred_len': self.pred_len,
                'features': self.features,
                'target': self.target,
                'num_workers': self.num_workers,
            }), 'val', current_historic_data.copy(), self.n_vali)
            
            early_stopping = EarlyStopping(patience=self.patience, verbose=False, source=current_historic_source)

            model_optim = optim.Adam(current_model.parameters(), lr=self.learning_rate)

            criterion = nn.MSELoss()

            if self.use_amp:
                scaler = torch.cuda.amp.GradScaler()

            for epoch in range(self.train_epochs):
                print(f'epoch: {epoch}')
                iter_count = 0
                train_loss = []

                current_model.train()
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_data_loader):
                    iter_count += 1
                    model_optim.zero_grad()
                    batch_x = batch_x.float().to(self.device)

                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.label_len, :], dec_inp], dim=1).float().to(self.device)

                    # encoder - decoder
                    if self.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = current_model(batch_x)

                            f_dim = -1 if self.features == 'MS' else 0
                            outputs = outputs[:, -self.pred_len:, f_dim:]
                            batch_y = batch_y[:, -self.pred_len:, f_dim:].to(self.device)
                            loss = criterion(outputs, batch_y)
                            train_loss.append(loss.item())
                    else:
                        outputs = current_model(batch_x)

                        f_dim = -1 if self.features == 'MS' else 0
                        outputs = outputs[:, -self.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())


                    if self.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        loss.backward()
                        model_optim.step()

                train_loss = np.average(train_loss)

                vali_loss = self._ltsf_predict(vali_data_loader, criterion, current_model)

                early_stopping(vali_loss, current_model, self.path)

                if early_stopping.early_stop:
                    break

                adjust_learning_rate(model_optim, epoch + 1, argparse.Namespace(**{
                    'lradj': self.lr_adjustment,
                    'learning_rate': self.learning_rate,
                }))
            
            torch.cuda.empty_cache()


    def predict(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> list[float]:
        if len(target_data) < self.seq_len + self.pred_len + 1:
            return [0 for i in range(len(target_data))]
        

        # TODO need to check if a model is available for the provided source    
        best_model_path = self.path + '/' + f'checkpoint_{source}.pth'
        current_model = self.model_per_source[source]
        current_model.load_state_dict(torch.load(best_model_path))

        scores = self._ltsf_test(source, target_data, current_model)

        torch.cuda.empty_cache()

        return [scores[0] for i in range(len(target_data) - len(scores))] + scores


    def predict_one(self, new_sample: pd.Series, source: str, is_event: bool) -> float:
        return 0 # TODO we need to keep the last seq_len points for a source and then start predicting, when source changes keep another buffer


    def get_library(self) -> str:
        return 'no_save'


    def get_params(self) -> dict:
        return {
            'n_vali': self.n_vali,
            'ltsf_type': self.ltsf_type,
            'features': self.features,
            'target': self.target,
            'freq': self.freq,
            'seq_len': self.seq_len,
            'label_len': self.label_len,
            'pred_len': self.pred_len,
            'individual': self.individual,
            'embed': self.embed,
            'num_workers': self.num_workers,
            'train_epochs': self.train_epochs,
            'batch_size': self.batch_size,
            'patience': self.patience,
            'learning_rate': self.learning_rate,
            'lr_adjustment': self.lr_adjustment,
            'use_amp': self.use_amp,
            'gpu': self.gpu,
            'use_gpu': self.use_gpu,
            'use_multi_gpu': self.use_multi_gpu,
            'devices': self.devices,
            'forecasting_distance': self.forecasting_distance,
            'path': self.path,
            'device': self.device,
        }

    
    def __str__(self) -> str:
        return 'LTSF'


    def get_all_models(self):
        return list(self.model_per_source.keys()), list(self.model_per_source.values())


    def _ltsf_predict(self, current_data_loader, criterion, current_model):
        total_loss = []
        current_model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(current_data_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = current_model(batch_x)
                else:
                    outputs = current_model(batch_x)

                f_dim = -1 if self.features == 'MS' else 0
                outputs = outputs[:, -self.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)

        total_loss = np.average(total_loss)
        current_model.train()

        return total_loss
    

    def _ltsf_test(self, current_source, target_data, current_model):
        target_data = target_data.copy()
        
        _, test_data_loader = data_provider(argparse.Namespace(**{
            'embed': self.embed,
            'batch_size': min(self.batch_size, target_data.shape[0]),
            'freq': self.freq,
            'seq_len': self.seq_len,
            'label_len': self.label_len,
            'pred_len': self.pred_len,
            'features': self.features,
            'target': self.target,
            'num_workers': self.num_workers,
        }), 'test', target_data, self.n_vali)

        preds = []
        trues = []

        current_model.eval()
        with torch.no_grad():
             for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_data_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = current_model(batch_x)
                else:
                    outputs = current_model(batch_x)

                f_dim = -1 if self.features == 'MS' else 0

                outputs = outputs[:, -self.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)


        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        preds_reshaped = preds.reshape(preds.shape[0], -1)
        trues_reshaped = trues.reshape(trues.shape[0], -1)

        if self.forecasting_distance == 'euclidean':
            return np.linalg.norm(trues_reshaped - preds_reshaped, axis=1).tolist()
        else:
            raise RuntimeError('Other forecasting distance options are not implemented yet')


    def destruct(self):
        for root, dirs, files in os.walk(self.path):
            for file in files:
                file_path = os.path.join(root, file)
                
                os.remove(file_path)
        
        for root, dirs, files in os.walk(self.path, topdown=False):
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                
                os.rmdir(dir_path)

        if os.path.exists(self.path):
            os.rmdir(self.path)