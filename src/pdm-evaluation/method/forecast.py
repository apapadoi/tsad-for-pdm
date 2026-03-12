import re

from sklearn.ensemble import IsolationForest as isolation_forest
import pandas as pd
import numpy as np
import mlflow
from tqdm import tqdm
import math
import subprocess
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from momentfm import MOMENTPipeline
from momentfm.data.anomaly_detection_dataset import AnomalyDetectionDataset
from momentfm.utils.masking import Masking
from momentfm.utils.forecasting_metrics import get_forecasting_metrics

from method.semi_supervised_method import SemiSupervisedMethodInterface
from exceptions.exception import NotFitForSourceException
from pdm_evaluation_types.types import EventPreferences
from method.unsupervised_method import UnsupervisedMethodInterface
from utils.utils import EarlyStoppingTorch
from utils.informer_dataset import InformerDataset

class ForecastingAnomalyPredictionMethod(SemiSupervisedMethodInterface):
    def __init__(self, 
                 event_preferences: EventPreferences, 
                 anomaly_detector: UnsupervisedMethodInterface,
                 # seq_len,
                #  input_c=1, 
                 batch_size=512,
                 epochs=100,
                #  validation_size=0,
                #  lr=1e-4,
                 forecast_horizon=1,
                #  forecasting_model: str = "amazon/chronos-t5-small", 
                 device_type: str = "cuda:1",
                #  torch_dtype: torch.Tensor = torch.bfloat16,
                vali_points=514,
                 aggregation_strategy: str = 'avg',
                 random_seed: int = 42,
                 *args, **kwargs):
        super().__init__(event_preferences=event_preferences)
        self.initial_args = args
        self.initial_kwargs = kwargs

        # self.seq_len = seq_len
        # self.input_c = input_c
        self.batch_size = batch_size
        # self.anomaly_criterion = nn.MSELoss(reduce=False)
        self.epochs = epochs
        # self.validation_size = validation_size
        # self.lr = lr
        self.forecast_horizon = forecast_horizon
        
        cuda = True        
        self.cuda = cuda
        if self.cuda == True and torch.cuda.is_available():
            self.device = torch.device(device_type)
            print("----- Using GPU -----")
        else:
            if self.cuda == True and not torch.cuda.is_available():
                print("----- GPU is unavailable -----")
            self.device = torch.device("cpu")
            print("----- Using CPU -----")

        self.model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large", 
            model_kwargs={
                "task_name": "forecasting",
                "forecast_horizon": self.forecast_horizon,
                'head_dropout': 0.1,
                'weight_decay': 0,
                'freeze_encoder': True,  # Freeze the patch embedding layer
                'freeze_embedder': True,  # Freeze the transformer encoder
                'freeze_head': False,  # The linear forecasting head must be trained
            }
        )

        self.device_type = device_type
        # self.torch_dtype = torch_dtype
        self.aggregation_strategy = aggregation_strategy
        self.random_seed = random_seed
        self.vali_points = vali_points

        self.model.init()
        # self.model = self.model.to(self.device_type).float()
        # Optimize Mean Squarred Error using your favourite optimizer
        # self.criterion = torch.nn.MSELoss() 
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.75)
        self.save_path = None
        self.early_stopping = EarlyStoppingTorch(save_path=self.save_path, patience=3)
        
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
        whole_scenario = pd.concat([
            self.historic_data_per_source[source],
            target_data
        ])
        
        self.early_stopping = EarlyStoppingTorch(save_path=self.save_path, patience=3)

        while whole_scenario.shape[0] < 1024:
            point_to_pre_append = whole_scenario.iloc[0]
            whole_scenario = pd.concat([
                pd.DataFrame([point_to_pre_append], columns=whole_scenario.columns),
                whole_scenario
            ])

        model = self.model

        train_dataset = InformerDataset(whole_scenario, vali_points=self.vali_points, train_points=self.historic_data_per_source[source].shape[0], data_split="train", random_seed=self.random_seed, forecast_horizon=self.forecast_horizon)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_dataset = InformerDataset(whole_scenario, vali_points=self.vali_points, train_points=self.historic_data_per_source[source].shape[0], data_split="val", random_seed=self.random_seed, forecast_horizon=self.forecast_horizon)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)

        test_dataset = InformerDataset(whole_scenario, vali_points=self.vali_points, train_points=self.historic_data_per_source[source].shape[0], data_split="test", random_seed=self.random_seed, forecast_horizon=self.forecast_horizon)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        cur_epoch = 0
        max_epoch = self.epochs

        model = model.to(self.device_type)

        criterion = criterion.to(self.device_type)

        scaler = torch.cuda.amp.GradScaler()

        max_lr = 1e-4
        total_steps = len(train_loader) * max_epoch
        scheduler = OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps, pct_start=0.3)

        max_norm = 5.0

        while cur_epoch < max_epoch:
            losses = []
            model.train()

            for timeseries, forecast, input_mask in tqdm(train_loader, total=len(train_loader)):
                timeseries = timeseries.float().to(self.device_type)
                input_mask = input_mask.to(self.device_type)
                forecast = forecast.float().to(self.device_type)

                with torch.cuda.amp.autocast():
                    output = model(timeseries, input_mask)
                
                loss = criterion(output.forecast, forecast)

                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                losses.append(loss.item())


            losses = np.array(losses)
            average_loss = np.average(losses)
            print(f"Epoch {cur_epoch}: Train loss: {average_loss:.3f}")

            scheduler.step()

            model.eval()
            val_losses = []
            with torch.no_grad():
                for timeseries, forecast, input_mask in tqdm(val_loader, total=len(val_loader)):
                    timeseries = timeseries.float().to(self.device_type)
                    input_mask = input_mask.to(self.device_type)
                    forecast = forecast.float().to(self.device_type)

                    with torch.cuda.amp.autocast():
                        output = model(timeseries, input_mask)
                    
                    loss = criterion(output.forecast, forecast)                
                    val_losses.append(loss.item())

            val_losses = np.array(val_losses)
            average_loss = np.average(val_losses)

            self.early_stopping(average_loss, self.model)
            if self.early_stopping.early_stop:
                print("   Early stopping<<<")
                break
        
            cur_epoch += 1


        trues, preds, histories, losses = [], [], [], []
        model.eval()
        with torch.no_grad():
            for timeseries, forecast, input_mask in tqdm(test_loader, total=len(test_loader)):
                timeseries = timeseries.float().to(self.device_type)
                input_mask = input_mask.to(self.device_type)
                forecast = forecast.float().to(self.device_type)

                with torch.cuda.amp.autocast():
                    output = model(timeseries, input_mask)
                
                loss = criterion(output.forecast, forecast)                
                losses.append(loss.item())

                trues.append(forecast.detach().cpu().numpy())
                preds.append(output.forecast.detach().cpu().numpy())
                histories.append(timeseries.detach().cpu().numpy())
        
        losses = np.array(losses)
        average_loss = np.average(losses)
        model.train()

        trues = np.concatenate(trues, axis=0)
        preds = np.concatenate(preds, axis=0)
        histories = np.concatenate(histories, axis=0)
        
        metrics = get_forecasting_metrics(y=trues, y_hat=preds, reduction='mean')
        # TODO need to load best model
        print(f"Test MSE: {metrics.mse:.3f} | Test MAE: {metrics.mae:.3f}")

        current_detector = self.anomaly_detector_per_source[source]
        # TODO need handling for unsupervised anomaly detector
        scores = current_detector.predict(
            pd.DataFrame(preds.reshape(preds.shape[0], -1), columns=target_data.columns),
            source,
            event_data
        )

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        return scores


    def predict_one(self, new_sample: pd.Series, source: str, is_event: bool) -> float:
         # TODO need to keep buffer until profile size are encountered and then start predicting
        return 0.0
    

    def get_library(self) -> str:
        return 'no_save'
    

    def __str__(self) -> str:
        return 'ForecastingAnomalyPrediction'
    

    def get_params(self) -> dict:
        return {
            # 'num_samples': self.num_samples,
            # 'context_length': self.context_length,
            # 'prediction_length': self.prediction_length,
            # 'forecasting_model': self.forecasting_model,
            # 'device_type': self.device_type,
            # 'torch_dtype': self.torch_dtype,
            # 'aggregation_strategy': self.aggregation_strategy,
            'anomaly_detector': str(self.anomaly_detector(self.event_preferences, **self.anomaly_detector_params)),
            **{'anomaly_detector_' + k: v for k, v in self.anomaly_detector_params.items()}
        }
    

    def get_all_models(self):
        pass