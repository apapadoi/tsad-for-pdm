import pandas as pd
import numpy as np
import mlflow

from exceptions.exception import NotFitForSourceException
from method.semi_supervised_method import SemiSupervisedMethodInterface
from pdm_evaluation_types.types import EventPreferences

import pypots
from pypots.forecasting import TimeLLM
from pypots.optim import Adam
from pypots.nn.functional import calc_mae


class TimeLLMPyPots(SemiSupervisedMethodInterface):
    def __init__(self, 
                 event_preferences: EventPreferences,
                 llm_model_type: str,
                 n_layers: int,
                 patch_size: int,
                 patch_stride: int,
                 d_llm: int,
                 d_model: int,
                 d_ffn: int,
                 n_heads: int,
                 domain_prompt_content: str,
                 batch_size: int,
                 epochs: int,

                 n_steps: int, 

                 term: str = 'short', # 'long',
                 dropout: float = 0.1,
                 patience: int = 3,
                 optimizer= Adam(lr=1e-3),
                 num_workers: int = 0,
                 device: str = 'cuda:1',
                 verbose: bool = True,

                 forecasting_distance: str = 'euclidean',
                 n_pred_steps: int = 1, 
                 *args, 
                 **kwargs):
        super().__init__(event_preferences=event_preferences)
        self.initial_args = args
        self.initial_kwargs = kwargs

        if 'profile_size' in kwargs:
            del self.initial_kwargs['profile_size']
        
        self.n_steps = n_steps
        self.epochs = epochs
        self.domain_prompt_content = domain_prompt_content
        self.forecasting_distance = forecasting_distance
        self.llm_model_type = llm_model_type
        self.n_layers = n_layers
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.d_llm = d_llm
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.n_heads = n_heads
        self.batch_size = batch_size

        self.term = term
        self.dropout = dropout
        self.patience = patience
        self.optimizer = optimizer
        self.num_workers = num_workers
        self.device = device
        self.verbose = verbose


        self.clf_class = TimeLLM
        self.model_per_source = {}

        self.historic_data_per_source = {}
        self.N_PRED_STEPS = n_pred_steps


    def fit(self, historic_data: list[pd.DataFrame], historic_sources: list[str], event_data: pd.DataFrame) -> None:
        for current_historic_data, current_historic_source in zip(historic_data, historic_sources):
            # Transform DataFrame to 3D numpy array for forecasting
            # Shape: (num_samples, n_steps, n_features)
            transformed_data = self._transform_to_forecasting_format(current_historic_data)
            self.historic_data_per_source[current_historic_source] = transformed_data


    def predict(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> list[float]:
        if source not in self.historic_data_per_source.keys():
            raise NotFitForSourceException()
        
        print(f'Predicting for source: {source}')
        current_historic_data = self.historic_data_per_source[source]
        initial_target_data = target_data.copy()
        print(f'initial_target_data shape: {initial_target_data.shape}')

        dataset_for_training = {
            'X': current_historic_data[:, :-self.N_PRED_STEPS],
            'X_pred': current_historic_data[:, -self.N_PRED_STEPS:],
        }

        for i in range(self.n_steps - 1):
            point_to_pre_append = target_data.iloc[0]
            target_data = pd.concat([
                pd.DataFrame([point_to_pre_append], columns=target_data.columns),
                target_data
            ])

        transformed_target_data = self._transform_to_forecasting_format(target_data)

        dataset_for_testing = {
            'X': transformed_target_data[:, :-self.N_PRED_STEPS],
        }
    
        timellm = TimeLLM(
            n_steps = self.n_steps - self.N_PRED_STEPS,
            n_features = target_data.shape[1],
            n_pred_steps = self.N_PRED_STEPS,
            n_pred_features = target_data.shape[1],
            term = self.term, 
            llm_model_type = self.llm_model_type,
            n_layers = self.n_layers,
            patch_size = self.patch_size,
            patch_stride = self.patch_stride,
            d_llm = self.d_llm,
            d_model = self.d_model,
            d_ffn = self.d_ffn,
            n_heads = self.n_heads,
            dropout = self.dropout,
            domain_prompt_content = self.domain_prompt_content,
            # n_fod = 2,
            batch_size=self.batch_size,
            # here we set epochs=10 for a quick demo, you can set it to 100 or more for better performance
            epochs=self.epochs,
            # here we set patience=3 to early stop the training if the evaluting loss doesn't decrease for 3 epoches.
            # You can leave it to defualt as None to disable early stopping.
            patience=self.patience,
            # give the optimizer. Different from torch.optim.Optimizer, you don't have to specify model's parameters when
            # initializing pypots.optim.Optimizer. You can also leave it to default. It will initilize an Adam optimizer with lr=0.001.
            optimizer=self.optimizer,
            # this num_workers argument is for torch.utils.data.Dataloader. It's the number of subprocesses to use for data loading.
            # Leaving it to default as 0 means data loading will be in the main process, i.e. there won't be subprocesses.
            # You can increase it to >1 if you think your dataloading is a bottleneck to your model training speed
            num_workers=self.num_workers,
            # just leave it to default as None, PyPOTS will automatically assign the best device for you.
            # Set it as 'cpu' if you don't have CUDA devices. You can also set it to 'cuda:0' or 'cuda:1' if you have multiple CUDA devices, even parallelly on ['cuda:0', 'cuda:1']
            device=self.device,
            # set the path for saving tensorboard and trained model files
            #saving_path="tutorial_results/forecasting/timellm",
            # only save the best model after training finished.
            # You can also set it as "better" to save models performing better ever during training.
            #model_saving_strategy="best",
            verbose=self.verbose
        )

        timellm.fit(train_set=dataset_for_training)

        timellm_results = timellm.predict(dataset_for_testing)
        timellm_prediction = timellm_results["forecasting"]
        print(f'timellm_prediction shape: {timellm_prediction.shape}')
        # Reshape from (num_samples, 1, n_features) to (num_samples, n_features)
        # Since N_PRED_STEPS = 1, we squeeze the middle dimension
        timellm_prediction_reshaped = timellm_prediction.squeeze(axis=1)
        print(f'timellm_prediction_reshaped shape: {timellm_prediction_reshaped.shape}')

        # Calculate Euclidean distance between each row in initial_target_data and forecasted row
        scores = []
        if self.forecasting_distance == 'euclidean':
            # Calculate Euclidean distance for each row in initial_target_data
            for i in range(initial_target_data.shape[0]):
                # Get the corresponding forecasted row (accounting for the sliding window)
                forecasted_row = timellm_prediction_reshaped[i]
                actual_row = initial_target_data.iloc[i].values
                # Calculate Euclidean distance
                distance = np.linalg.norm(actual_row - forecasted_row)
                scores.append(distance)
        else:
            raise RuntimeError('Other forecasting distance options are not implemented yet')
        
        print(f'scores length: {len(scores)}')

        return scores


    def predict_one(self, new_sample: pd.Series, source: str, is_event: bool) -> float:
        return 0.0


    def get_library(self) -> str:
        return 'no_save'
    

    def __str__(self) -> str:
        return 'TimeLLMPyPots'


    def get_params(self) -> dict:
        return {
            'n_steps': self.n_steps,
            'epochs': self.epochs,
            'domain_prompt_content': self.domain_prompt_content,
            'forecasting_distance': self.forecasting_distance,
            'n_pred_steps': self.N_PRED_STEPS,
            'd_model': self.d_model,
            'd_ffn': self.d_ffn,
            'n_heads': self.n_heads,
            'dropout': self.dropout,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'patience': self.patience,
            'optimizer': self.optimizer,
            'num_workers': self.num_workers,
            'device': self.device,
            'verbose': self.verbose,
            'llm_model_type': self.llm_model_type,
            'n_layers': self.n_layers,
            'patch_size': self.patch_size,
            'patch_stride': self.patch_stride,
            'd_llm': self.d_llm,
            'term': self.term
        }


    def get_all_models(self):
        pass


    def _transform_to_forecasting_format(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform DataFrame to 3D numpy array for forecasting.
        
        Args:
            df: DataFrame with shape (num_records, num_features)
                Each record represents a time step
        
        Returns:
            3D numpy array with shape (num_samples, n_steps, n_features)
        """
        if self.n_steps is None:
            raise ValueError("n_steps must be specified in kwargs during initialization")
        
        # Get number of features
        n_features = df.shape[1]
        
        # Calculate number of samples we can create
        # Each sample will have n_steps time steps
        total_records = len(df)
        num_samples = total_records - self.n_steps + 1
        
        if num_samples <= 0:
            raise ValueError(f"Not enough data: have {total_records} records, need at least {self.n_steps}")
        
        # Initialize 3D array: (num_samples, n_steps, n_features)
        transformed_data = np.zeros((num_samples, self.n_steps, n_features))
        
        # Create sliding windows
        for i in range(num_samples):
            transformed_data[i] = df.iloc[i:i + self.n_steps].values
        
        return transformed_data
    