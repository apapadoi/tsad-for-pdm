import logging

import os
import pandas as pd
import numpy as np
import torch
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

from method.unsupervised_method import UnsupervisedMethodInterface
from pdm_evaluation_types.types import EventPreferences


class AutogluonUns(UnsupervisedMethodInterface):
    def __init__(self, 
                 event_preferences: EventPreferences, 
                 context_length: int,
                 target: str,
                 slide: int = 1, # NOTE: the implementation works only for prediction length equal to 1
                 model: str = "chronos2",
                 forecasting_distance: str = 'euclidean',
                 random_state: int = 42,
                 *args, **kwargs):
        super().__init__(event_preferences=event_preferences)

        self.context_length = context_length
        self.target = target
        self.prediction_length = slide
        self.model = model
        self.forecasting_distance = forecasting_distance
        self.random_state = random_state
        

    def predict(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> list[float]:
        print(f'source: {source}')
        for root, dirs, files in os.walk('AutogluonModels'):
            for file in files:
                file_path = os.path.join(root, file)
                
                os.remove(file_path)
                    
        for root, dirs, files in os.walk('AutogluonModels', topdown=False):
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                
                os.rmdir(dir_path)

        initial_target_data = target_data.copy()
        internal_target_data = target_data.copy()

        # Padding logic to ensure context for the first point
        for i in range(self.context_length - 1):
            point_to_pre_append = internal_target_data.iloc[0]
            internal_target_data = pd.concat([
                pd.DataFrame([point_to_pre_append], columns=internal_target_data.columns),
                internal_target_data
            ])

        internal_target_data_transformed_batches = self._transform_to_forecasting_format(internal_target_data)

        prediction_length = self.prediction_length
        
        print(f'Total number of batches: {len(internal_target_data_transformed_batches)}')
        
        # Process each batch separately
        all_forecasted_results = []
        
        for batch_idx, batch in enumerate(internal_target_data_transformed_batches):
            print(f'Processing batch {batch_idx + 1}/{len(internal_target_data_transformed_batches)}')
            
            # Prepare batch data
            batch_dataframes = []
            for index, internal_target_data_df in enumerate(batch):
                # Reset index and create timestamp column with 1-hour intervals
                start_time = pd.Timestamp("2026-01-01 00:00:00")
                internal_target_data_df['timestamp'] = pd.date_range(
                    start=start_time, 
                    periods=len(internal_target_data_df), 
                    freq='1H'
                )
                internal_target_data_df.reset_index(drop=True, inplace=True)

                # Create item_id column with source value
                internal_target_data_df['item_id'] = f'{source}_batch{batch_idx}_{index}'
                batch_dataframes.append(internal_target_data_df)

            data_for_autogluon = pd.concat(batch_dataframes)
            print(f'Batch {batch_idx + 1} shape: {data_for_autogluon.shape}')

            data = TimeSeriesDataFrame.from_data_frame(data_for_autogluon)

            train_data, test_data = data.train_test_split(prediction_length=prediction_length)

            predictor = TimeSeriesPredictor(
                prediction_length=prediction_length,
                target=target_data.columns[0],#self.target,
                eval_metric="MSE",
                quantile_levels=[0.1, 0.5, 0.9]
            ).fit(
                train_data,
                presets=self.model,
                # hyperparameters={"Chronos2": {}},
                enable_ensemble=False,
                time_limit=60,
                random_seed=self.random_state,
            )

            result = predictor.predict(test_data)
            print(f'Batch {batch_idx + 1} predictions shape: {result.shape}')
            batch_forecasted_results = result['mean'].tolist()
            all_forecasted_results.extend(batch_forecasted_results)
            
            # Clean up predictor to free memory
            del predictor
            del data
            del train_data
            del test_data
            del result
        
        print(f'Total forecasted results: {len(all_forecasted_results)}')
        forecasted_results = all_forecasted_results

        # Calculate Euclidean distance between each row in initial_target_data and forecasted row
        scores = []
        if self.forecasting_distance == 'euclidean':
            # Calculate Euclidean distance for each row in initial_target_data
            for i in range(initial_target_data.shape[0]):
                # Get the corresponding forecasted value
                forecasted_row = forecasted_results[i]
                actual_row = initial_target_data[target_data.columns[0]].iloc[i]
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
        return f'AutoGluon_{self.model}'


    def get_params(self) -> dict:
        return {
            'context_length': self.context_length,
            'target': self.target,
            'slide': self.prediction_length,
            'model': self.model,
            'forecasting_distance': self.forecasting_distance,
            'random_state': self.random_state
        }


    def get_all_models(self):
        pass


    def _transform_to_forecasting_format(self, df: pd.DataFrame, max_samples_per_batch: int = 2_000_000):
        """
        Transform DataFrame to batches for forecasting.
        
        Args:
            df: DataFrame with shape (num_records, num_features)
                Each record represents a time step
            max_samples_per_batch: Maximum total number of rows across all windows in a batch (default: 10 million)
        
        Returns:
            List of lists, where each inner list contains DataFrames forming a batch.
            The sum of rows across all DataFrames in a batch does not exceed max_samples_per_batch.
        """
        if self.context_length is None:
            raise ValueError("context_length must be specified in kwargs during initialization")
        
        # Get number of features
        n_features = df.shape[1]
        
        # Calculate number of samples we can create
        # Each sample will have context_length time steps
        total_records = len(df)
        num_samples = total_records - self.context_length + 1
        
        if num_samples <= 0:
            raise ValueError(f"Not enough data: have {total_records} records, need at least {self.context_length}")
        
        # Create batches of sliding windows
        batches = []
        current_batch = []
        current_batch_total_rows = 0
        
        # Create sliding windows and organize into batches
        for i in range(num_samples):
            window = df.iloc[i:i + self.context_length]
            window_rows = len(window)  # This will be context_length
            
            # Check if adding this window would exceed the max total rows per batch
            if current_batch_total_rows + window_rows > max_samples_per_batch and current_batch:
                # Save current batch and start a new one
                batches.append(current_batch)
                current_batch = []
                current_batch_total_rows = 0
            
            current_batch.append(window)
            current_batch_total_rows += window_rows
        
        # Add the last batch if it has any elements
        if current_batch:
            batches.append(current_batch)
        
        print(f'Created {len(batches)} batches with max {max_samples_per_batch} total rows per batch')
        for i, batch in enumerate(batches):
            total_rows = sum(len(window) for window in batch)
            print(f'Batch {i + 1}: {len(batch)} samples, {total_rows} total rows')
        
        return batches
