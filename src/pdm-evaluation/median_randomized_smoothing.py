import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np

from method.semi_supervised_method import SemiSupervisedMethodInterface
from pdm_evaluation_types.types import EventPreferences


class MedianRandomizedSmoothing(SemiSupervisedMethodInterface):
    def __init__(self, 
                 event_preferences: EventPreferences, 
                 sources: list,
                 sigma: float = 100.0, 
                 moving_average: bool = False, 
                 technique: str = 'OCSVM', 
                 *args, 
                 **kwargs
    ):
        super().__init__(event_preferences=event_preferences)
        self.sources = sources
        self.sigma = sigma
        self.moving_average = moving_average
        self.technique = technique
        self.source_scores = {}
        self.batch_index = {}
        self.profile_size = kwargs.get("profile_size", -1)

        assert self.profile_size != -1

        self._load_and_aggregate_scores()


    def _load_and_aggregate_scores(self):
        # Construct experiment name
        # Experiment name format: "Median Randomized Smoothing (sigma=100.0) (moving_average=False) EDP OCSVM"
        experiment_name = f"Median Randomized Smoothing (sigma={float(self.sigma)}) (moving_average={self.moving_average}) EDP {self.technique}"
        
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment '{experiment_name}' not found.")
            
        # Query runs - get all (using large max_results)
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], max_results=1000)
        
        # Validation check assertion
        # assert len(runs) == 100, f"Expected 100 runs, but found {len(runs)}"
        
        if runs.empty:
             raise ValueError(f"No runs found for experiment '{experiment_name}'")
             
        score_dfs = []
        client = MlflowClient()
        
        for run_id in runs['run_id']:
            artifacts = client.list_artifacts(run_id)
            score_file_path = None
            for artifact in artifacts:
                # Look for files matching 'scores_{id}.csv'
                if artifact.path.startswith("scores_") and artifact.path.endswith(".csv"):
                    score_file_path = artifact.path
                    break
            
            if score_file_path:
                local_path = client.download_artifacts(run_id, score_file_path)
                # Load CSV, header=None as per instruction
                df = pd.read_csv(local_path, header=None)
                print(df.shape)
                score_dfs.append(df)
        
        if not score_dfs:
            raise RuntimeError(f"No score files found for experiment {experiment_name}")
        # elif len(score_dfs) != 100:
        #     raise RuntimeError(f"Expected 100 score files, but found {len(score_dfs)}")

        # Aggregate using median per cell
        try:
            # Stack all dataframes together
            stacked_scores = np.stack([df.values for df in score_dfs], axis=0)
            
            # Calculate median per cell, ignoring NaNs
            aggregated_matrix = np.nanmedian(stacked_scores, axis=0)
            aggregated_df = pd.DataFrame(aggregated_matrix)
            print(aggregated_df.shape)
        except Exception as e:
            raise RuntimeError(f"Error aggregating scores: {e}")
        
        # Parse aggregated scores into source-specific lists
        # Source order and counts from test.py
        sources = self.sources#['T07', 'T01', 'T06', 'T11']
        columns_per_source  = {
            'T07': 4,
            'T01': 2,
            'T06': 3,
            'T11': 3
        }
        # counts = [4, 2, 3, 3]
        
        current_col_idx = 0
        for source in sources:#, counts):
            self.source_scores[source] = []
            self.batch_index[source] = 0

            for _ in range(columns_per_source[source]):
                if current_col_idx >= aggregated_df.shape[1]:
                    break
                # Get column, drop NaNs, add to list
                col_values = aggregated_df.iloc[:, current_col_idx].dropna().tolist()
                print(len(col_values))
                self.source_scores[source].append(col_values)
                current_col_idx += 1


    def fit(self, historic_data: list[pd.DataFrame], historic_sources: list[str], event_data: pd.DataFrame) -> None:
        # No training needed; scores are pre-aggregated from MLflow
        pass


    def predict(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> list[float]:
        # Return scores for the given source (next batch)
        if source in self.source_scores:
            idx = self.batch_index[source]
            if idx < len(self.source_scores[source]):
                scores = self.source_scores[source][idx][self.profile_size:]
                self.batch_index[source] += 1
                
                # Handle size mismatch
                if len(scores) != len(target_data):
                    raise RuntimeError("Mismatch in number of scores and target data samples.")
                    # return scores[:len(target_data)]
                else:
                    return scores

        return [0.0] * len(target_data)


    def predict_one(self, new_sample: pd.Series, source: str, is_event: bool) -> float:
        # if source in self.source_scores:
        #     idx = self.current_index[source]
        #     scores = self.source_scores[source]
            
        #     if idx < len(scores):
        #         score = scores[idx]
        #         self.current_index[source] += 1
        #         return score
        
        return 0.0


    def get_library(self) -> str:
        return 'no_save'
    

    def get_params(self) -> dict:
        return {
            'sigma': self.sigma, 
            'moving_average': self.moving_average, 
            'technique': self.technique
        }
    

    def get_all_models(self):
        pass


    def __str__(self) -> str:
        return f"MedianRandomizedSmoothing_{self.technique}"