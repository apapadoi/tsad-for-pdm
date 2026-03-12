import pandas as pd
import mlflow

from method.unsupervised_method import UnsupervisedMethodInterface
from pdm_evaluation_types.types import EventPreferences
from method.sand_core import SAND as SANDInitial


class Sand(UnsupervisedMethodInterface):
    def __init__(self, 
                 event_preferences: EventPreferences, 
                 pattern_length: int, 
                 alpha: float, 
                 init_length: int, 
                 batch_size: int,
                 subsequence_length_multiplier: int = 4, 
                 k: int = 6,
                 aggregation_strategy: str = 'avg', 
                 *args, 
                 **kwargs
    ):
        super().__init__(event_preferences=event_preferences)
        self.initial_args = args
        self.initial_kwargs = kwargs

        self.pattern_length = pattern_length
        self.subsequence_length_multiplier = subsequence_length_multiplier
        self.subsequence_length = subsequence_length_multiplier * pattern_length
        self.k = k

        self.alpha = alpha
        self.init_length = init_length
        self.batch_size = batch_size
        self.overlapping_rate = subsequence_length_multiplier * pattern_length

        self.aggregation_strategy = aggregation_strategy


    def fit(self, historic_data: list[pd.DataFrame], historic_sources: list[str], event_data: pd.DataFrame) -> None:
        pass
        

    def predict(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> list[float]:
        scores_for_all_dimensions = []
        for column in target_data.columns:
            current_model = SANDInitial(pattern_length=self.pattern_length, subsequence_length=self.subsequence_length, k=self.k)
            current_model.fit(target_data[[column]].to_numpy().ravel(), online=True, alpha=self.alpha, init_length=self.init_length, batch_size=self.batch_size, overlaping_rate=self.overlapping_rate)
            scores_for_all_dimensions.append(current_model.decision_scores_.tolist())
        
        if self.aggregation_strategy == 'avg':
            num_dimensions = len(scores_for_all_dimensions)
            return [sum(current_time_step_values) / num_dimensions for current_time_step_values in zip(*scores_for_all_dimensions)]
        else:
            return [max(current_time_step_values) for current_time_step_values in zip(*scores_for_all_dimensions)]
    
        
    def predict_one(self, new_sample: pd.Series, source: str, is_event: bool) -> float:
        # TODO need to keep buffer until profile size are encountered and then start predicting
        return 0
    

    def get_library(self) -> str:
        return 'no_save'
    
    
    def __str__(self) -> str:
        return 'Sand'


    def get_params(self) -> dict:
        return {
            'pattern_length': self.pattern_length,
            'subsequence_length_multiplier': self.subsequence_length_multiplier,
            'k': self.k,
            'alpha': self.alpha,
            'init_length': self.init_length,
            'batch_size': self.batch_size,
            'aggregation_strategy': self.aggregation_strategy,
        }


    def get_all_models(self):
        pass
