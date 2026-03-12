import math

from method.NPcore import NeighborProfileAllFeatures
import pandas as pd

from method.semi_supervised_method import SemiSupervisedMethodInterface
from exceptions.exception import NotFitForSourceException
from pdm_evaluation_types.types import EventPreferences

import warnings
class NeighborProfileSemi(SemiSupervisedMethodInterface):
    def __init__(self,
                 event_preferences: EventPreferences,
                 n_nnballs=100,
                 max_sample=8,
                 random_state=None,
                 scale="zscore",
                 sub_sequence_length=10,
                 aggregation_strategy:str='avg',
                 *args,
                 **kwargs
                 ):
        super().__init__(event_preferences=event_preferences)
        self.n_nnballs = n_nnballs
        self.max_sample = max_sample
        self.random_state = random_state
        self.scale = scale
        self.sub_sequence_length = sub_sequence_length
        self.aggregation_strategy=aggregation_strategy
        self.model_per_source={}


    def fit(self, historic_data: list[pd.DataFrame], historic_sources: list[str], event_data: pd.DataFrame) -> None:
        for df,source in zip(historic_data,historic_sources):
            self.model_per_source[source]=NeighborProfileAllFeatures(n_nnballs=self.n_nnballs,
                 max_sample=self.max_sample,
                 random_state=self.random_state,
                 scale=self.scale,
                 sub_sequence_length=self.sub_sequence_length,
                 aggregation_strategy =self.aggregation_strategy)

            self.model_per_source[source].fit(df.copy())

    def predict(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> list[float]:
        if source in self.model_per_source.keys():
            if len(target_data.index)<=self.sub_sequence_length:
                scores=[0 for i in target_data.index]
            else:
                scores = self.model_per_source[source].predict(target_data)
                if len(scores) > 1:
                    sndmin = 0
                    tempsorted=sorted(scores)
                    for sc in tempsorted:
                        if math.isinf(sc):
                            continue
                        sndmin=sc
                        break
                else:
                    sndmin = 0
                scores = [sndmin if math.isinf(sc) else sc for sc in scores]
            if len(scores)<len(target_data.index):
                pad=[min(scores) for i in range(len(target_data.index)-len(scores))]
                pad.extend(scores)
                scores=pad
            return scores
        else:
            raise NotFitForSourceException()

    def predict_one(self, new_sample: pd.Series, source: str, is_event: bool) -> float:
        # TODO need to keep buffer until profile size are encountered and then start predicting
        return 0

    def get_library(self) -> str:
        return 'no_save'

    def __str__(self) -> str:
        return 'NeighborProfile'

    def get_params(self) -> dict:
        return {
            'n_nnballs': self.n_nnballs,
            'max_sample': self.max_sample,
            'scale': self.scale,
            'sub_sequence_length': self.sub_sequence_length,
            'aggregation_strategy':self.aggregation_strategy,
            'random_state': self.random_state
        }

    def get_all_models(self):
        pass
