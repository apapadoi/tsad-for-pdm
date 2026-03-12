import pandas as pd
from method.distance_based_k_r_Core import distance_based_k_r
from method.unsupervised_method import UnsupervisedMethodInterface
from exceptions.exception import NotFitForSourceException
from pdm_evaluation_types.types import EventPreferences

import warnings
class Distance_Based_Uns(UnsupervisedMethodInterface):
    def __init__(self,
                 event_preferences: EventPreferences,
                 k: int,
                 metric:str='euclidean',
                 window_norm=False,
                 window=40,
                 slide=0.5,
                 policy="or",
                 *args,
                 **kwargs
                 ):
        super().__init__(event_preferences=event_preferences)
        self.k = k
        self.metric = metric
        self.window_norm = window_norm
        self.model_per_source = {}
        self.slide = int(slide*window)
        self.window = window
        self.policy = policy



    def predict(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> list[float]:
        scores=self._inner_batch_predict(target_data)
        return scores


    def _inner_batch_predict(self,dfr:pd.DataFrame):
        tempmodel = distance_based_k_r(k=self.k,window_norm=self.window_norm,metric=self.metric)
        final_score = {}
        pos = 0
        df=dfr.values
        for pos in range(self.window, len(df), self.slide):
            currentdf = df[max(pos - self.window, 0):pos]
            ids = [kati for kati in range(max(pos - self.window, 0), pos)]
            tempmodel.fit(currentdf)
            scores = tempmodel.predict(currentdf)
            final_score = self.combinescores(final_score, ids,scores)
        if pos < len(df):
            pos = len(df)
            currentdf = df[max(pos - self.window, 0):pos]
            tempmodel.fit(currentdf)
            scores = tempmodel.predict(currentdf)
            ids = [kati for kati in range(max(pos - self.window, 0), pos)]
            final_score = self.combinescores(final_score, ids,scores)
        scores_to_return = []
        for ind in range(0, len(df)):
            scores_to_return.append(final_score[ind])
        return scores_to_return


    def combinescores(self, final_score, ids,scores):
        for sc, ind in zip(scores, ids):
            if ind in final_score.keys():
                if self.policy == "or":
                    final_score[ind] = max(final_score[ind], sc)
                elif self.policy == "and":
                    final_score[ind] = min(final_score[ind], sc)
                elif self.policy == "first":
                    final_score[ind] = final_score[ind]
                elif self.policy == "last":
                    final_score[ind] = sc
                else:
                    final_score[ind] = sc
            else:
                final_score[ind] = sc
        return final_score

    def predict_one(self, new_sample: pd.Series, source: str, is_event: bool) -> float:
        # TODO need to keep buffer until profile size are encountered and then start predicting
        return 0

    def get_library(self) -> str:
        return 'no_save'

    def __str__(self) -> str:
        return 'Distance Based k (uns)'

    def get_params(self) -> dict:
        return {
            'k': self.k,
            'window_norm': self.window_norm,
            'metric': self.metric,
            'slide':self.slide,
            'window':self.window,
            'policy':self.policy,
        }

    def get_all_models(self):
        pass