import math

from method.NPcore import NeighborProfileAllFeatures
import pandas as pd

from method.unsupervised_method import UnsupervisedMethodInterface
from exceptions.exception import NotFitForSourceException
from pdm_evaluation_types.types import EventPreferences


class NeighborProfileUns(UnsupervisedMethodInterface):
    def __init__(self,
                 event_preferences: EventPreferences,
                 n_nnballs=100,
                 max_sample=8,
                 random_state=None,
                 scale="zscore",
                 sub_sequence_length=10,
                 aggregation_strategy:str='avg',
                 window=60,
                 slide=0.5,
                 overlap_aggregation_strategy='first',
                 *args,
                 **kwargs
                 ):
        super().__init__(event_preferences=event_preferences)
        self.n_nnballs = n_nnballs
        self.max_sample = max_sample
        self.random_state = random_state
        self.scale = scale
        self.sub_sequence_length = sub_sequence_length
        self.aggregation_strategy = aggregation_strategy
        self.model_per_source={}
        self.window=window
        self.slide_to_log=slide
        self.slide=int(slide*window)
        self.overlap_aggregation_strategy=overlap_aggregation_strategy


    def _fit(self, historic_data: list[pd.DataFrame], historic_sources: list[str], event_data: pd.DataFrame) -> None:
        for df,source in zip(historic_data,historic_sources):
            self.model_per_source[source]=NeighborProfileAllFeatures(n_nnballs=self.n_nnballs,
                 max_sample=self.max_sample,
                 random_state=self.random_state,
                 scale=self.scale,
                 sub_sequence_length=self.sub_sequence_length,
                 aggregation_strategy =self.aggregation_strategy)

            self.model_per_source[source].fit(df)


    def fit(self, historic_data: list[pd.DataFrame], historic_sources: list[str], event_data: pd.DataFrame) -> None:
        pass

    def predict(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> list[float]:
        WindowsScores=[]
        Windowsdata=[]
        first=True
        for i in range(len(target_data.index)):
            if first:
                wdf=target_data.iloc[0:min(len(target_data.index),self.window+i*self.slide)]
                first=False
            else:
                wdf = target_data.iloc[min(len(target_data.index)-self.window,i*self.slide):min(len(target_data.index),self.window+i*self.slide)]
            Windowsdata.append(wdf)
            wscores=self._predict(wdf,source,event_data)
            WindowsScores.append(wscores)
            if self.window+i*self.slide>=len(target_data.index):
                break

        finalscores=self._combinescores(target_data,WindowsScores,Windowsdata)




        return finalscores
    
    
    def _predict(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> list[float]:
        self._fit([target_data],[source],event_data)

        if source in self.model_per_source.keys():
            scores=self.model_per_source[source].predict(target_data)
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
            scores=[sndmin if math.isinf(sc) else sc for sc in scores]
            if len(scores)<len(target_data.index):
                pad=[min(scores) for i in range(len(target_data.index)-len(scores))]
                pad.extend(scores)
                scores=pad
            return scores
        else:
            raise NotFitForSourceException()
        
        
    def _combinescores(self,target_data,WindowsScores,Windowsdata):
        finalscores=[]
        if self.overlap_aggregation_strategy == "avg":
            scores={}
            for ind in target_data.index:
                scores[ind]=[]
            for ws,wdf in zip(WindowsScores,Windowsdata):
                for sc,ind in zip(ws,wdf.index):
                    scores[ind].append(sc)
            for ind in target_data.index:
                finalscores.append(sum(scores[ind])/len(scores[ind]))
        elif self.overlap_aggregation_strategy == "first":
            scores = {}
            for ws,wdf in zip(WindowsScores,Windowsdata):
                for sc,ind in zip(ws,wdf.index):
                    if ind not in scores.keys():
                        scores[ind]=sc
            for ind in target_data.index:
                finalscores.append(scores[ind])
        elif self.overlap_aggregation_strategy == "last":
            scores = {}
            for ws,wdf in zip(WindowsScores,Windowsdata):
                for sc,ind in zip(ws,wdf.index):
                    scores[ind]=sc
            for ind in target_data.index:
                finalscores.append(scores[ind])
        return finalscores



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
            'random_state': self.random_state,
            'window':self.window,
            'slide':self.slide_to_log,
            'overlap_aggregation_strategy':self.overlap_aggregation_strategy
        }

    def get_all_models(self):
        pass