import math

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from utils.distance import Fourier

from method.semi_supervised_method import SemiSupervisedMethodInterface
from exceptions.exception import NotFitForSourceException
from pdm_evaluation_types.types import EventPreferences
from utils import utils


class Cnn(SemiSupervisedMethodInterface):
    def __init__(self,
                 event_preferences: EventPreferences,
                 sub_sequence_length = 100,
                 predict_time_steps=1, 
                 contamination = 0.1, 
                 epochs = 10, 
                 patience = 10, 
                 verbose=0,
                 ratio=0.15,
                 distance='Fourier',
                 *args,
                 **kwargs
                 ):
        super().__init__(event_preferences=event_preferences)
        
        self.sub_sequence_length = sub_sequence_length
        self.predict_time_steps = predict_time_steps
        self.contamination = contamination
        self.epochs = epochs
        self.patience = patience
        self.verbose = verbose
        self.ratio = ratio
        
        if distance == 'Fourier':
            self.measure = Fourier
        else:
            raise NotImplementedError('No other available distances')

        self.model_per_source = {}
        self.sub_len_per_source = {}
        self.n_initial_per_source = {}

    
    def fit(self, historic_data: list[pd.DataFrame], historic_sources: list[str], event_data: pd.DataFrame) -> None:
        for current_historic_data, current_historic_source in zip(historic_data, historic_sources):
            sub_sequence_length = self.sub_sequence_length
            predict_time_steps = self.predict_time_steps

            if len(current_historic_data.columns)>1:
                X_data = current_historic_data.values
                X_train, Y_train = self._create_dataset(X_data, 1, predict_time_steps)

                self.sub_len_per_source[current_historic_source] = 1
            else:
                if len(current_historic_data.index) < self.sub_sequence_length:
                    self.sub_len_per_source[current_historic_source] = len(current_historic_data.index) - 1
                    
                    X_train, Y_train = self._create_dataset(current_historic_data.values, self.sub_len_per_source[current_historic_source], predict_time_steps)
                else:
                    self.sub_len_per_source[current_historic_source] = self.sub_sequence_length

                    X_train, Y_train = self._create_dataset(current_historic_data.values, self.sub_len_per_source[current_historic_source], predict_time_steps)
            
            if self.sub_len_per_source[current_historic_source] >= 14:
                X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

                model = Sequential()
                model.add(Conv1D(filters=8,
                                kernel_size=2,
                                strides=1,
                                padding='same',
                                activation='relu',
                                input_shape=(sub_sequence_length, 1)))
                model.add(MaxPooling1D(pool_size=2)) 
                model.add(Conv1D(filters=16,
                                kernel_size=2,
                                strides=1,
                                padding='valid',
                                activation='relu'))
                model.add(MaxPooling1D(pool_size=2))
                model.add(Conv1D(filters=32,
                                kernel_size=2,
                                strides=1,
                                padding='valid',
                                activation='relu'))
                model.add(MaxPooling1D(pool_size=2))
                model.add(Flatten())
                model.add(Dense(units=64, activation='relu'))  
                model.add(Dropout(rate=0.2))
                model.add(Dense(units=predict_time_steps))
                
                model.compile(loss='mse', optimizer='adam')
                
                es = EarlyStopping(monitor='val_loss', mode='min', verbose=self.verbose, patience=self.patience)
                
                model.fit(X_train,Y_train,validation_split=self.ratio,
                        epochs=self.epochs,batch_size=64,verbose=self.verbose, callbacks=[es])
                
                self.model_per_source[current_historic_source] = model
                self.n_initial_per_source[current_historic_source] = X_train.shape[0]
            else:
                self.model_per_source[current_historic_source] = -1


    def predict(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> list[float]:
        if source not in self.model_per_source:
            raise NotFitForSourceException()
        
        if self.model_per_source[source] == -1:
            return [0 for i in range(target_data.shape[0])]

        predict_time_steps = self.predict_time_steps

        if len(target_data.columns) > 1:
            sub_sequence_length = 1
            X_test, Y_test = self._create_dataset(target_data.values, sub_sequence_length, predict_time_steps)
        else:
            if len(target_data.index) < self.sub_len_per_source[source]:
                return [0.0 for i in range(len(target_data.index))]
            
            sub_sequence_length = self.sub_len_per_source[source]

            X_test, Y_test = self._create_dataset(target_data.values, sub_sequence_length, predict_time_steps)
        
        self.n_test_ = len(target_data)
        
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        prediction = self.model_per_source[source].predict(X_test)

        score = []

        measure = self.measure()
        measure.detector = self.model_per_source[source]
        measure.set_param()

        for i in range(prediction.shape[0]):
            score.append(measure.measure(Y_test[i], prediction[i], 0))
        
        score = np.array(score)
        decision_scores_ = np.zeros(self.n_test_)
        
        decision_scores_[sub_sequence_length:(self.n_test_-self.predict_time_steps+1)] = score
        decision_scores_[:sub_sequence_length] = score[0]
        decision_scores_[self.n_test_ - self.predict_time_steps + 1:] = score[-1]

        return decision_scores_.tolist()


    def predict_one(self, new_sample: pd.Series, source: str, is_event: bool) -> float:
         # TODO need to keep buffer until profile size are encountered and then start predicting
        return 0.0


    def get_library(self) -> str:
        return 'no_save'
    

    def __str__(self) -> str:
        return 'CNN'
    

    def get_params(self) -> dict:
        return {
            'sub_sequence_length': self.sub_sequence_length, 
            'predict_time_steps': self.predict_time_steps, 
            'contamination': self.contamination, 
            'epochs': self.epochs, 
            'patience': self.patience, 
            'verbose': self.verbose,
        }


    def get_all_models(self):
        pass

    
    def _create_dataset(self, X, sub_sequence_length, predict_time_steps=1):
        Xs, ys = [], []
        for i in range(len(X) - sub_sequence_length - predict_time_steps + 1):
            tmp = X[i : i + sub_sequence_length + predict_time_steps]
            tmp= MinMaxScaler(feature_range=(0,1)).fit_transform(tmp.reshape(-1,1)).ravel()
            x = tmp[:sub_sequence_length]
            y = tmp[sub_sequence_length:]
            
            Xs.append(x)
            ys.append(y)

        return np.array(Xs), np.array(ys)