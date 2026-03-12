import statistics

import numpy as np

from method.lof_uns import LocalOutlierFactorUnsupervised
from method.isolation_forest import IsolationForest
from method.dist_k_Semi import Distance_Based_Semi

def uniform_even_numbers(min_val,max_val,num_params):
    params = np.linspace(min_val, max_val, num_params)
    params = [int(p) for p in params]
    params = [p if p%2==0  else max(2,p-1) for p in params ]
    params = list(set(params))
    params.sort()
    return params


def uniform(min_val,max_val,num_params,to_int=False):
    params= np.linspace(min_val, max_val, num_params)
    if to_int:
        params=[int(p) for p in params]
    params = list(set(params))
    params.sort()
    return params


def get_exponential_parameters(min_val, max_val, num_params,to_int=False):
    sqrt_max=int(np.sqrt(max_val))
    sqrt_min=max(int(np.sqrt(min_val)),1)
    positions= np.linspace(sqrt_min, sqrt_max, num_params)
    params=[]
    for x in positions:
        if to_int:
            params.append(int(x*x))
        else:
            params.append(x*x)
    params=list(set(params))
    params.sort()

    final_params = []
    for value in params:
        if min_val <= value <= max_val:
            final_params.append(value)

    return final_params


def online_technique(name,maximum_profile,multivariate=True):
    if name == 'IF':
        param_dict = {
        'n_estimators': [50, 100, 150, 200],
        'max_samples': uniform(min_val=max(min(maximum_profile,400),2)//4,max_val=min(maximum_profile,400),num_params=4,to_int=True),
        'random_state': [42],
        'max_features': [0.5, 0.6, 0.7, 0.8],
        'bootstrap': [True, False]
    }
    elif name == 'OCSVM':
        param_dict = {
            'kernel': ['linear', 'rbf', 'sigmoid'],
            'nu': [0.01, 0.05, 0.1, 0.15, 0.2, 0.5],
            'gamma': ['scale', 'auto'],
            'max_iter': [10000]
        }
    elif name == 'PB':
        param_dict = {}
    elif name == 'KNN':
        param_dict = {
            'k': get_exponential_parameters(min_val=1, max_val=min(100,maximum_profile), num_params=8,to_int=True),
            'window_norm': [False, True],
        }
    elif name == 'NP':
        param_dict ={
                'n_nnballs': uniform(min_val=10, max_val=150, num_params=5,to_int=True),
                'max_sample': uniform(min_val=max(min(maximum_profile,400)//4,2),max_val=min(maximum_profile,400),num_params=4,to_int=True),
                'sub_sequence_length': get_exponential_parameters(min_val=max(min(20,maximum_profile)//8,2), max_val=min(200,maximum_profile//2 if maximum_profile // 2 != 2 else maximum_profile), num_params=8,to_int=True),
                'aggregation_strategy': ['avg', 'max'],
                'random_state': [42]
            }
    elif name == 'LOF':
        param_dict =  {
            'n_neighbors': get_exponential_parameters(min_val=1, max_val=min(100,maximum_profile), num_params=8,to_int=True)
        }
    elif name == 'LTSF':
        param_dict = {
            'ltsf_type': ['Linear', 'DLinear', 'NLinear'],
            'features': ['M', 'MS'],
            'target': ['p2p_0'],
            'seq_len': uniform(min_val=max(min(100,maximum_profile)//8,2), max_val=min(200,maximum_profile), num_params=5,to_int=True),
            'pred_len': [1],
            'individual': [True, False],
            'train_epochs': [3, 5, 10, 15, 20, 25],
            'learning_rate': [0.001, 0.01, 0.1],
            'batch_size': [2, 4, 8, 16]
        }
    elif name == 'TRANAD':
        param_dict = {
            'window_size': get_exponential_parameters(min_val=10 if maximum_profile >= 22 else 2, max_val=min(200,maximum_profile//2 if maximum_profile // 2 != 2 else maximum_profile), num_params=6, to_int=True),
            'num_epochs': [5, 10, 15, 20, 25],
            'lr': [0.001, 0.01, 0.1, 0.05, 0.005]
        }
    elif name == 'USAD':
        param_dict = {
            'window_size': get_exponential_parameters(min_val=10 if maximum_profile >= 22 else 2, max_val=min(200,maximum_profile//2 if maximum_profile // 2 != 2 else maximum_profile), num_params=6, to_int=True),
            'num_epochs': [5, 10, 15, 20, 25],
            'lr': [0.001, 0.01, 0.1],
            'BATCH_SIZE': [2, 4, 8, 16],
            'hidden_size': [4, 8, 16, 32]
        }
    elif name == "HBOS":
        param_dict = {
            "n_bins": [5,10,15,20,30],
            "alpha": [0.1,0.3,0.5,0.7,0.9],
            "tol": [0.3,0.5,0.8],
            # only for univariate
        }
        if multivariate:
            param_dict["sub_sequence_length"]= [1]
        else:
            param_dict["sub_sequence_length"] = get_exponential_parameters(min_val=max(min(20,maximum_profile)//8,2), max_val=min(200,maximum_profile//2), num_params=8,to_int=True)

    elif name == "PCA":
        param_dict = {
            # only for univariate
        }
        if multivariate:
            param_dict["sub_sequence_length"] = [1]
        else:
            param_dict["sub_sequence_length"] = get_exponential_parameters(min_val=max(min(20,maximum_profile)//8,2), max_val=min(200,maximum_profile//2), num_params=8,to_int=True)

    elif name == "CNN" or name == "LSTM":
        param_dict = {
            'sub_sequence_length': get_exponential_parameters(min_val=14, max_val=min(200,maximum_profile//2), num_params=6,to_int=True),
            'predict_time_steps': [1], 
            'epochs': [3, 5, 10, 15, 20, 25, 100],
            'patience': [3, 5, 10]
        }
    elif name == "CHRONOS":
        param_dict = {
            'context_length': uniform(min_val=max(maximum_profile//4, 4), max_val=maximum_profile, num_params=8, to_int=True),
            'num_samples': [1, 3, 5, 10],
            'slide': [15],
            'max_steps': [1, 3, 5],
            'learning_rate': [0.001, 0.01, 0.1]
        }
    elif name == 'TimeLLM':
        param_dict = {
            'llm_model_type': ['GPT2'],
            'n_layers': [2], # 32
            'patch_size': [8], # 16
            'patch_stride': [4], # 8
            'd_llm': [768],
            'd_model': [2], # 32
            'd_ffn': [8], # 32
            'n_heads': [4], # 8
            'batch_size': [4],
            'n_steps': [max(
                get_exponential_parameters(min_val=10 if maximum_profile >= 22 else 2, max_val=min(200,maximum_profile//2 if maximum_profile // 2 != 2 else maximum_profile), num_params=6, to_int=True)
            )],
            'epochs': [25] # 10
        }
    elif name == 'TIMEMIXERPP':
        param_dict = {
            'seq_len': [max(uniform(min_val=max(min(100,maximum_profile)//8,2), max_val=min(200,maximum_profile), num_params=5,to_int=True))],
            'moving_avg': [statistics.median(get_exponential_parameters(min_val=10 if maximum_profile >= 22 else 2, max_val=min(200,maximum_profile//2 if maximum_profile // 2 != 2 else maximum_profile), num_params=6, to_int=True))],
            'train_epochs': [25],
        }
    # elif name == "ForecastingAnomalyPrediction":
    #     param_dict = {
    #         'device_type': ['cuda:1'],
    #         'random_seed': [42],
    #         # 'num_samples': [5],
    #         # 'seq_len': uniform(min_val=max(min(100, maximum_profile)//8,2), max_val=min(200,maximum_profile), num_params=5, to_int=True),
    #         # 'forecast_horizon': [10, 50, 64],
    #         'epochs': [1, 2, 3, 5, 10, 15, 20, 25],
    #         'anomaly_detector': [Distance_Based_Semi],
    #         'anomaly_detector_k': get_exponential_parameters(min_val=1, max_val=min(100,maximum_profile), num_params=8, to_int=True),
    #         'anomaly_detector_window_norm': [False, True],
    #     }
    else:
        assert False,"no method with that name"

    return param_dict


def incremental_technique(name,maximum_profile, multivariate=True):
    if name == 'IF':
        param_dict = {
        'n_estimators': [50, 100, 150, 200],
        'max_samples': uniform(min_val=max(min(maximum_profile,400),2)//4,max_val=min(maximum_profile,400),num_params=4,to_int=True),
        'random_state': [42],
        'max_features': [0.5, 0.6, 0.7, 0.8],
        'bootstrap': [True, False]
    }
    elif name == 'OCSVM':
        param_dict = {
            'kernel': ['linear', 'rbf', 'sigmoid'],
            'nu': [0.01, 0.05, 0.1, 0.15, 0.2, 0.5],
            'gamma': ['scale', 'auto'],
            'max_iter': [10000]
        }
    elif name == 'PB':
        param_dict = {}
    elif name == 'KNN':
        param_dict = {
            'k': get_exponential_parameters(min_val=1, max_val=min(100,maximum_profile), num_params=8,to_int=True),
            'window_norm': [False, True],
        }
    elif name == 'NP':
        param_dict ={
                'n_nnballs': uniform(min_val=10, max_val=150, num_params=5,to_int=True),
                'max_sample': uniform(min_val=max(min(maximum_profile,400)//4,2),max_val=min(maximum_profile,400),num_params=4,to_int=True),
                'sub_sequence_length': get_exponential_parameters(min_val=max(min(20,maximum_profile)//8,2), max_val=min(200,maximum_profile//2 if maximum_profile // 2 != 2 else maximum_profile), num_params=8,to_int=True),
                'aggregation_strategy': ['avg', 'max'],
                'random_state': [42]
            }
    elif name == 'LOF':
        param_dict =  {
            'n_neighbors': get_exponential_parameters(min_val=1, max_val=min(100,maximum_profile), num_params=8,to_int=True)
        }
    elif name == 'LTSF':
        param_dict = {
            'ltsf_type': ['Linear', 'DLinear', 'NLinear'],
            'features': ['M', 'MS'],
            'target': ['p2p_0'],
            'seq_len': uniform(min_val=max(min(200,maximum_profile)//8,2), max_val=min(200,maximum_profile), num_params=5,to_int=True),
            'pred_len': [1],
            'individual': [True, False],
            'train_epochs': [3, 5, 10, 15, 20, 25],
            'learning_rate': [0.001, 0.01, 0.1],
            'batch_size': [2, 4, 8, 16]
        }
    elif name == 'TRANAD':
        param_dict = {
            'window_size': get_exponential_parameters(min_val=10 if maximum_profile >= 22 else 2, max_val=min(200,maximum_profile//2 if maximum_profile // 2 != 2 else maximum_profile), num_params=6, to_int=True),
            'num_epochs': [5, 10, 15, 20, 25],
            'lr': [0.001, 0.01, 0.1, 0.05, 0.005]
        }
    elif name == 'USAD':
        param_dict = {
            'window_size': get_exponential_parameters(min_val=10 if maximum_profile >= 22 else 2, max_val=min(200,maximum_profile//2 if maximum_profile // 2 != 2 else maximum_profile), num_params=6, to_int=True),
            'num_epochs': [5, 10, 15, 20, 25],
            'lr': [0.001, 0.01, 0.1],
            'BATCH_SIZE': [2, 4, 8, 16],
            'hidden_size': [4, 8, 16, 32]
        }
    elif name == "HBOS":
        param_dict = {
            "n_bins": [5,10,15,20,30],
            "alpha": [0.1,0.3,0.5,0.7,0.9],
            "tol": [0.3,0.5,0.8],
            # only for univariate
        }
        if multivariate:
            param_dict["sub_sequence_length"] = [1]
        else:
            param_dict["sub_sequence_length"] = get_exponential_parameters(min_val=max(min(20,maximum_profile)//8,2), max_val=min(200,maximum_profile//2), num_params=8,to_int=True)
    elif name == "PCA":
        param_dict = {
            # only for univariate
        }
        if multivariate:
            param_dict["sub_sequence_length"] = [1]
        else:
            param_dict["sub_sequence_length"] =get_exponential_parameters(min_val=max(min(20,maximum_profile)//8,2), max_val=min(200,maximum_profile//2), num_params=8,to_int=True)
    elif name == "CNN" or name == "LSTM":
        param_dict = {
            'sub_sequence_length': get_exponential_parameters(min_val=14, max_val=min(200,maximum_profile//2), num_params=6,to_int=True),
            'predict_time_steps': [1],
            'epochs': [3, 5, 10, 15, 20, 25, 100],
            'patience': [3, 5, 10]
        }
    return param_dict


def unsupervised_technique(name,maximum_profile,multivariate=True):
    if name == 'NP':
        param_dict = {
            'n_nnballs': uniform(min_val=10, max_val=150, num_params=5,to_int=True),
            'max_sample': uniform(min_val=max(min(maximum_profile,160)//4,2),max_val=min(maximum_profile//2,160),num_params=4,to_int=True),
            'sub_sequence_length': get_exponential_parameters(min_val=max(min(20,maximum_profile)//8,3), max_val=min(200,maximum_profile//2 if maximum_profile // 2 != 2 else maximum_profile), num_params=8,to_int=True),
            'aggregation_strategy': ['avg', 'max'],
            'random_state': [42],
            'window': uniform(min_val=maximum_profile//4 if maximum_profile//4 > 5 else 5, max_val=maximum_profile, num_params=8,to_int=True),
            'slide': [0.33, 0.5, 1.0],
            'overlap_aggregation_strategy': ['first', 'last', 'avg'],
        }
    elif name == "DAMP":
        param_dict = {
            "sub_sequence_length":get_exponential_parameters(min_val=max(min(20,maximum_profile)//8,2), max_val=min(200,maximum_profile//2), num_params=8,to_int=True),
            "stride": [1],
            "init_length": uniform(min_val=maximum_profile//4, max_val=maximum_profile, num_params=8,to_int=True),
            "aggregation_strategy":['avg', 'max'],
        }
    elif name == 'KNN':
        param_dict = {
            'window': uniform(min_val=max(maximum_profile//4, 4), max_val=maximum_profile, num_params=8,to_int=True),
            'slide': [0.33, 0.5, 1.0],
            'k': get_exponential_parameters(min_val=1, max_val=min(100, maximum_profile), num_params=8,
                                            to_int=True),
            'window_norm': [False, True],
            'policy': ['or', 'and', 'first', 'last']
        }
    elif name == 'IF':
        param_dict = {
            'window': uniform(min_val=max(maximum_profile//4, 4), max_val=maximum_profile, num_params=8,to_int=True),
            'slide': [0.33, 0.5, 1.0],
            'n_estimators': [50, 100, 150, 200],
            'max_samples': uniform(min_val=max(min(maximum_profile, 400), 2) // 4, max_val=min(maximum_profile, 400),
                                   num_params=4, to_int=True),
            'max_features': [0.5, 0.6, 0.7, 0.8],
            'bootstrap': [True, False],
            'random_state': [42],
            'policy': ['or', 'and', 'first', 'last']
        }
    elif name == 'LOF':
        param_dict = {
            'n_neighbors': get_exponential_parameters(min_val=1, max_val=min(100, maximum_profile), num_params=8,
                                                      to_int=True),
            'window': uniform(min_val=max(maximum_profile//4, 4), max_val=maximum_profile, num_params=8, to_int=True),
            'slide': [0.33, 0.5, 1.0]
        }
    elif name == 'SAND':
        param_dict = {
            'pattern_length': get_exponential_parameters(min_val=max(min(20,maximum_profile)//8,3), max_val=min(200,maximum_profile//2 if maximum_profile // 2 != 2 else maximum_profile), num_params=8,to_int=True),
            'subsequence_length_multiplier': [3, 4, 5] if maximum_profile > 100 else [1, 2], #4*4 this is the sub size
            'alpha': [0.5, 0.75, 0.25],
            'init_length': uniform(min_val=maximum_profile//4, max_val=maximum_profile, num_params=8,to_int=True),
            'batch_size': uniform_even_numbers(min_val=maximum_profile//4, max_val=maximum_profile, num_params=8),
            'k': [4, 6, 7, 8, 9, 10],
            'aggregation_strategy': ['avg', 'max']
        }
    elif name == "HBOS":
        param_dict = {
            "n_bins": [5,10,15,20,30],
            "alpha": [0.1,0.3,0.5,0.7,0.9],
            "tol": [0.3,0.5,0.8],
            # only for univariate
            "window": [ uniform(min_val=maximum_profile//4, max_val=maximum_profile, num_params=8,to_int=True)]
        }
        if multivariate:
            param_dict["sub_sequence_length"] = [1]
        else:
            param_dict["sub_sequence_length"] = get_exponential_parameters(min_val=max(min(20,maximum_profile)//8,2), max_val=min(200,maximum_profile//2), num_params=8,to_int=True)
    elif name == "PCA":
        param_dict = {
            "window": [uniform(min_val=maximum_profile // 4, max_val=maximum_profile, num_params=8, to_int=True)]
        }
        if multivariate:
            param_dict["sub_sequence_length"] = [1]
        else:
            param_dict["sub_sequence_length"] = get_exponential_parameters(min_val=max(min(20,maximum_profile)//8,2), max_val=min(200,maximum_profile//2), num_params=8,to_int=True)
    elif name == "CHRONOS":
        param_dict = {
            'context_length': uniform(min_val=max(maximum_profile//4, 4), max_val=maximum_profile, num_params=13, to_int=True),
            'num_samples': [1, 3, 5, 10],
            'slide': [15],
        }
    elif name == "AUTOGLUON":
        param_dict = {
            'context_length': [max(uniform(min_val=max(maximum_profile//4, 4), max_val=maximum_profile, num_params=8, to_int=True))],
        }
    else:
        assert False,"no method with that name"

    return param_dict


def semi_technique(name,maximum_profile,multivariate=True):
    if name == 'IF':
        param_dict = {
        'n_estimators': [50, 100, 150, 200],
        'max_samples': uniform(min_val=max(min(maximum_profile,400),2)//4,max_val=min(maximum_profile,400),num_params=4,to_int=True),
        'random_state': [42],
        'max_features': [0.5, 0.6, 0.7, 0.8],
        'bootstrap': [True, False]
    }
    elif name == 'OCSVM':
        param_dict = {
            'kernel': ['linear', 'rbf', 'sigmoid'],
            'nu': [0.01, 0.05, 0.1, 0.15, 0.2, 0.5],
            'gamma': ['scale', 'auto'],
            'max_iter': [10000]
        }
    elif name == 'PB':
        param_dict = {}
    elif name == 'KNN':
        param_dict = {
            'k': get_exponential_parameters(min_val=1, max_val=min(100,maximum_profile), num_params=8,to_int=True),
            'window_norm': [False, True],
        }
    elif name == 'NP':
        param_dict ={
                'n_nnballs': uniform(min_val=10, max_val=150, num_params=5,to_int=True),
                'max_sample': uniform(min_val=max(min(maximum_profile,400)//4,2),max_val=min(maximum_profile,400),num_params=4,to_int=True),
                'sub_sequence_length':get_exponential_parameters(min_val=max(min(20,maximum_profile)//8,2), max_val=min(200,maximum_profile//2), num_params=8,to_int=True),
                'aggregation_strategy': ['avg', 'max'],
                'random_state': [42]
            }
    elif name == 'LOF':
        param_dict =  {
            'n_neighbors': get_exponential_parameters(min_val=1, max_val=min(100,maximum_profile), num_params=8,to_int=True)
        }
    elif name == 'LTSF':
        param_dict = {
            'ltsf_type': ['Linear', 'DLinear', 'NLinear'],
            'features': ['M', 'MS'],
            'target': ['p2p_0'],
            'seq_len': uniform(min_val=max(min(200,maximum_profile)//8,2), max_val=min(200,maximum_profile), num_params=5,to_int=True),
            'pred_len': [1],
            'individual': [True, False],
            'train_epochs': [3, 5, 10, 15, 20, 25],
            'learning_rate': [0.001, 0.01, 0.1],
            'batch_size': [2, 4, 8, 16]
        }
    elif name == 'TRANAD':
        param_dict = {
            'window_size':get_exponential_parameters(min_val=2, max_val=min(200,maximum_profile//2), num_params=6,to_int=True),
            'num_epochs': [5, 10, 15, 20, 25],
            'lr': [0.001, 0.01, 0.1, 0.05, 0.005]
        }
    elif name == 'USAD':
        param_dict = {
            'window_size': get_exponential_parameters(min_val=2, max_val=min(200,maximum_profile//2), num_params=6,to_int=True),
            'num_epochs': [5, 10, 15, 20, 25],
            'lr': [0.001, 0.01, 0.1],
            'BATCH_SIZE': [2, 4, 8, 16],
            'hidden_size': [4, 8, 16, 32]
        }
    elif name == "HBOS":
        param_dict = {
            "n_bins": [5,10,15,20,30],
            "alpha": [0.1,0.3,0.5,0.7,0.9],
            "tol": [0.3,0.5,0.8],
            # only for univariate
        }
        if multivariate:
            param_dict["sub_sequence_length"] = [1]
        else:
            param_dict["sub_sequence_length"] = get_exponential_parameters(min_val=max(min(20,maximum_profile)//8,2), max_val=min(200,maximum_profile//2), num_params=8,to_int=True)
    elif name == "PCA":
        param_dict = {
            # only for univariate
        }
        if multivariate:
            param_dict["sub_sequence_length"] = [1]
        else:
            param_dict["sub_sequence_length"] =get_exponential_parameters(min_val=max(min(20,maximum_profile)//8,2), max_val=min(200,maximum_profile//2), num_params=8,to_int=True)
    elif name == "CNN" or name == "LSTM":
        param_dict = {
            'sub_sequence_length': get_exponential_parameters(min_val=14, max_val=min(200,maximum_profile//2), num_params=6,to_int=True),
            'predict_time_steps': [1],
            'epochs': [3, 5, 10, 15, 20, 25, 100],
            'patience': [3, 5, 10]
        }
    return param_dict


def default_TSB_unsupervised(name,maximum_profile):
    if name == "IF":
        param_dict = {
            'n_estimators': [100],
            'max_samples': ['auto'],
            'random_state': [42],
            'max_features': [1.],
            'bootstrap': [False],
            'window': [maximum_profile],
            'slide': [1.0],
            'policy': ['first']
        }
    elif name == "SAND":
        param_dict = {
            # TO DO: correlation for subseuent lengh ?
            'pattern_length': [min(200,maximum_profile//8)],
            'subsequence_length_multiplier': [4],  # 4*4 this is the sub size
            'alpha': [0.5],
            'init_length': [maximum_profile],
            'batch_size': [] ,
            'k': [6],
            'aggregation_strategy': ['avg']
        }

        # clf = SAND(pattern_length=slidingWindow,subsequence_length=4*(slidingWindow))
        # x = data
        # clf.fit(x,overlaping_rate=int(1.5*slidingWindow))
        # 			self.alpha = 0.5
        # 			self.batch_size = 0
    elif name == "LOF":
        param_dict = {
            'n_neighbors': 20,
            'window': [maximum_profile],
            'slide': [1.0]
        }
    elif name == "DAMP":
        param_dict = {
            "sub_sequence_length": [min(200, maximum_profile // 8)],
            "stride": [1],
             "init_length": [maximum_profile],
             'aggregation_strategy': ['avg'],
        }
    elif name == 'NP':
        param_dict = {
            "n_nnballs": [1],
            "max_sample": [maximum_profile//2],
            "sub_sequence_length": [min(200, maximum_profile // 8)],
            'aggregation_strategy': ['avg'],
            'random_state': [42],
            #These are for unsupervised:
            'window': [maximum_profile],
            'slide': [1.0],
            'overlap_aggregation_strategy': ['first'],
        }
    elif name == "KNN":
        param_dict = {
            'window': [maximum_profile],
            'slide': [1.0],
            'k': [20],
            'window_norm': [False],
            'policy': ['first']
        }
    elif name == "HBOS":
        param_dict = {
            "n_bins": [10],
            "alpha": [0.1],
            "tol": [0.5],
            # only for univariate
            "sub_sequence_length": [min(200, maximum_profile // 8)],
            "window": [maximum_profile]
        }
    elif name == "PCA":
        param_dict = {
            # only for univariate
            "sub_sequence_length": [min(200, maximum_profile // 8)],
            "window": [maximum_profile]
        }
    else:
        assert False, f"no default parameters for technique with name {name}"

    return param_dict

def default_TSB_semi(name,maximum_profile):
    if name == "IF":
        param_dict = {
            'n_estimators': [100],
            'max_samples': ['auto'],
            'random_state': [42],
            'max_features': [1.],
            'bootstrap': [False],
        }
    elif name == 'OCSVM':
        param_dict = {
            'kernel': ['rbf'],
            'nu': [0.5],
            'gamma': ['auto'],
        }
        # kernel = 'rbf', degree = 3, gamma = 'auto', coef0 = 0.0,
        # tol = 1e-3, nu = 0.5, shrinking = True, cache_size = 200,
        # verbose = False, max_iter = -1, contamination = 0.1
    elif name == "LOF":
        param_dict = {
            'n_neighbors': [20],
        }
    elif name == 'NP':
        param_dict = {
            "n_nnballs": [1],
            "max_sample": [maximum_profile//2],
            "sub_sequence_length": [min(200, maximum_profile // 8)],
            'aggregation_strategy': ['avg'],
            'random_state': [42],
        }
    elif name == "PCA":
        param_dict = {
            "sub_sequence_length": [min(200, maximum_profile // 8)]
        }
    elif name == "LSTM" or name == "CNN":
        param_dict = {
            "sub_sequence_length":[min(100, maximum_profile // 8)],
        }
    elif name == "CNN":
            param_dict = {
            "sub_sequence_length":[min(100, maximum_profile // 8)],
        }
    elif name == "HBOS":
        param_dict = {
            "n_bins": [10],
            "alpha": [0.1],
            "tol": [0.5],
            # only for univariate
            "sub_sequence_length": [min(200, maximum_profile // 8)]
        }
    elif name == 'KNN':
        param_dict = {
            'k': [20],
            'window_norm': [False],
        }
    else:
        assert False, f"no default parameters for technique with name {name}"

    return param_dict

def post_proccessing_params(name,maximum_profile):
    if name == "Default":
        param_dict={}
    elif name == 'Dynamic Threshold':
        param_dict = {
            "epsilon":[0.05],
            "history_window":[1000],
        }
    elif name == 'Moving2T':
        param_dict = {
            "factor":[3],
            "history_window":[1000],
            "exclude":[False]
        }
    elif name == 'SelfTuning':
        param_dict = {
            "window_length":get_exponential_parameters(min_val=10, max_val=min(200,maximum_profile//2), num_params=6,to_int=True),
        }
    elif name == 'Moving Average':
        param_dict = {
            "window_length":get_exponential_parameters(min_val=10, max_val=min(200,maximum_profile//2), num_params=6,to_int=True),
        }
    else:
        assert False, f"no default post_processing for technique with name {name}"
    return param_dict

def pre_proccessing_params(name,maximum_profile):
    if name == "Default":
        param_dict={}
    elif name == "Keep Features":
        param_dict={
            "selected_features":[]
        }
    elif name == "MinMax Scaler (semi)":
        param_dict={
        }
    elif name == "Windowing (one column)":
        param_dict={
            "slidingWindow":[10],
            "col_pos":0
        }
    elif name=="Mean Aggregator":
        param_dict = {
            "period": ['10T'],
        }
    else:
        assert False, f"no default pre_processing for technique with name {name}"
    return param_dict

def profile_values(max_wait, moment=False):
    if not moment:
        result = uniform(min_val= max(max_wait// 10, 5), max_val=max_wait, num_params=16, to_int=True)

        if 0 in result:
            result.remove(0)

        return result
    else:
        return [1027]#uniform(min_val=1024, max_val=max_wait if max_wait >= 1024 else 1024, num_params=16, to_int=True)


def incremental_windows(max_wait):
    values=uniform(min_val= max(max_wait// 10, 1), max_val=max_wait, num_params=13, to_int=True),
    incremental_slide = values[0]
    if 1 in incremental_slide:
        incremental_slide.remove(1)

    if 0 in incremental_slide:
        incremental_slide.remove(0)

    initial_incremental_window_length = values[0]
    incremental_window_length = values[0] #+ [1000000000]

    return incremental_slide, initial_incremental_window_length, incremental_window_length