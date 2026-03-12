




from __future__ import division
from __future__ import print_function

import numbers

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from utils import utils

from method.semi_supervised_method import SemiSupervisedMethodInterface
from pdm_evaluation_types.types import EventPreferences


class HBOS(SemiSupervisedMethodInterface):
    def __init__(self, event_preferences: EventPreferences,n_bins=10, alpha=0.1, tol=0.5, sub_sequence_length=1, *args, **kwargs):
        super().__init__(event_preferences=event_preferences)
        self.initial_args = args
        self.initial_kwargs = kwargs
        self.n_bins = n_bins
        self.alpha = alpha
        self.tol = tol
        # This is used only in case of univariate data.
        self.sub_sequence_length = sub_sequence_length
        self.model_per_source={}
        self.sub_len_per_source={}


    def fit(self, historic_data: list[pd.DataFrame], historic_sources: list[str], event_data: pd.DataFrame) -> None:
        for current_historic_data, current_historic_source in zip(historic_data, historic_sources):
            if len(current_historic_data.columns)>1:
                X_data = current_historic_data.values
            else:
                if len(current_historic_data.index)<self.sub_sequence_length:
                    X_data = utils.Window(window=len(current_historic_data.index)-1).convert(
                        current_historic_data[0].values).to_numpy()
                    self.sub_len_per_source[current_historic_source]=len(current_historic_data.index)-1
                else:
                    X_data = utils.Window(window=self.sub_sequence_length).convert(current_historic_data[0].values).to_numpy()
                    self.sub_len_per_source[current_historic_source]=self.sub_sequence_length

            self.model_per_source[current_historic_source] = HBOScore(n_bins=self.n_bins,alpha=self.alpha,tol=self.tol)
            self.model_per_source[current_historic_source].fit(X_data)


    def predict(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> list[float]:
        if len(target_data.columns) > 1:
            X_data = target_data.values
        else:
            if len(target_data.index)<self.sub_len_per_source[source]:
                return [0.0 for i in range(len(target_data.index))]
            X_data = utils.Window(window=self.sub_len_per_source[source]).convert(target_data[0].values).to_numpy()

        scores=self.model_per_source[source].decision_function(X_data)
        scores=[min(scores) for i in range(len(target_data.index)-len(scores))]+list(scores)
        return scores

    def predict_one(self, new_sample: pd.Series, source: str, is_event: bool) -> float:
        # TODO need to keep buffer until profile size are encountered and then start predicting
        return -self.model_per_source[source].score_samples([new_sample.to_numpy()]).tolist()[0]

    def get_library(self) -> str:
        return 'no_save'

    def __str__(self) -> str:
        return 'HBOS'

    def get_params(self) -> dict:
        return {
            "sub_sequence_length": self.sub_sequence_length,
            "n_bins": self.n_bins,
            "tol": self.tol,
            'alpha': self.alpha,
        }

    def get_all_models(self):
        pass


class HBOScore():
    """Histogram- based outlier detection (HBOS) is an efficient unsupervised
    method. It assumes the feature independence and calculates the degree
    of outlyingness by building histograms. See :cite:`goldstein2012histogram`
    for details.
    Parameters
    ----------
    n_bins : int, optional (default=10)
        The number of bins.
    alpha : float in (0, 1), optional (default=0.1)
        The regularizer for preventing overflow.
    tol : float in (0, 1), optional (default=0.5)
        The parameter to decide the flexibility while dealing
        the samples falling outside the bins.
    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set,
        i.e. the proportion of outliers in the data set. Used when fitting to
        define the threshold on the decision function.
    Attributes
    ----------
    bin_edges_ : numpy array of shape (n_bins + 1, n_features )
        The edges of the bins.
    hist_ : numpy array of shape (n_bins, n_features)
        The density of each histogram.
    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is fitted.
    threshold_ : float
        The threshold is based on ``contamination``. It is the
        ``n_samples * contamination`` most abnormal samples in
        ``decision_scores_``. The threshold is calculated for generating
        binary outlier labels.
    labels_ : int, either 0 or 1
        The binary labels of the training data. 0 stands for inliers
        and 1 for outliers/anomalies. It is generated by applying
        ``threshold_`` on ``decision_scores_``.
    """

    def __init__(self, n_bins=10, alpha=0.1, tol=0.5, contamination=0.1):
        self.n_bins = n_bins
        self.alpha = alpha
        self.tol = tol
        self.model_name = 'HBOS'

        check_parameter(alpha, 0, 1, param_name='alpha')
        check_parameter(tol, 0, 1, param_name='tol')

    def fit(self, X, y=None):
        """Fit detector. y is ignored in unsupervised methods.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.
        y : Ignored
            Not used, present for API consistency by convention.
        Returns
        -------
        self : object
            Fitted estimator.
        """
        try:
            X = check_array(X)
        except:
            X = X.reshape(-1, 1)
        # validate inputs X and y (optional)
        X = check_array(X)
        #self._set_n_classes(y)

        n_features = X.shape[1]
        self.hist_ = np.zeros([self.n_bins, n_features])
        self.bin_edges_ = np.zeros([self.n_bins + 1, n_features])

        # build the histograms for all dimensions
        for i in range(n_features):
            self.hist_[:, i], self.bin_edges_[:, i] = \
                np.histogram(X[:, i], bins=self.n_bins, density=True)
            # the sum of (width * height) should equal to 1
            assert (np.isclose(1, np.sum(
                self.hist_[:, i] * np.diff(self.bin_edges_[:, i])), atol=0.1))

        outlier_scores = _calculate_outlier_scores(X, self.bin_edges_,
                                                   self.hist_,
                                                   self.n_bins,
                                                   self.alpha, self.tol)

        # invert decision_scores_. Outliers comes with higher outlier scores
        self.decision_scores_ = invert_order(np.sum(outlier_scores, axis=1))
        return self

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.
        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.
        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        check_is_fitted(self, ['hist_', 'bin_edges_'])
        X = check_array(X)

        outlier_scores = _calculate_outlier_scores(X, self.bin_edges_,
                                                   self.hist_,
                                                   self.n_bins,
                                                   self.alpha, self.tol)
        return invert_order(np.sum(outlier_scores, axis=1))


def _calculate_outlier_scores(X, bin_edges, hist, n_bins, alpha,
                              tol):  # pragma: no cover
    """The internal function to calculate the outlier scores based on
    the bins and histograms constructed with the training data. The program
    is optimized through numba. It is excluded from coverage test for
    eliminating the redundancy.
    Parameters
    ----------
    X : numpy array of shape (n_samples, n_features)
        The input samples.
    bin_edges : numpy array of shape (n_bins + 1, n_features )
        The edges of the bins.
    hist : numpy array of shape (n_bins, n_features)
        The density of each histogram.
    n_bins : int, optional (default=10)
        The number of bins.
    alpha : float in (0, 1), optional (default=0.1)
        The regularizer for preventing overflow.
    tol : float in (0, 1), optional (default=0.1)
        The parameter to decide the flexibility while dealing
        the samples falling outside the bins.
    Returns
    -------
    outlier_scores : numpy array of shape (n_samples, n_features)
        Outlier scores on all features (dimensions).
    """

    n_samples, n_features = X.shape[0], X.shape[1]
    outlier_scores = np.zeros(shape=(n_samples, n_features))

    for i in range(n_features):

        # Find the indices of the bins to which each value belongs.
        # See documentation for np.digitize since it is tricky
        # >>> x = np.array([0.2, 6.4, 3.0, 1.6, -1, 100, 10])
        # >>> bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])
        # >>> np.digitize(x, bins, right=True)
        # array([1, 4, 3, 2, 0, 5, 4], dtype=int64)

        bin_inds = np.digitize(X[:, i], bin_edges[:, i], right=True)

        # Calculate the outlying scores on dimension i
        # Add a regularizer for preventing overflow
        out_score_i = np.log2(hist[:, i] + alpha)

        for j in range(n_samples):

            # If the sample does not belong to any bins
            # bin_ind == 0 (fall outside since it is too small)
            if bin_inds[j] == 0:
                dist = bin_edges[0, i] - X[j, i]
                bin_width = bin_edges[1, i] - bin_edges[0, i]

                # If it is only slightly lower than the smallest bin edge
                # assign it to bin 1
                if dist <= bin_width * tol:
                    outlier_scores[j, i] = out_score_i[0]
                else:
                    outlier_scores[j, i] = np.min(out_score_i)

            # If the sample does not belong to any bins
            # bin_ind == n_bins+1 (fall outside since it is too large)
            elif bin_inds[j] == n_bins + 1:
                dist = X[j, i] - bin_edges[-1, i]
                bin_width = bin_edges[-1, i] - bin_edges[-2, i]

                # If it is only slightly larger than the largest bin edge
                # assign it to the last bin
                if dist <= bin_width * tol:
                    outlier_scores[j, i] = out_score_i[n_bins - 1]
                else:
                    outlier_scores[j, i] = np.min(out_score_i)
            else:
                outlier_scores[j, i] = out_score_i[bin_inds[j] - 1]

    return outlier_scores

MAX_INT = np.iinfo(np.int32).max
MIN_INT = -1 * MAX_INT
def check_parameter(param, low=MIN_INT, high=MAX_INT, param_name='',
                    include_left=False, include_right=False):
    """Check if an input is within the defined range.
    Parameters
    ----------
    param : int, float
        The input parameter to check.
    low : int, float
        The lower bound of the range.
    high : int, float
        The higher bound of the range.
    param_name : str, optional (default='')
        The name of the parameter.
    include_left : bool, optional (default=False)
        Whether includes the lower bound (lower bound <=).
    include_right : bool, optional (default=False)
        Whether includes the higher bound (<= higher bound).
    Returns
    -------
    within_range : bool or raise errors
        Whether the parameter is within the range of (low, high)
    """

    # param, low and high should all be numerical
    if not isinstance(param, (numbers.Integral, np.integer, float)):
        raise TypeError('{param_name} is set to {param} Not numerical'.format(
            param=param, param_name=param_name))

    if not isinstance(low, (numbers.Integral, np.integer, float)):
        raise TypeError('low is set to {low}. Not numerical'.format(low=low))

    if not isinstance(high, (numbers.Integral, np.integer, float)):
        raise TypeError('high is set to {high}. Not numerical'.format(
            high=high))

    # at least one of the bounds should be specified
    if low is MIN_INT and high is MAX_INT:
        raise ValueError('Neither low nor high bounds is undefined')

    # if wrong bound values are used
    if low > high:
        raise ValueError(
            'Lower bound > Higher bound')

    # value check under different bound conditions
    if (include_left and include_right) and (param < low or param > high):
        raise ValueError(
            '{param_name} is set to {param}. '
            'Not in the range of [{low}, {high}].'.format(
                param=param, low=low, high=high, param_name=param_name))

    elif (include_left and not include_right) and (
            param < low or param >= high):
        raise ValueError(
            '{param_name} is set to {param}. '
            'Not in the range of [{low}, {high}).'.format(
                param=param, low=low, high=high, param_name=param_name))

    elif (not include_left and include_right) and (
            param <= low or param > high):
        raise ValueError(
            '{param_name} is set to {param}. '
            'Not in the range of ({low}, {high}].'.format(
                param=param, low=low, high=high, param_name=param_name))

    elif (not include_left and not include_right) and (
            param <= low or param >= high):
        raise ValueError(
            '{param_name} is set to {param}. '
            'Not in the range of ({low}, {high}).'.format(
                param=param, low=low, high=high, param_name=param_name))
    else:
        return True

from sklearn.utils import column_or_1d
def invert_order(scores, method='multiplication'):
    """ Invert the order of a list of values. The smallest value becomes
    the largest in the inverted list. This is useful while combining
    multiple detectors since their score order could be different.
    Parameters
    ----------
    scores : list, array or numpy array with shape (n_samples,)
        The list of values to be inverted
    method : str, optional (default='multiplication')
        Methods used for order inversion. Valid methods are:
        - 'multiplication': multiply by -1
        - 'subtraction': max(scores) - scores
    Returns
    -------
    inverted_scores : numpy array of shape (n_samples,)
        The inverted list
    """

    scores = column_or_1d(scores)

    if method == 'multiplication':
        return scores.ravel() * -1

    if method == 'subtraction':
        return (scores.max() - scores).ravel()
