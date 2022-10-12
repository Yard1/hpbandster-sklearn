from logging import DEBUG, error
from logging import INFO
from logging import WARNING
from time import time

from typing import Any, Callable, Dict, Iterable, List, Optional, Union
from joblib.parallel import Parallel, delayed

import numpy as np
from math import ceil

from sklearn.base import is_classifier, clone
from sklearn.utils import indexable, resample
from sklearn.metrics._scorer import _check_multimetric_scoring, check_scoring
from sklearn.model_selection._split import check_cv
from sklearn.model_selection._validation import (
    _fit_and_score,
    _aggregate_score_dicts,
    _normalize_score_results,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from .booster_integration import (
    is_catboost_model,
    is_lightgbm_model_of_required_version,
    is_xgboost_model,
)

# adapted from sklearn.model_selection._search_successive_halving
class _SubsampleMetaSplitter:
    """Splitter that subsamples a given fraction of the dataset"""

    def __init__(self, *, base_cv, fraction, subsample_test, random_state):
        self.base_cv = base_cv
        self.fraction = fraction
        self.subsample_test = subsample_test
        self.random_state = random_state

    def split(self, X, y, groups=None):
        for train_idx, test_idx in self.base_cv.split(X, y, groups):
            if self.fraction < 1:
                train_idx = resample(                                # when self.fraction=1, which means to use the entire train_idx.
                    train_idx,                                       # If sampling randomly, the order of samples will be changed, which means
                    replace=False,                                   # their corresponding samples are the same, but the order is different.
                    random_state=self.random_state,                  # This will affect model like random forest, which is sensitive to the sample order.
                    n_samples=int(ceil(self.fraction * train_idx.shape[0])),
                )
            if self.subsample_test:
                if self.fraction < 1:
                    test_idx = resample(
                        test_idx,
                        replace=False,
                        random_state=self.random_state,
                        n_samples=int(ceil(self.fraction * test_idx.shape[0])),
                    )
            yield train_idx, test_idx


# adapted from sklearn.model_selection._validation
# the estimator is not cloned but deepcopied to take advantage of warm_start
def _cross_validate_with_warm_start(
    estimators,
    X,
    y=None,
    *,
    groups=None,
    scoring=None,
    cv=None,
    n_jobs=None,
    verbose=0,
    fit_params=None,
    pre_dispatch="2*n_jobs",
    return_train_score=False,
    return_estimator=False,
    error_score=np.nan,
):
    """Evaluate metric(s) by cross-validation and also record fit/score times.

    Read more in the :ref:`User Guide <multimetric_cross_validation>`.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like of shape (n_samples, n_features)
        The data to fit. Can be for example a list, or an array.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs), \
            default=None
        The target variable to try to predict in the case of
        supervised learning.

    groups : array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting the dataset into
        train/test set. Only used in conjunction with a "Group" :term:`cv`
        instance (e.g., :class:`GroupKFold`).

    scoring : str, callable, list/tuple, or dict, default=None
        A single str (see :ref:`scoring_parameter`) or a callable
        (see :ref:`scoring`) to evaluate the predictions on the test set.

        For evaluating multiple metrics, either give a list of (unique) strings
        or a dict with names as keys and callables as values.

        NOTE that when using custom scorers, each scorer should return a single
        value. Metric functions returning a list/array of values can be wrapped
        into multiple scorers that return one value each.

        See :ref:`multimetric_grid_search` for an example.

        If None, the estimator's score method is used.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For int/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.

    n_jobs : int, default=None
        The number of CPUs to use to do the computation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    verbose : int, default=0
        The verbosity level.

    fit_params : dict, default=None
        Parameters to pass to the fit method of the estimator.

    pre_dispatch : int or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A str, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    return_train_score : bool, default=False
        Whether to include train scores.
        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.

        .. versionadded:: 0.19

        .. versionchanged:: 0.21
            Default value was changed from ``True`` to ``False``

    return_estimator : bool, default=False
        Whether to return the estimators fitted on each split.

        .. versionadded:: 0.20

    error_score : 'raise' or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised.
        If a numeric value is given, FitFailedWarning is raised. This parameter
        does not affect the refit step, which will always raise the error.

        .. versionadded:: 0.20

    Returns
    -------
    scores : dict of float arrays of shape (n_splits,)
        Array of scores of the estimator for each run of the cross validation.

        A dict of arrays containing the score/time arrays for each scorer is
        returned. The possible keys for this ``dict`` are:

            ``test_score``
                The score array for test scores on each cv split.
                Suffix ``_score`` in ``test_score`` changes to a specific
                metric like ``test_r2`` or ``test_auc`` if there are
                multiple scoring metrics in the scoring parameter.
            ``train_score``
                The score array for train scores on each cv split.
                Suffix ``_score`` in ``train_score`` changes to a specific
                metric like ``train_r2`` or ``train_auc`` if there are
                multiple scoring metrics in the scoring parameter.
                This is available only if ``return_train_score`` parameter
                is ``True``.
            ``fit_time``
                The time for fitting the estimator on the train
                set for each cv split.
            ``score_time``
                The time for scoring the estimator on the test set for each
                cv split. (Note time for scoring on the train set is not
                included even if ``return_train_score`` is set to ``True``
            ``estimator``
                The estimator objects for each cv split.
                This is available only if ``return_estimator`` parameter
                is set to ``True``.

    Examples
    --------
    >>> from sklearn import datasets, linear_model
    >>> from sklearn.model_selection import cross_validate
    >>> from sklearn.metrics import make_scorer
    >>> from sklearn.metrics import confusion_matrix
    >>> from sklearn.svm import LinearSVC
    >>> diabetes = datasets.load_diabetes()
    >>> X = diabetes.data[:150]
    >>> y = diabetes.target[:150]
    >>> lasso = linear_model.Lasso()

    Single metric evaluation using ``cross_validate``

    >>> cv_results = cross_validate(lasso, X, y, cv=3)
    >>> sorted(cv_results.keys())
    ['fit_time', 'score_time', 'test_score']
    >>> cv_results['test_score']
    array([0.33150734, 0.08022311, 0.03531764])

    Multiple metric evaluation using ``cross_validate``
    (please refer the ``scoring`` parameter doc for more information)

    >>> scores = cross_validate(lasso, X, y, cv=3,
    ...                         scoring=('r2', 'neg_mean_squared_error'),
    ...                         return_train_score=True)
    >>> print(scores['test_neg_mean_squared_error'])
    [-3635.5... -3573.3... -6114.7...]
    >>> print(scores['train_r2'])
    [0.28010158 0.39088426 0.22784852]

    See Also
    ---------
    :func:`sklearn.model_selection.cross_val_score`:
        Run cross-validation for single metric evaluation.

    :func:`sklearn.model_selection.cross_val_predict`:
        Get predictions from each split of cross-validation for diagnostic
        purposes.

    :func:`sklearn.metrics.make_scorer`:
        Make a scorer from a performance metric or loss function.

    """
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimators[0]))
    if callable(scoring):
        scorers = {"score": scoring}
    elif scoring is None or isinstance(scoring, str):
        scorers = {"score": check_scoring(estimators[0], scoring=scoring)}
    else:
        try:
            scorers = _check_multimetric_scoring(estimators[0], scoring=scoring)
            # sklearn < 0.24.0 compatibility
            if isinstance(scorers, tuple):
                scorers = scorers[0]
        except KeyError:
            pass

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.

    parallel = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)
    results_org = parallel(
        delayed(_fit_and_score)(
            estimators[i],
            X,
            y,
            scorers,
            train_test_tuple[0],
            train_test_tuple[1],
            verbose,
            None,
            fit_params[i] if isinstance(fit_params, list) else fit_params,
            return_train_score=return_train_score,
            return_times=True,
            return_n_test_samples=True,
            return_estimator=return_estimator,
            error_score=error_score,
        )
        for i, train_test_tuple in enumerate(cv.split(X, y, groups))
    )

    results = _aggregate_score_dicts(results_org)

    ret = {}
    ret["fit_time"] = results["fit_time"]
    ret["score_time"] = results["score_time"]

    if return_estimator:
        ret["estimator"] = results["estimator"]

    test_scores_dict = _normalize_score_results(results["test_scores"])
    if return_train_score:
        train_scores_dict = _normalize_score_results(results["train_scores"])

    for name in test_scores_dict:
        ret["test_%s" % name] = test_scores_dict[name]
        if return_train_score:
            key = "train_%s" % name
            ret[key] = train_scores_dict[name]

    return (ret, results_org)


class SklearnWorker(Worker):
    AUTO_BUDGET_PARAMS = {"n_estimators": int, "max_iters": int}

    def __init__(
        self,
        run_id,
        X,
        y,
        base_estimator,
        nameserver=None,
        nameserver_port=None,
        logger=None,
        host=None,
        id=None,
        timeout=None,
        min_budget=0,
        max_budget=1,
        scoring=None,
        warm_start=True,
        metric="score",
        resource_name=None,
        resource_type=None,
        cv=None,
        cv_n_splits=None,
        error_score=np.nan,
        fit_params=None,
        groups=None,
        return_train_score=False,
        random_state=None,
    ):
        """
        
        Parameters
        ----------
        run_id: anything with a __str__ method
            unique id to identify individual HpBandSter run
        nameserver: str
            hostname or IP of the nameserver
        nameserver_port: int
            port of the nameserver
        logger: logging.logger instance
            logger used for debugging output
        host: str
            hostname for this worker process
        id: anything with a __str__method
            if multiple workers are started in the same process, you MUST provide a unique id for each one of them using the `id` argument.
        timeout: int or float
            specifies the timeout a worker will wait for a new after finishing a computation before shutting down.
            Towards the end of a long run with multiple workers, this helps to shutdown idling workers. We recommend
            a timeout that is roughly half the time it would take for the second largest budget to finish.
            The default (None) means that the worker will wait indefinitely and never shutdown on its own.
        """
        super().__init__(
            run_id=run_id,
            nameserver=nameserver,
            nameserver_port=nameserver_port,
            logger=logger,
            host=host,
            id=id,
            timeout=timeout,
        )
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.base_estimator = base_estimator
        self.X = X
        self.y = y
        self.scoring = scoring
        self.warm_start = warm_start
        self.metric = metric
        self.cv = cv
        self.cv_n_splits = cv_n_splits
        self.error_score = error_score
        self.fit_params = fit_params
        self.groups = groups
        self.return_train_score = return_train_score
        self.resource_name = resource_name
        self.resource_type = resource_type
        self.random_state = random_state

        self._prepare_estimator()

    def _get_actual_estimator(self, estimator):
        if isinstance(estimator, Pipeline):
            estimator = estimator._final_estimator
        if isinstance(estimator, TransformedTargetRegressor):
            try:
                estimator = estimator.regressor_
            except:
                estimator = estimator.regressor
        return estimator

    @property
    def actual_base_estimator(self):
        return self._get_actual_estimator(self.base_estimator)

    @property
    def actual_estimators(self):
        return [self._get_actual_estimator(x) for x in self.estimators]

    @property
    def pipeline_estimator_name(self):
        if isinstance(self.base_estimator, Pipeline):
            return self.base_estimator.steps[-1][0]
        return ""

    @property
    def pipeline_estimator_name_prefix(self):
        prefix = self.pipeline_estimator_name
        if prefix:
            return f"{prefix}__"
        return ""

    @property
    def actual_estimator_name_prefix(self):
        estimator = self.base_estimator
        prefix = []
        if isinstance(estimator, Pipeline):
            prefix.append(estimator.steps[-1][0])
        if isinstance(estimator, TransformedTargetRegressor):
            prefix.append("regressor")
        if prefix:
            return f"{'__'.join(prefix)}__"
        return ""

    def _prepare_estimator(self):
        self.base_estimator = clone(self.base_estimator)

        if self.resource_name != "n_samples":
            try:
                if not is_catboost_model(self.actual_base_estimator):
                    self.base_estimator.set_params(
                        **{f"{self.actual_estimator_name_prefix}warm_start": self.warm_start}
                    )
            except:
                pass

        try:
            if not is_catboost_model(self.actual_base_estimator):
                self.base_estimator.set_params(
                    **{f"{self.actual_estimator_name_prefix}n_jobs": 1}
                )
            else:
                self.base_estimator.set_params(
                    **{f"{self.actual_estimator_name_prefix}thread_count": 1}
                )
        except:
            pass

        def is_resource_in_estimator(estimator, resource_name):
            if is_catboost_model(estimator) and resource_name == "n_estimators":
                return True
            return hasattr(estimator, resource_name)

        if not self.resource_name:
            for k, v in self.AUTO_BUDGET_PARAMS.items():
                if is_resource_in_estimator(self.actual_base_estimator, k):
                    self.resource_name = f"{self.actual_estimator_name_prefix}{k}"
                    self.resource_type = v
                    break
            if not self.resource_name:
                self.resource_name = "n_samples"

        if not self.resource_type:
            if self.min_budget is not None and type(self.min_budget) not in (
                float,
                int,
            ):
                raise TypeError(
                    f"'min_budget' must be of type 'float' or 'int', got '{type(self.min_budget)}'."
                )
            if self.max_budget is not None and type(self.max_budget) not in (
                float,
                int,
            ):
                raise TypeError(
                    f"'max_budget' must be of type 'float' or 'int', got '{type(self.max_budget)}'."
                )
            if (
                type(self.min_budget) is float
                or type(self.max_budget) is float
                or (self.min_budget == 0 and self.max_budget == 1)
            ):
                self.resource_type = float
            else:
                self.resource_type = int

        if self.resource_type not in (float, int):
            raise ValueError(
                f"'resource_type' must be 'float' or 'int', got '{self.resource_type}'."
            )

        if self.min_budget is None:
            if self.resource_name == "n_samples":
                self.min_budget = (
                    self.cv_n_splits * len(np.unique(self.y)) * 2
                    if is_classifier(self.actual_base_estimator)
                    else self.cv_n_splits * 2
                )
                if self.resource_type is float:
                    self.min_budget = float(self.min_budget / self.X.shape[0])
            elif self.resource_type is int:
                self.min_budget = 10
            else:
                raise ValueError(
                    "Couldn't automatically determine min_budget value. Please set min_budget explicitly."
                )

        if self.max_budget is None:
            if self.resource_name == "n_samples":
                if self.resource_type is float:
                    self.max_budget = 1.0
                else:
                    self.max_budget = self.X.shape[0]
            elif self.resource_type is int:
                self.max_budget = 100
            else:
                raise ValueError(
                    "Couldn't automatically determine max_budget value. Please set max_budget explicitly."
                )

        if self.max_budget < self.min_budget:
            raise ValueError(
                "max_budget {self.max_budget} is smaller than min_budget {self.min_budget}. Please set max_budget explicitly."
            )

        if self.resource_name != "n_samples":
            self.base_estimator.set_params(
                **{self.resource_name: self.resource_type(self.min_budget)}
            )

        self.estimators = [clone(self.base_estimator) for i in range(self.cv_n_splits)]

    def _get_booster_fit_params(self, estimator):
        booster = estimator
        if isinstance(estimator, Pipeline):
            estimator = estimator._final_estimator
        if isinstance(estimator, TransformedTargetRegressor):
            try:
                estimator = estimator.regressor_
            except:
                estimator = estimator.regressor
        fit_params = {}
        try:
            if is_catboost_model(estimator) and estimator.is_fitted():
                booster = estimator
                fit_params = {
                    f"{self.pipeline_estimator_name_prefix}init_model": booster
                }
            elif is_lightgbm_model_of_required_version(estimator):
                booster = estimator.booster_
                fit_params = {
                    f"{self.pipeline_estimator_name_prefix}init_model": booster
                }
            elif is_xgboost_model(estimator):
                booster = estimator.get_booster()
                fit_params = {
                    f"{self.pipeline_estimator_name_prefix}xgb_model": booster
                }
        except:
            pass
        return fit_params

    def compute(self, config_id, config, budget, working_directory):
        def try_set_resource_param(estimator, resource, budget):
            try:
                resource_type = self.resource_type
                new_resource = resource_type(budget)
                old_resource = estimator.get_params()[resource]
                if new_resource < old_resource:
                    estimator = clone(estimator)
                estimator.set_params(**{resource: new_resource})
            except Exception as e:
                return estimator, None
            return estimator, (resource, new_resource, resource_type(budget), budget)

        resources_set = (self.resource_name, None, None, budget)

        fit_params = []

        for i, estimator in enumerate(self.estimators):
            booster_fit_params = self._get_booster_fit_params(estimator)
            if is_catboost_model(estimator):
                estimator = clone(estimator)
            estimator.set_params(**config)
            if self.resource_name != "n_samples":
                new_estimator, resources_set = try_set_resource_param(
                    estimator, self.resource_name, budget,
                )
                if new_estimator is not estimator:
                    booster_fit_params = {}
                self.estimators[i] = new_estimator
            fit_params.append({**self.fit_params, **booster_fit_params})

        if self.resource_name == "n_samples":
            subsample_fraction = (
                budget
                if self.resource_type == float
                else self.resource_type(ceil(budget / self.X.shape[0]))
            )
            resources_set = (
                "n_samples",
                int(ceil(subsample_fraction * self.X.shape[0])),
                self.resource_type(budget),
                budget,
            )
            cv = _SubsampleMetaSplitter(
                base_cv=self.cv,
                fraction=subsample_fraction,
                subsample_test=True,
                random_state=self.random_state,
            )
        else:
            cv = self.cv

        #print(self.estimators[0].get_params())
        ret, scores = _cross_validate_with_warm_start(
            self.estimators,
            self.X,
            self.y,
            cv=cv,
            error_score=self.error_score,
            fit_params=fit_params,
            groups=self.groups,
            return_train_score=self.return_train_score,
            scoring=self.scoring,
            return_estimator=True,
        )

        self.estimators = list(ret.pop("estimator"))

        test_score_mean = np.mean(ret[f"test_{self.metric}"])
        try:
            train_score_mean = np.mean(ret[f"train_{self.metric}"])
        except KeyError:
            train_score_mean = None

        del ret

        result = {
            # this is the a mandatory field to run hyperband
            # remember: HpBandSter always minimizes!
            # and sklearn always maximizes, therefore *-1.0
            "loss": float(test_score_mean * -1.0),
            "info": {
                "test_score_mean": test_score_mean,
                "train_score_mean": train_score_mean,
                "resources": resources_set,
            },
        }

        # remove estimators
        def numpy_to_native(x):
            try:
                assert "numpy" in str(type(x))
                return x.item()
            except:
                return x

        def array_to_list(x):
            if isinstance(x, np.ndarray):
                return [numpy_to_native(y) for y in x]
            return x

        result["info"]["cv"] = [
            {k: array_to_list(v) for k, v in score.items() if k != "estimator"}
            for score in scores
        ]

        self.logger.debug(result)

        return result
