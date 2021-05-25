import logging
import numbers
import os
import time
import gc
from copy import deepcopy
from collections import Counter
from time import sleep

from joblib.parallel import cpu_count

import numpy as np

from sklearn.base import clone, is_classifier
from sklearn.model_selection._search import BaseSearchCV
from sklearn.utils import check_random_state
from sklearn.model_selection._split import check_cv
from sklearn.metrics._scorer import _check_multimetric_scoring, check_scoring
from sklearn.utils.validation import _check_fit_params, indexable

import hpbandster.core.nameserver as hpns
from hpbandster.optimizers import BOHB, RandomSearch, HyperBand, H2BO

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from .sklearn_worker import SklearnWorker
from .context import NameServerContext, OptimizerContext

_logger = logging.getLogger(__name__)
_logger.addHandler(logging.StreamHandler())


class HpBandSterSearchCV(BaseSearchCV):
    """
    Hyper parameter search using HpBandSter.

    This class provides a scikit-learn compatible wrapper over
    HpBandSter, implementing the entire HpBandSter search process
    (:class:`Nameserver`, :class:`Worker`, :class:`Optimizer`).

    In addition to scikit-learn estimators, early stopping support is built in for
    :class:`LightGBM`, :class:`XGBoost` and :class:`CatBoost` estimators.

    Parameters
    ----------
    estimator : estimator object
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    param_distributions : dict or ConfigurationSpace
        Either a ConfigurationSpace object or a dictionary with parameters names (string)
        as keys and lists of parameters to try. If a list is given, it is sampled uniformly.
        Using a ConfigurationSpace object is highly recommended. Refer to ConfigSpace documentation.

    n_iter : int, default=10
        The number of optimizer iterations to perform.

    optimizer : str or Optimizer type, default='bohb'
        The HpBandSter optimizer to use. Can be either an Optimizer type (not object!), or one of the
        folowing strings representing a HpBandSter optimizer.

            * 'bohb' - :class:`BOHB`
            * 'random' or 'randomsearch' - :class:`RandomSearch`
            * 'hyperband' - :class:`HyperBand`
            * 'h2bo' - :class:`H2BO`

    nameserver_host : str, default='127.0.0.1'
        The hostname to use for the HpBandSter nameserver. Required even when ran locally.

    nameserver_port : int, default=9090
        The port to use for the HpBandSter nameserver. Required even when ran locally.

    min_budget : int or float, default=None
        The minimum budget (amount of resource) to consider. Must be bigger than 0.
        If ``None``, will be:

            * ``n_splits * 2`` when ``resource_name='n_samples'`` for a regression problem
            * ``n_classes * n_splits * 2`` when ``resource_name='n_samples'`` for a classification problem
            * ``10`` when ``resource_name != 'n_samples'``

        If ``resource_name`` is or is determined to be ``n_samples``, an int will
        translate to that many samples in the dataset and float will translate to that big
        fraction of a dataset.

    max_budget : int or float, default=None
        The maximum budget (amount of resource) to consider. Must be bigger than 0 and min_budget.

        If ``None``, will be:

            * ``n_samples`` (the size ``X`` passed in ``fit``) when ``resource_name='n_samples'``
            * ``100`` when ``resource_name != 'n_samples'``

        If ``resource_name`` is or is determined to be ``n_samples``, an int will
        translate to that many samples in the dataset and float will translate to that big
        fraction of a dataset.

    resource_name : 'n_samples', str or ``None``, default=None
        Defines the name of the resource to be increased with each iteration.
        If ``None`` (default), the resource name will be automatically determined to be one of
        (in order):

            * 'n_estimators' - if estimator posses that attribute and has ``warm_start`` attribute
            * 'max_iter' - if estimator posses that attribute and has ``warm_start`` attribute
            * 'n_samples' - the number/fraction of samples

        'n_estimators' will also be used for :class:`LightGBM`, :class:`XGBoost` and :class:`CatBoost` estimators.

    resource_type : type, default=None
        Defines the Python type of resource - either :class:`int` or :class:`float`. If ``None``,
        (default), will try to automatically determine the type based on ``resource_name``,
        ``min_budget`` and ```max_budget```.

    cv : int, cross-validation generator or an iterable, default=5
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - `CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

        .. note::
            Due to implementation details, the folds produced by `cv` must be
            the same across multiple calls to `cv.split()`. For
            built-in `scikit-learn` iterators, this can be achieved by
            deactivating shuffling (`shuffle=False`), or by setting the
            `cv`'s `random_state` parameter to an integer.

    scoring : str, callable, or None, default=None
        A single string (see `scoring_parameter`) or a callable
        (see `scoring`) to evaluate the predictions on the test set.
        If None, the estimator's score method is used.

    refit : bool, default=True
        If True, refit an estimator using the best found parameters on the
        whole dataset. The estimator will be refit with the maximum amount of
        the resource.

        The refitted estimator is made available at the ``best_estimator_``
        attribute and permits using ``predict`` directly on this
        ``GridSearchCV`` instance.

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.

    return_train_score : bool, default=False
        If ``False``, the ``cv_results_`` attribute will not include training
        scores.
        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.

    random_state : int, RandomState instance or None, default=None
        Pseudo random number generator state used for subsampling the dataset
        when `resources != 'n_samples'`. Also used for random uniform
        sampling from lists of possible values instead of scipy.stats
        distributions.
        Pass an int for reproducible output across multiple function calls.

    n_jobs : int or None, default=None
        Number of workers to spawn. Each worker runs in a separate thread.
        ``None`` means 1. ``-1`` means maxiumum amount of cores, ``-2`` means one less
        than maximum and so on. If ``LOKY_MAX_CPU_COUNT`` OS enviromental variables
        is set, it will be used as the maximum number of CPU cores. Otherwise,
        for better performance, if ``psutil`` is installed the maximum value will be the
        number of physical CPU cores. Otherwise, the number of logical CPU cores
        will be used.

    verbose : int
        Controls the verbosity: the higher, the more messages.

    **kwargs
        Keyword arguments to be passed to the Optimizer. Refer to HpBandSter documentation.

    Attributes
    ----------
    n_resources_ : list of int or float
        The amount of resources used at each iteration.

    n_candidates_ : list of int
        The number of candidate parameters that were evaluated at each
        iteration.

    n_remaining_candidates_ : int
        The number of candidate parameters that are left after the last
        iteration. It corresponds to `ceil(n_candidates[-1] / factor)`

    max_resources_ : int or float
        The maximum number of resources that any candidate is allowed to use
        for a given iteration. Note that since the number of resources used at
        each iteration must be a multiple of ``min_resources_``, the actual
        number of resources used at the last iteration may be smaller than
        ``max_resources_``.

    min_resources_ : int or float
        The amount of resources that are allocated for each candidate at the
        first iteration.

    resource_name_ : str
        The name of the resource.

    n_iterations_ : int
        The actual number of iterations that were run.

    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``. It contains many informations for
        analysing the results of a search.

    best_estimator_ : estimator or dict
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if ``refit=False``.

    best_score_ : float
        Mean cross-validated score of the best_estimator.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.

        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).

    scorer_ : function or a dict
        Scorer function used on the held out data to choose the best
        parameters for the model.

    n_splits_ : int
        The number of cross-validation splits (folds/iterations).

    refit_time_ : float
        Seconds used for refitting the best model on the whole dataset.

        This is present only if ``refit`` is not False.

    Notes
    -----
    The parameters selected are those that maximize the score of the held-out
    data, according to the scoring parameter.

    Examples
    --------

    >>> from sklearn.datasets import load_iris
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from hpbandster_sklearn import HpBandSterSearchCV
    ...
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = RandomForestClassifier(random_state=0)
    >>> np.random.seed(0)
    ...
    >>> param_distributions = {"max_depth": [3, 4],
    ...                        "min_samples_split": list(range(2, 12))}
    >>> search = HpBandSterSearchCV(clf, param_distributions,
    ...                                resource_name='n_estimators',
    ...                                random_state=0, n_jobs=1).fit(X, y)
    >>> search.best_params_  # doctest: +SKIP
    """

    _required_parameters = ["estimator", "param_distributions"]

    _optimizer_dict = {
        "bohb": BOHB,
        "random": RandomSearch,
        "randomsearch": RandomSearch,
        "hyperband": HyperBand,
        "h2bo": H2BO,
    }

    def __init__(
        self,
        estimator,
        param_distributions,
        *,
        n_iter=10,
        optimizer="bohb",
        nameserver_host="127.0.0.1",
        nameserver_port=9090,
        min_budget=None,
        max_budget=None,
        resource_name=None,
        resource_type=None,
        cv=None,
        scoring=None,
        refit=True,
        error_score=np.nan,
        return_train_score=False,
        random_state=None,
        n_jobs=None,
        verbose=0,
        **kwargs,
    ):
        if not isinstance(optimizer, (str, type)):
            raise TypeError(
                f"'optimizer' must be of type 'str' or 'type' (an Optimizer class), got '{type(optimizer)}."
            )
        elif isinstance(optimizer, str) and optimizer not in self._optimizer_dict:
            raise ValueError(
                f"'optimizer' must be one of: {', '.join(self._optimizer_dict.keys())}."
            )

        if min_budget is not None and min_budget < 0:
            raise ValueError(f"min_budget cannot be negative. Got {min_budget}.")

        if max_budget is not None and max_budget < 0:
            raise ValueError(f"max_budget cannot be negative. Got {min_budget}.")

        if (
            max_budget is not None
            and min_budget is not None
            and max_budget <= min_budget
        ):
            raise ValueError(
                f"max_budget {max_budget} must be bigger than min_budget {min_budget}."
            )

        self.n_iter = n_iter
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.resource_name = resource_name
        self.resource_type = resource_type
        self.random_state = random_state
        self.optimizer = optimizer
        self.nameserver_port = nameserver_port
        self.nameserver_host = nameserver_host
        self.bohb_kwargs = kwargs
        self.param_distributions = self._param_distributions_check(param_distributions)
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            error_score=error_score,
            return_train_score=return_train_score,
        )

        self._res = None

    def _param_distributions_check(self, param_distributions):
        if not isinstance(param_distributions, (dict, CS.ConfigurationSpace)):
            raise TypeError(
                f"param_distributions must be of 'dict' or 'ConfigurationSpace' type, got {type(param_distributions)}."
            )

        if isinstance(param_distributions, dict):
            cs = CS.ConfigurationSpace()
            for k, v in param_distributions.items():
                if not isinstance(k, str):
                    raise TypeError(
                        f"If param_distributions is a dict, all keys must be str. Key '{k}' was of type {type(k)}."
                    )
                if not isinstance(v, list):
                    raise TypeError(
                        f"If param_distributions is a dict, all values must be lists. Value at key '{k}' was of type {type(v)}."
                    )
                cs.add_hyperparameter(CSH.CategoricalHyperparameter(k, v))
            param_distributions = cs

        return param_distributions

    def _runs_to_results(self, runs, id2config, scorers, n_splits, n_resources):
        all_candidate_params = []
        all_out = []
        resources = []
        iteration = []
        config_ids = []

        for run in runs:
            all_candidate_params.append(id2config[run.config_id]["config"])
            all_out.extend(run.info["cv"])

            resources.append(run.info["resources"][1])
            iteration.append(n_resources.index(run.info["resources"][2]))
            config_ids.append(run.config_id[-1])
        #print(all_out)
        results = list(
            self._format_results(all_candidate_params, n_splits, all_out).items()
        )
        results.insert(0, ("n_resources", resources))
        results.insert(0, ("iter", iteration))
        results.insert(0, ("run", config_ids))

        # multimetric is determined here because in the case of a callable
        # self.scoring the return type is only known after calling
        first_test_score = all_out[0]["test_scores"]
        self.multimetric_ = isinstance(first_test_score, dict)

        refit_metric = None

        # check refit_metric now for a callabe scorer that is multimetric
        if callable(self.scoring) and self.multimetric_:
            self._check_refit_for_multimetric(first_test_score)
            refit_metric = self.refit

        return dict(results), refit_metric

    def _calculate_n_jobs_and_actual_iters(self):
        # because HpBandSter assigns n_iter jobs to each worker, we need to divide
        n_jobs = self.n_jobs
        if not n_jobs:
            n_jobs = 1
        elif n_jobs < 0:
            try:
                import psutil

                cpus = int(
                    os.environ.get(
                        "LOKY_MAX_CPU_COUNT", psutil.cpu_count(logical=False)
                    )
                )
            except:
                cpus = cpu_count()
            n_jobs = max(cpus + 1 + n_jobs, 1)

        if n_jobs > self.n_iter:
            n_jobs = self.n_iter

        actual_iterations = self.n_iter // n_jobs + (self.n_iter % n_jobs > 0)
        return (n_jobs, actual_iterations)

    def fit(self, X, y, groups=None, **fit_params):
        # sklearn prep
        cv = check_cv(self.cv, y, classifier=is_classifier(self.estimator))
        refit_metric = "score"

        if callable(self.scoring):
            scorers = self.scoring
        elif self.scoring is None or isinstance(self.scoring, str):
            scorers = check_scoring(self.estimator, self.scoring)
        else:
            scorers = _check_multimetric_scoring(self.estimator, self.scoring)
            # sklearn < 0.24.0 compatibility
            if isinstance(scorers, tuple):
                scorers = scorers[0]

            self._check_refit_for_multimetric(scorers)
            refit_metric = self.refit

        X, y, groups = indexable(X, y, groups)
        fit_params = _check_fit_params(X, fit_params)
        n_splits = cv.get_n_splits(X, y, groups)
        base_estimator = clone(self.estimator)
        rng = check_random_state(self.random_state)
        np.random.set_state(rng.get_state(legacy=True))
        np_random_seed = rng.get_state(legacy=True)[1][0]

        n_jobs, actual_iterations = self._calculate_n_jobs_and_actual_iters()

        # default port is 9090, we must have one, this is how BOHB workers communicate (even locally)
        run_id = f"HpBandSterSearchCV_{time.time()}"
        _nameserver = hpns.NameServer(
            run_id=run_id, host=self.nameserver_host, port=self.nameserver_port
        )

        gc.collect()

        if self.verbose > 1:
            _logger.setLevel(logging.DEBUG)
        elif self.verbose > 0:
            _logger.setLevel(logging.INFO)
        else:
            _logger.setLevel(logging.ERROR)

        if "logger" in self.bohb_kwargs:
            self.bohb_kwargs.pop("logger")

        with NameServerContext(_nameserver):
            workers = []
            # each worker is a separate thread
            for i in range(n_jobs):
                # SklearnWorker clones the estimator
                w = SklearnWorker(
                    min_budget=self.min_budget,
                    max_budget=self.max_budget,
                    base_estimator=self.estimator,
                    X=X,
                    y=y,
                    cv=cv,
                    cv_n_splits=n_splits,
                    groups=groups,
                    scoring=scorers,
                    metric=refit_metric,
                    fit_params=fit_params,
                    nameserver=self.nameserver_host,
                    nameserver_port=self.nameserver_port,
                    run_id=run_id,
                    id=i,
                    return_train_score=self.return_train_score,
                    error_score=self.error_score,
                    resource_name=self.resource_name,
                    resource_type=self.resource_type,
                    random_state=rng,
                    logger=_logger,
                )
                w.run(background=True)
                workers.append(w)

            converted_min_budget = float(workers[0].min_budget)
            converted_max_budget = float(workers[0].max_budget)
            self.resource_name_ = workers[0].resource_name

            if (
                self.resource_name_
                in self.param_distributions.get_hyperparameter_names()
            ):
                _logger.warning(
                    f"Found hyperparameter with name '{self.resource_name_}', same as resource_name_. Removing it from ConfigurationSpace."
                )
                param_distributions = CS.ConfigurationSpace(
                    name=self.param_distributions.name,
                    meta=self.param_distributions.meta,
                )
                param_distributions.add_hyperparameters(
                    [
                        x
                        for x in self.param_distributions.get_hyperparameters()
                        if x.name != self.resource_name_
                    ]
                )
            else:
                param_distributions = deepcopy(self.param_distributions)
            param_distributions.seed = np_random_seed

            # sleep for a moment to make sure all workers are initialized
            sleep(0.2)

            # BOHB by default
            if isinstance(self.optimizer, str):
                optimizer = self._optimizer_dict[self.optimizer.lower()](
                    configspace=param_distributions,
                    run_id=run_id,
                    min_budget=converted_min_budget,
                    max_budget=converted_max_budget,
                    logger=_logger,
                    **self.bohb_kwargs,
                )
            else:
                optimizer = self.optimizer(
                    configspace=param_distributions,
                    run_id=run_id,
                    min_budget=converted_min_budget,
                    max_budget=converted_max_budget,
                    logger=_logger,
                    **self.bohb_kwargs,
                )
            with OptimizerContext(optimizer, n_iterations=actual_iterations,) as res:
                self._res = res

        id2config = self._res.get_id2config_mapping()
        incumbent = self._res.get_incumbent_id()
        runs_all = self._res.get_all_runs()
        self.best_params_ = id2config[incumbent]["config"]

        resource_type = workers[0].resource_type
        self.n_resources_ = [resource_type(x) for x in optimizer.budgets]
        self.min_resources_ = self.n_resources_[0]
        self.max_resources_ = self.n_resources_[-1]

        results, new_refit_metric = self._runs_to_results(
            runs_all, id2config, scorers, n_splits, self.n_resources_
        )

        if new_refit_metric is not None:
            refit_metric = new_refit_metric

        iter_counter = sorted(Counter(results["iter"]).items())
        self.n_candidates_ = [x[1] for x in iter_counter]
        self.n_remaining_candidates_ = iter_counter[-1][1]
        self.n_iterations_ = iter_counter[-1][0] + 1

        # For multi-metric evaluation, store the best_index_, best_params_ and
        # best_score_ iff refit is one of the scorer names
        # In single metric evaluation, refit_metric is "score"
        if self.refit or not self.multimetric_:
            # If callable, refit is expected to return the index of the best
            # parameter set.
            if callable(self.refit):
                self.best_index_ = self.refit(results)
                if not isinstance(self.best_index_, numbers.Integral):
                    raise TypeError("best_index_ returned is not an integer")
                if self.best_index_ < 0 or self.best_index_ >= len(results["params"]):
                    raise IndexError("best_index_ index out of range")
            else:
                self.best_index_ = results["rank_test_%s" % refit_metric].argmin()
                self.best_score_ = results["mean_test_%s" % refit_metric][
                    self.best_index_
                ]
            self.best_params_ = results["params"][self.best_index_]

        _logger.info(
            f"\nBest {refit_metric}: {self._res.get_runs_by_id(incumbent)[-1].info['test_score_mean']}"
        )
        _logger.info(f"Best found configuration: {self.best_params_}")
        _logger.info(
            f"A total of {len(id2config.keys())} unique configurations where sampled."
        )
        _logger.info(f"A total of {len(runs_all)} runs where executed.")
        _logger.info(
            f"Total budget of resource '{self.resource_name_}' corresponds to {sum([r.budget for r in runs_all]) / converted_max_budget} full function evaluations."
        )

        gc.collect()

        if self.refit:
            # we clone again after setting params in case some
            # of the params are estimators as well.
            refit_params = self.best_params_.copy()
            if self.resource_name_ != "n_samples":
                refit_params[self.resource_name_] = self.max_resources_
            self.best_estimator_ = clone(
                clone(base_estimator).set_params(**refit_params)
            )
            refit_start_time = time.time()
            if y is not None:
                self.best_estimator_.fit(X, y, **fit_params)
            else:
                self.best_estimator_.fit(X, **fit_params)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time

        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers

        self.cv_results_ = results
        self.n_splits_ = n_splits

        return self
