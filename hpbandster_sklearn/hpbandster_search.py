from logging import DEBUG
from logging import INFO
from logging import WARNING
import numbers
import os
import time
import gc
from collections import Counter

from typing import Any, Callable, Dict, Iterable, List, Optional, Union
from joblib.parallel import cpu_count

import numpy as np

from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone, is_classifier
from sklearn.model_selection._search import BaseSearchCV
from sklearn.model_selection._split import check_cv
from sklearn.metrics._scorer import _check_multimetric_scoring

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB, RandomSearch
from sklearn.utils.validation import _check_fit_params, indexable

from .internal import SklearnWorker
from .context import NameServerContext, OptimizerContext


class HpBandSterSearchCV(BaseSearchCV):
    def __init__(
        self,
        estimator,
        param_distributions,
        *,
        n_iter=10,
        min_budget=10,
        max_budget=100,
        scoring=None,
        n_jobs=None,
        iid="deprecated",
        refit=True,
        cv=None,
        verbose=0,
        pre_dispatch="2*n_jobs",
        random_state=None,
        error_score="raise",
        return_train_score=False,
        local_port=9090,
        host="127.0.0.1",
        **kwargs,
    ):
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.random_state = random_state
        self._nameserver = None
        self.optimizer = None
        self._res = None
        self._nameserver_port = local_port
        self._nameserver_host = host
        self.bohb_kwargs = kwargs
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            iid=iid,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )

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

        results = list(
            self._format_results(
                all_candidate_params, scorers, n_splits, all_out
            ).items()
        )
        results.insert(0, ("n_resources", resources))
        results.insert(0, ("iter", iteration))
        results.insert(0, ("run", config_ids))
        return dict(results)

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

        scorers, self.multimetric_ = _check_multimetric_scoring(
            self.estimator, scoring=self.scoring
        )

        if self.multimetric_:
            if (
                self.refit is not False
                and (
                    not isinstance(self.refit, str)
                    or
                    # This will work for both dict / list (tuple)
                    self.refit not in scorers
                )
                and not callable(self.refit)
            ):
                raise ValueError(
                    "For multi-metric scoring, the parameter "
                    "refit must be set to a scorer key or a "
                    "callable to refit an estimator with the "
                    "best parameter setting on the whole "
                    "data and make the best_* attributes "
                    "available for that metric. If this is "
                    "not needed, refit should be set to "
                    "False explicitly. %r was passed." % self.refit
                )
            else:
                refit_metric = self.refit
        else:
            refit_metric = "score"

        X, y, groups = indexable(X, y, groups)
        fit_params = _check_fit_params(X, fit_params)
        n_splits = cv.get_n_splits(X, y, groups)
        base_estimator = clone(self.estimator)

        # this should never be necessary, but just in case
        try:
            self._nameserver.shutdown()
        except:
            pass

        n_jobs, actual_iterations = self._calculate_n_jobs_and_actual_iters()

        # default port is 9090, we must have one, this is how BOHB workers communicate (even locally)
        run_id = f"HpBandSterSearchCV_{time.time()}"
        self._nameserver = hpns.NameServer(
            run_id=run_id, host=self._nameserver_host, port=self._nameserver_port
        )
        gc.collect()
        with NameServerContext(self._nameserver):
            workers = []
            # each worker is a separate thread
            for i in range(n_jobs):
                # SklearnWorker clones the estimator
                w = SklearnWorker(
                    min_budget=self.min_budget,
                    base_estimator=self.estimator,
                    X=X,
                    y=y,
                    cv=self.cv,
                    cv_n_splits=n_splits,
                    groups=groups,
                    scoring=scorers,
                    metric=refit_metric,
                    fit_params=fit_params,
                    nameserver=self._nameserver_host,
                    nameserver_port=self._nameserver_port,
                    run_id=run_id,
                    id=i,
                    return_train_score=self.return_train_score,
                    error_score=self.error_score,
                )
                w.run(background=True)
                workers.append(w)

            # BOHB by default
            if not self.optimizer:
                self.optimizer = BOHB(
                    configspace=self.param_distributions,
                    run_id=run_id,
                    min_budget=self.min_budget,
                    max_budget=self.max_budget,
                    **self.bohb_kwargs,
                )
            with OptimizerContext(
                self.optimizer, n_iterations=actual_iterations,
            ) as res:
                self._res = res

        id2config = self._res.get_id2config_mapping()
        incumbent = self._res.get_incumbent_id()
        runs_all = self._res.get_all_runs()
        self.best_params_ = id2config[incumbent]["config"]

        resource_type = workers[0].resource_type
        self.n_resources_ = [resource_type(x) for x in self.optimizer.budgets]
        self.min_resources_ = self.n_resources_[0]
        self.max_resources_ = self.n_resources_[-1]

        print(
            f"Best {refit_metric}: {self._res.get_runs_by_id(incumbent)[-1].info['test_score_mean']}"
        )
        print("Best found configuration:", self.best_params_)
        print(
            f"A total of {len(id2config.keys())} unique configurations where sampled."
        )
        print(f"A total of {len(runs_all)} runs where executed.")
        print(
            f"Total budget corresponds to {sum([r.budget for r in runs_all]) / self.max_budget} full function evaluations."
        )

        results = self._runs_to_results(
            runs_all, id2config, scorers, n_splits, self.n_resources_
        )

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

        gc.collect()

        if self.refit:
            # we clone again after setting params in case some
            # of the params are estimators as well.
            self.best_estimator_ = clone(
                clone(base_estimator).set_params(**self.best_params_)
            )
            refit_start_time = time.time()
            if y is not None:
                self.best_estimator_.fit(X, y, **fit_params)
            else:
                self.best_estimator_.fit(X, **fit_params)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time

        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers if self.multimetric_ else scorers["score"]
        self.cv_results_ = results
        self.n_splits_ = n_splits

        return self
