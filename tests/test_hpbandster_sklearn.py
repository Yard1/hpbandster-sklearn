import pytest


def test_default():
    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.utils.validation import check_is_fitted
    from hpbandster_sklearn import HpBandSterSearchCV

    X, y = load_iris(return_X_y=True)
    clf = RandomForestClassifier(random_state=0)
    np.random.seed(0)

    param_distributions = {"max_depth": [3, 4], "min_samples_split": list(range(2, 12))}

    search = HpBandSterSearchCV(
        clf,
        param_distributions,
        random_state=0,
        n_jobs=1,
        n_iter=4,
        verbose=1,
        error_score="raise",
    ).fit(X, y)
    assert isinstance(search.best_estimator_, RandomForestClassifier)
    check_is_fitted(search.best_estimator_)


def test_CS():
    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.utils.validation import check_is_fitted
    from hpbandster_sklearn import HpBandSterSearchCV

    X, y = load_iris(return_X_y=True)
    clf = RandomForestClassifier(random_state=0)
    np.random.seed(0)

    import ConfigSpace as CS
    import ConfigSpace.hyperparameters as CSH

    param_distributions = CS.ConfigurationSpace(seed=42)
    param_distributions.add_hyperparameter(
        CSH.UniformIntegerHyperparameter("min_samples_split", 2, 10)
    )
    param_distributions.add_hyperparameter(
        CSH.UniformIntegerHyperparameter("min_samples_leaf", 1, 5)
    )
    param_distributions.add_hyperparameter(
        CSH.UniformIntegerHyperparameter("max_depth", 2, 10)
    )
    param_distributions.add_hyperparameter(
        CSH.UniformFloatHyperparameter(
            "min_impurity_decrease", 0.000000001, 0.5, log=True
        )
    )
    param_distributions.add_hyperparameter(
        CSH.UniformFloatHyperparameter("max_features", 0.01, 1)
    )

    search = HpBandSterSearchCV(
        clf,
        param_distributions,
        random_state=0,
        resource_name="n_samples",
        n_jobs=1,
        n_iter=4,
        verbose=1,
        error_score="raise",
    ).fit(X, y)
    assert isinstance(search.best_estimator_, RandomForestClassifier)
    check_is_fitted(search.best_estimator_)


def test_resource_name():
    """
    Created on Sun Oct  9 21:43:41 2022

    @author: poetair
    """
    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from hpbandster_sklearn import HpBandSterSearchCV
    from sklearn.model_selection import KFold, cross_val_score

    import ConfigSpace as CS
    import ConfigSpace.hyperparameters as CSH

    X, y = load_iris(return_X_y=True)
    clf = RandomForestClassifier(n_estimators=10, random_state=2)
    kf = KFold(shuffle=True, random_state=1)

    param_distributions = CS.ConfigurationSpace(seed=1111)
    param_distributions.add_hyperparameter(
        CSH.UniformIntegerHyperparameter("min_samples_split", 2, 11)
    )
    param_distributions.add_hyperparameter(
        CSH.UniformIntegerHyperparameter("max_depth", 2, 4)
    )

    ##### test resource_name='n_estimators'

    search = HpBandSterSearchCV(
        clf,
        param_distributions,
        resource_name="n_estimators",
        min_budget=10,
        max_budget=80,
        cv=kf,
        random_state=2,
        warm_start=False,
        refit=True,
        n_iter=4,
        **{"eta": 2}
    ).fit(X, y)

    # if cross_val_score is equal to mean_test_score with same kf and same params
    # The search process and the test process have same results.
    for p, s, c in zip(
        search.cv_results_["params"],
        search.cv_results_["n_resources"],
        search.cv_results_["mean_test_score"],
    ):
        clf = clf.set_params(**p)
        test_score = np.mean(cross_val_score(clf, X, y, cv=kf))
        assert test_score - c == 0

    ##### test resource_name='n_samples'
    max_budget = 1
    search = HpBandSterSearchCV(
        clf,
        param_distributions,
        resource_name="n_samples",
        min_budget=0.1,
        max_budget=max_budget,
        cv=kf,
        random_state=2,
        warm_start=False,
        refit=True,
        n_iter=4,
        **{"eta": 2}
    ).fit(X, y)

    # if cross_val_score is equal to mean_test_score with same kf and same params
    # The search process and the test process have same results.
    for p, s, c in zip(
        search.cv_results_["params"],
        search.cv_results_["n_resources"],
        search.cv_results_["mean_test_score"],
    ):
        if s == len(X) * max_budget:
            clf = clf.set_params(**p)
            test_score = np.mean(cross_val_score(clf, X, y, cv=kf))
            assert test_score - c == 0
