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
