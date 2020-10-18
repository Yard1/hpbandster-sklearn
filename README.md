# hpbandster-sklearn

`hpbandster-sklearn` is a Python library providing a [`scikit-learn`](http://scikit-learn.org/) wrapper - `HpBandSterSearchCV` - for [`HpBandSter`](https://github.com/automl/HpBandSter), a hyper parameter tuning library.

## Motivation

`HpBandSter` implements several cutting-edge hyper parameter algorithms, including HyperBand and BOHB. They often outperform standard Random Search, finding best parameter combinations in less time.

`HpBandSter` is powerful and configurable, but its usage is often unintuitive for beginners and necessitating a large amount of boilerplate code. In order to solve that issue, `HpBandSterSearchCV` was created as a drop-in replacement for `scikit-learn` hyper parameter searchers, following its well-known and popular API, making it possible to tune `scikit-learn` API estimators with minimal setup.

`HpBandSterSearchCV` API has been based on `scikit-learn`'s [`HalvingRandomSearchCV`](https://scikit-learn.org/dev/modules/generated/sklearn.model_selection.HalvingRandomSearchCV.html), implementing nearly all of the parameters it does.

## Installation

```
pip install hpbandster-sklearn
```

## Usage

Use it like any other `scikit-learn` hyper parameter searcher:

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted
from hpbandster_sklearn import HpBandSterSearchCV

X, y = load_iris(return_X_y=True)
clf = RandomForestClassifier(random_state=0)
np.random.seed(0)

param_distributions = {"max_depth": [2, 3, 4], "min_samples_split": list(range(2, 12))}

search = HpBandSterSearchCV(clf, param_distributions,random_state=0, n_jobs=1, n_iter=10, verbose=1).fit(X, y)
search.best_params_
```

You can also use `ConfigSpace.ConfigurationSpace` objects instead of dicts (in fact, it is recommended)!

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted
from hpbandster_sklearn import HpBandSterSearchCV
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

X, y = load_iris(return_X_y=True)
clf = RandomForestClassifier(random_state=0)
np.random.seed(0)

param_distributions = CS.ConfigurationSpace(seed=42)
param_distributions.add_hyperparameter(CSH.UniformIntegerHyperparameter("min_samples_split", 2, 11))
param_distributions.add_hyperparameter(CSH.UniformIntegerHyperparameter("max_depth", 2, 4))

search = HpBandSterSearchCV(clf, param_distributions,random_state=0, n_jobs=1, n_iter=10, verbose=1).fit(X, y)
search.best_params_
```

Please refer to the [documentation of this library](https://hpbandster-sklearn.readthedocs.io/en/latest/), as well as to the documentation of [`HpBandSter`](https://automl.github.io/HpBandSter/build/html/index.html) and [`ConfigSpace`](https://automl.github.io/ConfigSpace/master/index.html) for more information.

Pipelines and `TransformedTargetRegressor` are also supported. Make sure to prefix the hyper parameter and resource names accordingly should you use either (or both) - for example, `final_estimator__n_estimators`. `n_samples` is not to be prefixed.

## Early stopping

As almost every search algorithm in `HpBandSter` leverages early stopping (mostly through Successive Halving), the user can configure the resource and budget to be used through the arguments of `HpBandSterSearchCV` object.

```python
search = HpBandSterSearchCV(
    clf,
    param_distributions,
    resource_name='n_samples', # can be either 'n_samples' or a string corresponding to an estimator attribute, eg. 'n_estimators' for an ensemble
    resource_type=float, # if specified, the resource value will be cast to that type before being passed to the estimator, otherwise it will be derived automatically
    min_budget=0.2,
    max_budget=1,
)

search = HpBandSterSearchCV(
    clf,
    param_distributions,
    resource_name='n_estimators', # can be either 'n_samples' or a string corresponding to an estimator attribute, eg. 'n_estimators' for an ensemble
    resource_type=int, # if specified, the resource value will be cast to that type before being passed to the estimator, otherwise it will be derived automatically
    min_budget=20,
    max_budget=200,
)
```

By default, the object will try to automatically determine the best resource, by checking the following in order:

- `'n_estimators'`, if the model has that attribute and the `warm_start` attribute
- `'max_iter'`, if the model has that attribute and the `warm_start` attribute
- `'n_samples'` - if the model doesn't support `warm_start`, the dataset samples will be used as the resource instead, meaing the model will be iteratively fitted on a bigger and bigger portion of the dataset.

Furthermore, special support has been added for `LightGBM`, `XGBoost` and `CatBoost` `scikit-learn` estimators.

## Documentation

https://hpbandster-sklearn.readthedocs.io/en/latest/

## References

- `HpBandSter` - https://github.com/automl/HpBandSter
- `ConfigSpace` - https://github.com/automl/ConfigSpace
- `scikit-learn` - http://scikit-learn.org/

## Author

Antoni Baum (Yard1)

## License

[MIT](https://github.com/Yard1/hpbandster-sklearn/blob/master/LICENSE)