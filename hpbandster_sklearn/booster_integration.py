# Adapted from https://github.com/ray-project/tune-sklearn/blob/master/tune_sklearn/_detect_booster.py

def has_xgboost():
    try:
        import xgboost

        return True
    except ImportError:
        return False


def is_xgboost_model(clf):
    if not has_xgboost():
        return False
    from xgboost.sklearn import XGBModel

    return isinstance(clf, XGBModel)


def has_lightgbm():
    try:
        import lightgbm

        return True
    except ImportError:
        return False


def has_required_lightgbm_version():
    """
    lightgbm>=3.0.0 is required for early stopping
    """
    try:
        import lightgbm

        version = [int(x) for x in lightgbm.__version__.split(".")]
        return version[0] >= 3
    except ImportError:
        return False


def is_lightgbm_model(clf):
    if not has_lightgbm():
        return False
    from lightgbm.sklearn import LGBMModel

    return isinstance(clf, LGBMModel)


def is_lightgbm_model_of_required_version(clf):
    if not has_required_lightgbm_version():
        return False
    from lightgbm.sklearn import LGBMModel

    return isinstance(clf, LGBMModel)


def has_catboost():
    try:
        import catboost

        return True
    except ImportError:
        return False


def is_catboost_model(clf):
    if not has_catboost():
        return False
    from catboost import CatBoost

    return isinstance(clf, CatBoost)
