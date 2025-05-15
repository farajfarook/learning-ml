from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array


class StandardScalerClone(BaseEstimator, TransformerMixin):
    def __init__(self, with_mean=True):
        self.with_mean = with_mean

    def fit(self, X, y=None):
        # Check if X is a 2D array
        X = check_array(X, ensure_2d=True)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        # Check if the scaler has been fitted
        check_is_fitted(self)
        # Check if X is a 2D array
        X = check_array(X, ensure_2d=True)
        # Check if the number of features matches
        assert self.n_features_in_ == X.shape[1], (
            "X has different number of features than fitted data"
        )
        if self.with_mean:
            X -= self.mean_
        return X / self.scale_
