from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class MultiColumnLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.label_encoders = {}

    def fit(self, X, y=None):
        self.feature_names_in_ = list(X.columns)
        self.n_features_in_ = len(self.feature_names_in_)
        for c in X.columns:
            le = LabelEncoder()
            self.label_encoders[c] = le.fit(X[c])
        return self

    def transform(self, X):
        check_is_fitted(self)
        assert list(X.columns) == self.feature_names_in_
        output = X.copy()
        for c in output.columns:
            le = self.label_encoders[c]
            output[c] = le.transform(output[c])
        return output

    def inverse_transform(self, X):
        check_is_fitted(self)
        assert list(X.columns) == self.feature_names_in_
        output = X.copy()
        for c in output.columns:
            le = self.label_encoders[c]
            output[c] = le.inverse_transform(output[c])
        return output

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, names=None):
        return self.feature_names_in_
