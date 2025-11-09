from sklearn.ensemble import IsolationForest, RandomForestClassifier
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

import joblib


class MLAnomalyModels:
    def __init__(self, method='isolation_forest', **kwargs):
        self.method = method
        self.model = None
        self.kwargs = kwargs
        self.is_fitted = False

    def fit(self, X, y=None):
        if self.method == 'isolation_forest':
            self.model = IsolationForest(**self.kwargs)
            self.model.fit(X)
            self.is_fitted = True
        elif self.method == 'random_forest':
            self.model = RandomForestClassifier(**self.kwargs)
            self.model.fit(X, y)
            self.is_fitted = True
        elif self.method == 'xgboost':
            if XGBClassifier is None:
                raise ImportError('xgboost is not installed')
            self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', **self.kwargs)
            self.model.fit(X, y)
            self.is_fitted = True
        else:
            raise ValueError(f"Unsupported method: {self.method}")
        return self

    def predict(self, X):
        if self.model is None or not self.is_fitted:
            raise RuntimeError('Model is not trained.')
        return self.model.predict(X)

    def save(self, path):
        if self.method == 'xgboost' and self.model is not None:
            self.model.save_model(path + '.json')
        else:
            joblib.dump(self.model, path)

    def load(self, path):
        if self.method == 'xgboost':
            if XGBClassifier is None:
                raise ImportError('xgboost is not installed')
            self.model = XGBClassifier()
            self.model.load_model(path + '.json')
            self.is_fitted = True
        else:
            self.model = joblib.load(path)
            self.is_fitted = True
