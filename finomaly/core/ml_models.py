import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator

class MLAnomalyModels:
    def __init__(self, method='isolation_forest', random_state=42):
        self.method = method
        self.model = None
        self.random_state = random_state
        self._build_model()

    def _build_model(self):
        if self.method == 'isolation_forest':
            self.model = IsolationForest(n_estimators=100, contamination=0.01, random_state=self.random_state)
        elif self.method == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        elif self.method == 'xgboost':
            self.model = XGBClassifier(
                n_estimators=100,
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='logloss',
                objective='binary:logistic',
                base_score=0.5
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def fit(self, X, y=None):
        if self.method == 'isolation_forest':
            self.model.fit(X)
        elif self.method in ['random_forest', 'xgboost']:
            if y is None:
                raise ValueError(
                    "Denetimli modeller (RandomForest, XGBoost) için etiketli (y) veri gereklidir. Lütfen y_train parametresini sağlayın. Etiketsiz veriyle sadece denetimsiz modeller (IsolationForest) kullanılabilir."
                )
            # Ensure y's type and values are correct
            if hasattr(y, 'values'):
                y_fit = y.values.astype(int)
            else:
                y_fit = np.array(y).astype(int)
            # For XGBoost, y must contain only 0 and 1
            if self.method == 'xgboost':
                if y_fit.min() < 0 or y_fit.max() > 1:
                    raise ValueError("XGBoost ile binary classification için y_train sadece 0 ve 1 değerlerinden oluşmalıdır. Şu an min: {} max: {}".format(y_fit.min(), y_fit.max()))
                if X.shape[0] != y_fit.shape[0]:
                    raise ValueError(f"X ve y satır sayısı eşit olmalı! X: {X.shape[0]}, y: {y_fit.shape[0]}")
            self.model.fit(X, y_fit)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def predict(self, X):
        if self.method == 'isolation_forest':
            return self.model.predict(X)
        elif self.method in ['random_forest', 'xgboost']:
            return self.model.predict(X)
        else:
            raise ValueError(f"Unknown method: {self.method}")
