import pandas as pd
import joblib
import os
import json
from sklearn.ensemble import IsolationForest

class ExcelAmountAnomalyDetector:
    def __init__(self, model_path="amount_anomaly_model.pkl", lang="en", messages_path=None):
        self.model_path = model_path
        self.model = None
        self.is_trained = False
        self.lang = lang if lang in ("en", "tr") else "en"
        if messages_path is None:
            messages_path = os.path.join(os.path.dirname(__file__), "messages_config.json")
        with open(messages_path, "r", encoding="utf-8") as f:
            self.MESSAGES = json.load(f)
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            self.is_trained = True

    def set_language(self, lang):
        if lang in self.MESSAGES['anomaly']:
            self.lang = lang
        else:
            self.lang = "en"

    def fit(self, excel_path, amount_column="Tutar"):
        df = pd.read_excel(excel_path)
        if amount_column not in df.columns:
            msg = self.MESSAGES['column_missing'][self.lang].format(col=amount_column)
            raise ValueError(msg)
        X = df[[amount_column]].values
        self.model = IsolationForest(contamination=0.05, random_state=42)
        self.model.fit(X)
        joblib.dump(self.model, self.model_path)
        self.is_trained = True
        return self

    def update(self, excel_path, amount_column="Tutar"):
        return self.fit(excel_path, amount_column)

    def predict(self, excel_path, amount_column="Tutar", output_path=None, result_column=None):
        if not self.is_trained:
            msg = self.MESSAGES['not_trained'][self.lang]
            raise RuntimeError(msg)
        df = pd.read_excel(excel_path)
        if amount_column not in df.columns:
            msg = self.MESSAGES['column_missing'][self.lang].format(col=amount_column)
            raise ValueError(msg)
        X = df[[amount_column]].values
        preds = self.model.predict(X)
        if result_column is None:
            result_column = self.MESSAGES['result_column'][self.lang]
        anomaly_label = self.MESSAGES['anomaly'][self.lang]
        df[result_column] = [anomaly_label if p == -1 else "" for p in preds]
        if output_path is None:
            output_path = excel_path
        df.to_excel(output_path, index=False)
        return output_path
