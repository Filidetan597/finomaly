from finomaly.core.data_handler import DataHandler
from finomaly.rules.rule_engine import RuleEngine
from finomaly.ml.ml_models import MLAnomalyModels
from finomaly.profile.profile_engine import ProfileEngine
from finomaly.report.reporter import Reporter
import joblib
import os

class CorporateAnomalySystem:
    def __init__(self, features, rules_path=None, ml_method='isolation_forest', lang='en', model_path=None):
        self.data_handler = DataHandler(required_columns=features)
        self.rule_engine = RuleEngine(rules_path)
        self.ml_model = MLAnomalyModels(method=ml_method)
        self.profile_engine = ProfileEngine()
        self.reporter = Reporter(lang=lang)
        self.features = features
        self.lang = lang
        self.profiles = None
        self.model_path = model_path
        self.ml_method = ml_method
        if self.model_path and os.path.exists(self.model_path):
            self.load_model(self.model_path)

    def fit(self, excel_path, y=None, customer_col=None, amount_col=None, save_model=True):
        df = self.data_handler.load_excel(excel_path)
        df_proc = self.data_handler.preprocess(df, fit_scaler=True)
        X = df_proc[self.features].values
        X_scaled = self.data_handler.scale_features(X, fit_scaler=True)
        self.ml_model.fit(X_scaled, y)
        if customer_col and amount_col:
            self.profiles = self.profile_engine.build_profile(df, customer_col, amount_col)
        if save_model and self.model_path:
            self.save_model(self.model_path)
        return self

    def predict(self, excel_path, output_path=None, customer_col=None, amount_col=None):
        df = self.data_handler.load_excel(excel_path)
        df_proc = self.data_handler.preprocess(df, fit_scaler=False)
        X = df_proc[self.features].values
        X_scaled = self.data_handler.scale_features(X, fit_scaler=False)
        ml_preds = self.ml_model.predict(X_scaled)
        rule_results = self.rule_engine.apply(df) if self.rule_engine.rules else [[] for _ in range(len(df))]
        # ML_Anomaly column may differ depending on the model
        if self.ml_method == 'isolation_forest':
            ml_anomaly = ['Anomaly' if p == -1 else '' for p in ml_preds]
        else:
            ml_anomaly = ['Anomaly' if p == 1 else '' for p in ml_preds]
        # Profile and behavioral analysis
        if self.profiles is not None and customer_col and amount_col:
            profile_results = self.profile_engine.detect_deviation(df, self.profiles, customer_col, amount_col)
            ts_anomaly = self.profile_engine.time_series_anomaly(df, customer_col, amount_col)
            behavior_dev = self.profile_engine.behavior_pattern_deviation(df, self.profiles, customer_col, amount_col, freq_col='Saat')
        else:
            profile_results = [''] * len(df)
            ts_anomaly = [''] * len(df)
            behavior_dev = [''] * len(df)
        df['ML_Anomaly'] = ml_anomaly
        df['Rule_Anomaly'] = [','.join(r) if r else '' for r in rule_results]
        df['Profile_Anomaly'] = profile_results
        df['TS_Anomaly'] = ts_anomaly
        df['Behavior_Deviation'] = behavior_dev
        if output_path is None:
            output_path = excel_path
        self.reporter.generate_report(df, output_path)
        return output_path

    def save_model(self, path):
        self.ml_model.save(path)
        self.model_path = path

    def load_model(self, path):
        self.ml_model.load(path)
        self.model_path = path
