import pandas as pd
import json
import os

class ProfileEngine:
    def __init__(self, lang='en', messages_path=None):
        self.lang = lang
        if messages_path is None:
            base = os.path.dirname(os.path.dirname(__file__))
            messages_path = os.path.join(base, 'core', 'messages_config.json')
        with open(messages_path, 'r', encoding='utf-8') as f:
            self.messages = json.load(f)

    def get_message(self, key):
        return self.messages.get(key, {}).get(self.lang, key)

    def time_series_anomaly(self, df, customer_col='MusteriID', amount_col='Tutar', window=10, threshold=3):
        # Z-score based time series anomaly detection per customer (rolling window)
        df = df.sort_values([customer_col, 'Saat'])
        anomalies = []
        for cust, group in df.groupby(customer_col):
            amounts = group[amount_col].rolling(window, min_periods=1).mean()
            stds = group[amount_col].rolling(window, min_periods=1).std().fillna(0)
            zscores = (group[amount_col] - amounts) / (stds + 1e-6)
            for z in zscores:
                if abs(z) > threshold:
                    anomalies.append('TS_Anomaly')
                else:
                    anomalies.append('')
        return anomalies

    def behavior_pattern_deviation(self, df, profiles, customer_col='MusteriID', amount_col='Tutar', freq_col='Saat', threshold=3):
        # Customer behavior pattern: deviation from mean amount, transaction frequency, hour range, etc.
        results = []
        for _, row in df.iterrows():
            if row[customer_col] in profiles.index:
                mean = profiles.loc[row[customer_col], 'mean']
                std = profiles.loc[row[customer_col], 'std']
                max_ = profiles.loc[row[customer_col], 'max']
                min_ = profiles.loc[row[customer_col], 'min']
                # Example: time deviation, e.g. night transaction when customer usually transacts during the day
                if std > 0 and abs(row[amount_col] - mean) > threshold * std:
                    results.append(self.get_message('behavior_deviation'))
                elif row[freq_col] < 6 and mean > 1000:  # example rule
                    results.append(self.get_message('unusual_time'))
                else:
                    results.append('')
            else:
                results.append('')
        return results


    def build_profile(self, df, customer_col='MusteriID', amount_col='Tutar'):
        # Extract mean, std, max, min statistics for each customer
        return df.groupby(customer_col)[amount_col].agg(['mean', 'std', 'max', 'min'])

    def detect_deviation(self, df, profiles, customer_col='MusteriID', amount_col='Tutar', threshold=3):
        # Example: deviation detection using Z-score
        results = []
        for _, row in df.iterrows():
            if row[customer_col] in profiles.index:
                mean = profiles.loc[row[customer_col], 'mean']
                std = profiles.loc[row[customer_col], 'std']
                if std > 0 and abs(row[amount_col] - mean) > threshold * std:
                    results.append('ProfileAnomaly')
                else:
                    results.append('')
            else:
                results.append('')
        return results
