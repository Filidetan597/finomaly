
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np

class DataHandler:
    def __init__(self, required_columns=None, categorical_columns=None, scaler=None):
        self.required_columns = required_columns or []
        self.categorical_columns = categorical_columns or []
        self.scaler = scaler or StandardScaler()
        self.encoder = None

    def load_excel(self, path):
        df = pd.read_excel(path)
        for col in self.required_columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in Excel.")
        return df

    def preprocess(self, df, fit_scaler=True):
        for col in self.required_columns:
            if df[col].isnull().any():
                if df[col].dtype in [np.float64, np.int64]:
                    df[col] = df[col].fillna(df[col].mean())
                else:
                    df[col] = df[col].fillna('Unknown')
        if self.categorical_columns:
            if fit_scaler or self.encoder is None:
                self.encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                cat_data = self.encoder.fit_transform(df[self.categorical_columns])
            else:
                cat_data = self.encoder.transform(df[self.categorical_columns])
            cat_cols = self.encoder.get_feature_names_out(self.categorical_columns)
            cat_df = pd.DataFrame(cat_data, columns=cat_cols, index=df.index)
            df = pd.concat([df.drop(columns=self.categorical_columns), cat_df], axis=1)
        return df

    def scale_features(self, X, fit_scaler=True):
        if fit_scaler:
            return self.scaler.fit_transform(X)
        else:
            return self.scaler.transform(X)
