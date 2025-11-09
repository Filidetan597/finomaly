import pandas as pd

class Reporter:
    def __init__(self, lang='en'):
        self.lang = lang

    def generate_report(self, df, output_path):
        df.to_excel(output_path, index=False)
        return output_path
