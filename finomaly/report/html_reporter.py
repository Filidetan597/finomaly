import pandas as pd

class HTMLReporter:
    def __init__(self):
        pass

    def generate_html_report(self, df, output_path):
        html = df.to_html(index=False)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        return output_path
