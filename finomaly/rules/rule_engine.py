import json


import pandas as pd

class RuleEngine:
    def __init__(self, rules_path=None):
        self.rules = []
        if rules_path:
            if rules_path.endswith('.json'):
                self.load_rules_json(rules_path)
            elif rules_path.endswith('.xlsx') or rules_path.endswith('.xls'):
                self.load_rules_excel(rules_path)

    def load_rules_json(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            self.rules = json.load(f)

    def load_rules_excel(self, path):
        df = pd.read_excel(path)
        self.rules = df.to_dict(orient='records')

    def add_rule(self, rule):
        self.rules.append(rule)

    def apply(self, df):
        results = []
        for _, row in df.iterrows():
            row_result = []
            for rule in self.rules:
                col, op, val = rule['column'], rule['op'], rule['value']
                label = rule.get('label', 'RuleAnomaly')
                try:
                    if op == '>':
                        if row[col] > val:
                            row_result.append(label)
                    elif op == '<':
                        if row[col] < val:
                            row_result.append(label)
                    elif op == '==':
                        if row[col] == val:
                            row_result.append(label)
                    elif op == '!=':
                        if row[col] != val:
                            row_result.append(label)
                    elif op == 'in':
                        if row[col] in val:
                            row_result.append(label)
                    elif op == 'not in':
                        if row[col] not in val:
                            row_result.append(label)
                    elif op == 'contains':
                        if isinstance(row[col], str) and str(val) in row[col]:
                            row_result.append(label)
                    elif op == 'startswith':
                        if isinstance(row[col], str) and row[col].startswith(str(val)):
                            row_result.append(label)
                    elif op == 'endswith':
                        if isinstance(row[col], str) and row[col].endswith(str(val)):
                            row_result.append(label)
                except Exception:
                    continue
            results.append(row_result)
        return results
