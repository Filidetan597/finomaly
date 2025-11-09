# Finomaly

Finomaly is a modular, open-source Python library for anomaly detection in financial transactions. It supports both rule-based and machine learning-based detection, multi-language reporting, and professional reporting formats.

## Features
- Rule-based anomaly detection (JSON-configurable, customer-specific rules)
- Machine learning models: IsolationForest, RandomForest, XGBoost
- Profile-based analysis (behavioral deviation, unusual time, etc.)
- Multi-language support (TR/EN) for all messages and reports
- Centralized message and rule management
- Professional reporting: Excel, HTML, PDF (with optional charts)
- Visual analytics: anomaly distribution, scatter plots
- Easy integration, clean API, and extensible modular structure

## Installation
```bash
pip install finomaly
```

## Quick Start
```python
import pandas as pd
from finomaly.core.anomaly_system import CorporateAnomalySystem

# Load your data
train_df = pd.read_excel('train.xlsx')
predict_df = pd.read_excel('predict.xlsx')

# Define features and rules
features = ['Tutar', 'Saat']
rules_path = 'rules.json'

# Initialize system
system = CorporateAnomalySystem(features, rules_path=rules_path, ml_method='isolation_forest', lang='en')

# Train model
system.fit('train.xlsx', customer_col='MusteriID', amount_col='Tutar')

# Predict anomalies
output_path = system.predict('predict.xlsx', customer_col='MusteriID', amount_col='Tutar')
result = pd.read_excel(output_path)
print(result.head())
```

## Reporting & Visualization
```python
from finomaly.report.visualizer import Visualizer
from finomaly.report.pdf_reporter import PDFReporter

visualizer = Visualizer()
visualizer.plot_anomaly_distribution(result, amount_col='Tutar', anomaly_col='ML_Anomaly')

pdf_reporter = PDFReporter()
pdf_reporter.generate_pdf_report(result, 'report.pdf')
```

## Project Structure
- `core/` : Rule engine, model management, utilities
- `ml/` : ML models (IsolationForest, RandomForest, XGBoost)
- `profile/` : Profile-based analysis (behavioral, time-based)
- `report/` : Reporting and visualization (Excel, HTML, PDF, charts)

## Contributing
Finomaly is open-source and welcomes contributions. Please open issues or pull requests for improvements, bug fixes, or new features.

## License
MIT License

## Author
Barış
