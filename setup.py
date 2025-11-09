from setuptools import setup, find_packages

setup(
    name="finomaly",
    version="0.1.0",
    author="Barisaksel",
    description="Finansal işlemler için kural ve makine öğrenmesi tabanlı anomali tespit kütüphanesi.",
    packages=find_packages(),
    install_requires=[
        "scikit-learn",
        "numpy",
        "fpdf",
        "matplotlib",
        "seaborn",
        "pandas",
        "xgboost"
    ],
    python_requires=">=3.8",
)
