from setuptools import setup, find_packages

setup(
    name="finomaly",
    version="0.1.4",
    author="Barisaksel",
    description="A rule-based and machine learning-based anomaly detection library for financial transactions.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
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
    keywords="anomaly detection, financial, machine learning, rule-based, fraud, outlier, python, data science, fintech, unsupervised learning",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    project_urls={
        "Source": "https://github.com/Barisaksel/finomaly",
        "PyPI": "https://pypi.org/project/finomaly/",
        # "Documentation": "https://github.com/Barisaksel/finomaly#readme"  # Uncomment if you have a docs site
    },
)
