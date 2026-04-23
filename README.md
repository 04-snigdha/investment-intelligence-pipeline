# 📈 Investment Intelligence: Financial Sentiment Pipeline

A production-grade, modular Machine Learning pipeline built with **Kedro** to classify the sentiment of financial news and corporate disclosures. 

Unlike experimental Jupyter Notebooks, this project demonstrates enterprise software engineering practices for Data Science, including **configuration-driven execution**, **data cataloging**, and **DAG-based pipeline orchestration**.

## 🎯 Executive Summary
Financial sentiment analysis requires domain-specific understanding (e.g., "cost cutting" is often positive for shareholders but negative for employees). This pipeline uses the **Financial PhraseBank** dataset to train a predictive model.

* **Baseline Model:** TF-IDF Vectorizer + Random Forest Classifier
* **Baseline Accuracy:** ~72% (Three-class: Positive, Negative, Neutral)
* **Framework:** Kedro (QuantumBlack / McKinsey & Company)

## 🏗️ Pipeline Architecture

The project is structured as a Directed Acyclic Graph (DAG) consisting of two distinct pipelines:

### 1. Data Processing Pipeline (`data_processing`)
Responsible for data engineering and standardization.
* **Ingestion:** Automatically fetches the Parquet-formatted Financial PhraseBank dataset from the Hugging Face hub via a custom ingestion script.
* **Standardization:** Cleans text (lowercasing, punctuation removal) and handles missing values.
* **Optimization:** Converts the raw CSV into a highly compressed `.parquet` format for fast downstream read/write operations.

### 2. Data Science Pipeline (`data_science`)
Responsible for feature engineering, model training, and evaluation.
* **Dynamic Splitting:** Train/Test splits are controlled via `parameters.yml` (no hardcoded variables).
* **Training:** Vectorizes text and trains the Random Forest algorithm.
* **Evaluation:** Outputs a classification report and saves performance metrics to `data/08_reporting/metrics.json`.
* **Artifacts:** Serializes and exports the trained model pipeline as a `.pkl` file.

## 🛠️ Tech Stack
* **Orchestration:** Kedro
* **Data Processing:** Pandas, PyArrow
* **Machine Learning:** Scikit-Learn
* **Data Sourcing:** Hugging Face `datasets`

## 🚀 How to Run

**1. Clone the repository and install dependencies:**
```bash
git clone [https://github.com/04-snigdha/investment-intelligence-pipeline.git](https://github.com/04-snigdha/investment-intelligence-pipeline.git)
cd investment-intelligence-pipeline
python -m venv venv
source venv/Scripts/activate  # On Windows: .\venv\Scripts\Activate
pip install -r src/requirements.txt
pip install kedro-datasets pyarrow datasets