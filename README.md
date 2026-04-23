# Financial Sentiment Classification Pipeline

**Framework:** Kedro (QuantumBlack) | **Domain:** Natural Language Processing (Finance) | **Status:** Production-Baseline

## 1. Executive Summary
This repository contains a production-ready machine learning pipeline designed to classify the semantic orientation of financial texts (Positive, Negative, Neutral). Financial language presents unique challenges for standard sentiment classifiers due to domain-specific vernacular (e.g., "cost-cutting" typically denotes a positive trajectory for shareholders). 

To address this, the pipeline is trained on the curated Financial PhraseBank dataset. Built upon the **Kedro** framework, the architecture strictly adheres to enterprise software engineering principles, prioritizing reproducibility, modularity, and configuration-driven execution.

![Pipeline Directed Acyclic Graph (DAG)](pipeline_dag.png)
*Figure 1: Kedro-Viz representation of the pipeline's Directed Acyclic Graph (DAG).*

## 2. Pipeline Architecture & Methodology
The project operates as a Directed Acyclic Graph (DAG), segmented into two primary pipelines to enforce the separation of data engineering and data science concerns.

### 2.1 Data Processing Pipeline
* **Ingestion:** Automated retrieval of the Parquet-formatted Financial PhraseBank dataset.
* **Standardization:** Text normalization (lowercasing, specialized character removal) and strict null-handling.
* **Serialization:** Data is transcoded from raw CSV format into highly compressed `.parquet` formats to optimize downstream memory utilization and I/O operations.

### 2.2 Data Science Pipeline
* **Feature Engineering:** Text vectorization via Term Frequency-Inverse Document Frequency (TF-IDF).
* **Modeling:** Implementation of a Random Forest Classifier to establish a robust baseline.
* **Evaluation:** Automated generation of classification reports (Precision, Recall, F1-Score) and serialization of metrics to `metrics.json`.

## 3. Engineering Best Practices
This project is engineered for cross-functional team collaboration and seamless deployment:

* **Data Cataloging:** I/O operations are fully abstracted using the Kedro Data Catalog. This enables the codebase to transition seamlessly from local file execution to cloud-based storage (e.g., AWS S3, Azure S3) without modifying underlying Python logic.
* **Configuration Management:** Hyperparameters (e.g., `n_estimators`, `max_depth`, `test_size`) are externalized in `parameters.yml`. This allows data scientists to conduct hyperparameter tuning and experiment tracking without altering the core pipeline code.
* **Containerization:** The repository includes a `Dockerfile` to ensure environment consistency and scalable deployment across diverse computing environments.

## 4. Baseline Performance
The current baseline model achieves a classification accuracy of **~72%** on the three-class testing subset. This establishes a highly functional, fully orchestrated baseline designed to be easily swapped with advanced transformer-based architectures (e.g., FinBERT) in future iterations.

## 5. Execution Instructions

### 5.1 Local Environment Setup
```bash
# Initialize and activate virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows: .\venv\Scripts\Activate

# Install pipeline dependencies
pip install -r src/requirements.txt
pip install kedro-datasets pyarrow datasets