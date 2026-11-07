# Machine Learning Algorithms with Python

A hands-on collection of supervised machine learning examples implemented in Python and scikit-learn. The repository covers regression, binary and multiclass classification, nearest neighbors, decision trees, and support vector machines.

The scripts are organized as a learning path, from foundational regression models to classification examples using real-world datasets.

## Topics covered

| # | Script | Main topic | Example / dataset |
|---|---|---|---|
| 01 | [Simple Linear Regression](01_simple_linear_regression.py) | Predict a continuous value from one feature | Vehicle fuel consumption and CO₂ emissions |
| 02 | [Multiple Linear Regression](02_multiple_linear_regression.py) | Predict a continuous value from multiple features | Vehicle fuel consumption and CO₂ emissions |
| 03 | [Logistic Regression](03_logistic_regression.py) | Binary classification | Classification exercise using a labeled dataset |
| 04 | [K-Nearest Neighbors (KNN)](04_knn_classification.py) | Distance-based classification | Telecommunications customer groups |
| 05 | [Decision Trees](05_decision_trees.py) | Tree-based classification | Drug-response dataset |
| 06 | [Decision Tree Regression](06_regression_trees_taxi_tip.py) | Tree-based regression | Taxi tip prediction |
| 07 | [Multiclass Classification](07_multiclass_classification.py) | Classification with multiple classes | Multiclass classification exercise |
| 08 | [Decision Tree and SVM](08_decision_tree_svm_ccfraud.py) | Classification and model comparison | Credit-card fraud detection |

## Tech stack

- **Python** — programming language
- **Pandas** — data loading and manipulation
- **NumPy** — numerical operations
- **Matplotlib** and **Seaborn** — data visualization
- **Scikit-learn** — preprocessing, model training, prediction, and evaluation

## Repository structure

```text
machine-learning-algorithms/
├── 01_simple_linear_regression.py
├── 02_multiple_linear_regression.py
├── 03_logistic_regression.py
├── 04_knn_classification.py
├── 05_decision_trees.py
├── 06_regression_trees_taxi_tip.py
├── 07_multiclass_classification.py
├── 08_decision_tree_svm_ccfraud.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/ArturDeathCode0x/machine-learning-algorithms.git
cd machine-learning-algorithms
```

### 2. Create a virtual environment

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run an example

For example:

```bash
python 01_simple_linear_regression.py
```

Choose any script from the table above. Some examples download their datasets from public URLs, so an internet connection may be required. Matplotlib charts may open in separate windows.

## Learning objectives

By working through these examples, you can practice:

- Preparing datasets and selecting input features.
- Splitting data into training and test sets.
- Scaling and encoding features when needed.
- Training regression and classification models.
- Generating predictions on unseen data.
- Evaluating models with metrics such as MAE, MSE, RMSE, R², accuracy, and ROC-AUC.
- Visualizing data distributions, relationships, and decision trees.

## Notes

- This is an educational repository for study and experimentation, not a production-ready machine learning package.
- The examples use public datasets referenced in the scripts; datasets are not stored in this repository.
- Some scripts retain practice exercises or TODO sections from their original learning materials. Review each file and complete the exercises as part of the learning process.
- Results can vary depending on dependency versions, dataset availability, and execution environment.
- The fraud-detection example is for educational purposes only and must not be used as a real payment or fraud-prevention system.

## Author

**Pedro Artur Carvalho da Silva**  
GitHub: [@ArturDeathCode0x](https://github.com/ArturDeathCode0x)

---

*Built as part of my ongoing studies in Python, data analysis, and machine learning.*
