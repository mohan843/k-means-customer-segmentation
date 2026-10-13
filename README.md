# K-Means Customer Segmentation

A small, dependency-free Python project that segments customers into groups using K-Means. It standardizes numeric features before clustering so columns with larger units do not dominate, reports an interpretable average profile for each segment, and supports reproducible runs with a random seed.

## Features

- Pure Python K-Means implementation with deterministic seeded initialization
- Feature standardization, prediction, cluster labels, and within-cluster sum of squares
- CSV input with feature selection and helpful validation errors
- Included sample customer dataset and automated unit tests

## Setup

Requires Python 3.10 or newer. No third-party packages are needed.

```bash
git clone https://github.com/mohan843/k-means-customer-segmentation.git
cd k-means-customer-segmentation
python -m venv .venv
# Activate the environment: source .venv/bin/activate (macOS/Linux)
# On Windows use: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Usage

Run three customer segments on the included data:

```bash
python segmentation.py --data data/customers.csv --k 3
```

Select particular numeric columns and set a reproducible seed:

```bash
python segmentation.py --data data/customers.csv --features annual_income_k spending_score --k 2 --seed 7
```

Use the implementation from Python:

```python
from segmentation import KMeans, load_csv

feature_names, customers = load_csv("data/customers.csv")
model = KMeans(n_clusters=3, random_state=42).fit(customers)
print(model.labels_)
print(model.inertia_)
```

## Sample data

`data/customers.csv` contains 18 illustrative customer records with annual income (in thousands), spending score, and age. Values are synthetic and intended for demonstration, not business decisions.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Project files

- `segmentation.py`: K-Means implementation, CSV loader, and command-line interface
- `data/customers.csv`: sample customer records
- `tests/test_segmentation.py`: unit and CLI smoke tests
- `requirements.txt`: dependency note; standard library only
