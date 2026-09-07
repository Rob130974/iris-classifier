# Iris Classifier (Decision Tree)

## Overview

This project is an end-to-end machine learning example for the AI Fundamentals course assessment. It builds a Decision Tree classifier using the classic Iris dataset from scikit-learn.

The model is trained to predict the species of an Iris flower from its measurements.

## Quick start

Create and activate a Python virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the model:

```bash
python src/train.py
```

Run the tests:

```bash
pytest
```

## Project structure

- `data/` - data folder
- `notebooks/iris_model.ipynb` - walkthrough Jupyter notebook
- `src/train.py` - Decision Tree training script
- `tests/test_train.py` - automated test
- `outputs/` - generated model outputs and figures
- `requirements.txt` - Python dependencies

## Model

The project uses a Decision Tree classifier trained on the Iris dataset. The dataset contains 150 samples representing three Iris species: Setosa, Versicolor and Virginica.

Using an 80/20 train/test split with `random_state=42`, the model achieved an accuracy of 100% on the test set.
## Results

The Decision Tree classifier achieved 100% accuracy on the test set. The model successfully classified all 30 test samples.
## Testing

Automated tests are included in `tests/test_train.py` and can be run using `pytest`.

## License

This project includes a LICENSE file.
