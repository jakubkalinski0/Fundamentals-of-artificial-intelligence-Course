# Fundamentals of Artificial Intelligence Course

Course laboratory materials for **Basics of Artificial Intelligence** at AGH University of Science and Technology. The repository contains Jupyter notebooks and supporting scripts that walk through core machine learning workflows: data preparation, model training, evaluation, and tuning.

## Repository structure

| Lab | Topic | Main files |
|-----|-------|------------|
| Lab 1 | Linear and logistic regression, preprocessing, regularization | `Lab1/lab_1.ipynb`, `Lab1/lab_1.py` |
| Lab 2 | Imbalanced classification, ensemble methods, feature importance | `Lab2/lab_2.ipynb`, `Lab2/lab_2.py` |
| Lab 3 | Neural networks with PyTorch (MLP, regularization, early stopping) | `Lab3/lab_3.ipynb` |
| Lab 5 | Natural language processing with Hugging Face Transformers | `Lab5/lab_5.ipynb` |
| Lab 6 | Game-playing agents (minimax, alpha-beta, MCTS) | `Lab6/lab_6.ipynb` |
| Lab 7 | Recommender systems (collaborative filtering, matrix factorization) | `Lab7/lab_7.ipynb` |

Lab 4 is not included in this repository.

Some labs also ship paired `.py` files generated with [Jupytext](https://jupytext.readthedocs.io/) for editing outside the notebook interface.

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

Core dependencies are listed in `pyproject.toml` and `requirements.txt`. They include NumPy, Pandas, scikit-learn, PyTorch, LightGBM, imbalanced-learn, and Jupyter.

Additional packages used in optional or advanced exercises:

- `statsmodels` (Lab 1, regression diagnostics)
- `optuna` (Lab 3, hyperparameter search)
- `transformers`, `datasets`, `peft`, `evaluate` (Lab 5, NLP)
- `librecommender` (Lab 7, LightGCN bonus task)

Lab 5 benefits from a CUDA-capable GPU. CPU-only execution is possible but training will be slower.

## Setup

### Using uv (recommended)

```bash
uv sync
uv run jupyter lab
```

### Using pip

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows

pip install -r requirements.txt
jupyter lab
```

Install extra packages when needed:

```bash
pip install statsmodels optuna transformers datasets peft evaluate librecommender
```

## Running the labs

1. Open the notebook for the lab you want to work on (for example `Lab1/lab_1.ipynb`).
2. Select the project virtual environment as the Jupyter kernel.
3. Run cells from top to bottom. Exercise cells are marked in the notebook.
4. Datasets are either bundled in the lab folder or downloaded inside the notebook.

Lab 5 may write trained model checkpoints under `Lab5/output/`. Lab 7 downloads MovieLens 100k on first run if the dataset is not present locally.

## Lab summaries

**Lab 1** covers the end-to-end regression pipeline on the Ames Housing dataset: exploratory analysis, missing values, categorical encoding, linear regression, Ridge and LASSO regularization, cross-validation, and logistic regression for classification.

**Lab 2** focuses on imbalanced binary classification (Polish companies bankruptcy data): stratified splits, SMOTE, decision trees, random forests, LightGBM, hyperparameter tuning, and feature importance. The bonus task compares feature selection methods (filter, embedded, wrapper).

**Lab 3** introduces feedforward neural networks in PyTorch: custom datasets, training loops, dropout, batch normalization, class weights, early stopping, and optional GPU acceleration and Optuna-based tuning.

**Lab 5** explores transformer models for Polish NLP tasks using the Hugging Face ecosystem, including fine-tuning and evaluation on question answering and related datasets.

**Lab 6** implements tic-tac-toe players with increasing sophistication: random and rule-based bots, minimax, alpha-beta pruning, Monte Carlo Tree Search, and full MCTS with UCT selection.

**Lab 7** builds recommender systems on MovieLens-style rating data: baseline predictors, k-nearest neighbors, FunkSVD, ranking metrics, and an optional LightGCN graph-based model.

## License

See [LICENSE](LICENSE) for license terms.
