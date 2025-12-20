# jlmlcasestudies

A collection of machine learning case studies with interactive challenges.

## Project Structure

- `frontend/` - React + Vite + Tailwind frontend application
- `dataset1/` - Shared dataset for case studies
  - `data/` - Generated data files (train.npy, test_x.npy, test_y.npy)
  - `scripts/` - Data generation and baseline training scripts
- `pyproject.toml` - Python dependencies and project configuration
- `netlify.toml` - Netlify deployment configuration

## Getting Started

### Python Setup

1. Install Python dependencies:
```bash
pip install -e .
```

2. Generate data for case studies:
```bash
python dataset1/scripts/generate_data.py
```

### Frontend Development

1. Install dependencies:
```bash
cd frontend
npm install
```

2. Run development server:
```bash
npm run dev
```

The data files will be automatically copied from `dataset1/data/` to `frontend/public/case1/data/` and `frontend/public/case2/data/` during the dev/build process.

3. Build for production:
```bash
npm run build
```

## Case Study 1: Point Prediction

In this challenge, you are given training data where you can observe both x and y. Your goal is to predict a single y value for each test x.

**Evaluation:** Root Mean Squared Error (RMSE)

**Files:**
- `train.npy` - 900×2 matrix with [x, y] pairs (float16)
- `test_x.npy` - 100 vector of x values (float16)
- `test_y.npy` - 100 vector of y values (float16) - used for scoring
- `mlp_test_yhat.npy` - 100 vector of baseline predictions using a tiny MLP

**Baseline:**
To generate a baseline using a tiny Multi-Layer Perceptron:
```bash
python dataset1/scripts/train_mlp.py
```

This script trains a model and reports both a baseline RMSE and the theoretically best-possible RMSE.

## Case Study 2: Distribution Sampling

Uses the same dataset as Case Study 1, but now you must produce TWO samples from the conditional distribution for each test x.

**Evaluation:** 2-sample Energy Score (CRPS - Continuous Ranked Probability Score)

**Files:**
- Same training and test data as Case Study 1
- Predictions should be a 100×2 matrix (two samples per test point)

**Energy Score:**
The energy score is calculated as:
```
ES = E[||Y - X1||] + E[||Y - X2||] - 0.5 * E[||X1 - X2||]
```
where Y is the true value, X1 and X2 are the two predicted samples.

**Baselines:**
Generate a naive baseline (adds noise to mean prediction):
```bash
python case2/scripts/generate_baseline.py
```

Generate an optimal baseline (samples from true mixture):
```bash
python case2/scripts/generate_optimal.py
```

## Deployment

This project is configured for Netlify deployment. Simply connect your repository to Netlify and it will automatically build and deploy the frontend.

