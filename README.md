# jlmlcasestudies

A collection of machine learning case studies with interactive challenges.

## Project Structure

- `jlmlcasestudies/` - Python package
  - `cases/` - Case study Python implementations
    - `case1/` - Case Study 1: Point Prediction
      - `train_mlp.py` - MLP baseline training script
      - `train_hgb.py` - Histogram Gradient Boosting baseline script
    - `case2/` - Case Study 2: Distribution Sampling
      - `generate_reference.py` - Reference solution script
      - `generate_groundtruth.py` - Ground truth oracle script
  - `dataset1/` - Shared dataset module
    - `generate_data.py` - Dataset generation script
- `cases/` - Case study data outputs
  - `case1/data/` - Generated predictions and baseline results
  - `case2/data/` - Generated samples and training history
- `dataset1/data/` - Shared dataset files (train.npy, test_x.npy, test_y.npy)
- `frontend/` - React + Vite + Tailwind frontend application
  - `src/pages/case1/` - Case 1 frontend pages
  - `src/pages/case2/` - Case 2 frontend pages
  - `public/` - Static assets (data files are copied here during build)
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
python -m jlmlcasestudies.dataset1.generate_data
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

The data files will be automatically copied from `dataset1/data/` and `cases/case2/data/` to `frontend/public/case1/data/` and `frontend/public/case2/data/` during the dev/build process.

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
python -m jlmlcasestudies.cases.case1.train_mlp
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

**Reference Solution - Rectified Flow Matching:**
The reference solution uses rectified flow matching to learn the conditional distribution:
```bash
python -m jlmlcasestudies.cases.case2.generate_reference
```

This approach (Energy Score: ~2.0):
1. Standardizes inputs for stable training
2. Generates 10 t values per training datapoint with endpoint bias
3. For each t, generates random standard normal eps
4. Trains MLP to predict y-eps from raw features + Fourier embeddings of (x, t, y*t+(1-t)*eps)
5. Uses scipy solve_ivp (RK45) with tight tolerances and N(0,1) initial conditions to generate samples
6. Computes metrics before float16 quantization

**Ground Truth Oracle (for comparison):**
For comparison, ground truth sampling from the true mixture distribution:
```bash
python -m jlmlcasestudies.cases.case2.generate_groundtruth
```
This achieves Energy Score: ~0.5 (best possible with oracle access to true distribution).

## Deployment

This project is configured for Netlify deployment. Simply connect your repository to Netlify and it will automatically build and deploy the frontend.

