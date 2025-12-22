# jlmlcasestudies

A collection of machine learning case studies with interactive challenges.

## Project Structure

- Root contains React + Vite + Tailwind frontend application (package.json, vite.config.js, src/, etc.)
- `case1/` - Case Study 1: Point Prediction
  - `frontend/` - Case 1 specific frontend pages (Case1.jsx, Case1Solutions.jsx)
  - `scripts/` - Case 1 specific Python scripts (train_mlp.py, train_hgb.py)
- `case2/` - Case Study 2: Distribution Sampling
  - `frontend/` - Case 2 specific frontend pages (Case2.jsx, Case2Solutions.jsx)
  - `scripts/` - Case 2 specific Python scripts (generate_reference.py, generate_groundtruth.py)
- `dataset1/` - Shared dataset for case studies
  - `data/` - Generated data files (train.npy, test_x.npy, test_y.npy)
  - `scripts/` - Data generation script (generate_data.py)
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
npm install
```

2. Run development server:
```bash
npm run dev
```

The data files will be automatically copied from `dataset1/data/` to `public/case1/data/` and `public/case2/data/` during the dev/build process.

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
python case1/scripts/train_mlp.py
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
python case2/scripts/generate_reference.py
```

This approach (Energy Score: ~1.8):
1. Standardizes inputs for stable training
2. Generates 3 t values per training datapoint: t=0, t=1, and random t
3. For each t, generates random standard normal eps
4. Trains MLP to predict y-eps from raw features of (x, t, y*t+(1-t)*eps)
5. Uses scipy solve_ivp (RK45) with tight tolerances and N(0,1) initial conditions to generate samples
6. Computes metrics before float16 quantization
7. Architecture: (256, 128, 128, 64) hidden layers
8. Training data: 900 samples from dataset1

**Infinite Data Solution - Rectified Flow with Infinite Training Data:**
An alternative solution that trains on infinite data by generating fresh samples from the true generative model:
```bash
python case2/scripts/generate_infinitedata.py
```

This approach (Energy Score: ~1.8):
1. Generates fresh training data each epoch from true distribution: x ~ N(4, 1), y|x ~ mixture of N(10*cos(x), 1) and N(0, 1)
2. Uses raw features (same as reference)
3. Same architecture: (256, 128, 128, 64) hidden layers
4. Trains with partial_fit on fresh samples each epoch
5. Uses scipy solve_ivp (RK45) with tight tolerances and N(0,1) initial conditions to generate samples

**JAX + Modal.com GPU Training:**
For faster training on GPU infrastructure using JAX:
```bash
# Run locally (requires Modal token)
modal run case2/scripts/modal_train_infinitedata_jax.py --duration-minutes 10
```

This approach uses JAX with T4 GPU on Modal.com infrastructure:
1. Same infinite data generation as above
2. JAX for GPU-accelerated training
3. Same architecture: (256, 128, 128, 64) hidden layers
4. Trains for fixed duration (10 minutes default) without early stopping
5. Outputs training loss, energy score CSV files and plots
6. Generates 1000 sample scatter plot with CSV

**GitHub Actions Workflows:**

*Manual Workflow:*
1. Go to Actions → "Train Case2 Infinite Data with JAX on Modal"
2. Click "Run workflow" and optionally specify training duration
3. Artifacts include: training_loss.csv, energy_score.csv, scatter_samples.csv, and corresponding plots
4. Artifacts are retained for 3 days

*Comment-Triggered Workflow:*
1. In any PR, repo owner can comment `/runmodal`
2. Automatically runs all `modal_*.py` scripts that differ from main
3. Posts results as a reply comment with artifact links
4. Useful for testing Modal script changes in PRs

**Comparison:**
All solutions use the same architecture and features but differ in training approach:
- Reference: Fixed dataset (900 samples from dataset1), CPU-based
- Infinite data: Fresh samples each epoch (unlimited data from true distribution), CPU-based
- JAX + Modal: Fresh samples each epoch with GPU acceleration

**Ground Truth Oracle (for comparison):**
For comparison, ground truth sampling from the true mixture distribution:
```bash
python case2/scripts/generate_groundtruth.py
```
This achieves Energy Score: ~0.5 (best possible with oracle access to true distribution).

## Deployment

This project is configured for Netlify deployment. Simply connect your repository to Netlify and it will automatically build and deploy the frontend.

