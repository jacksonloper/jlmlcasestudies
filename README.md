# jlmlcasestudies

A collection of machine learning case studies with interactive challenges.

## Project Structure

- `frontend/` - React + Vite + Tailwind frontend application
- `case1/` - First case study: Conditional Distribution Prediction
  - `data/` - Generated data files (train.npy, test_x.npy, test_y.npy)
  - `scripts/` - Data generation scripts
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
python case1/scripts/generate_data.py
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

The data files will be automatically copied from `case1/data/` to `frontend/public/case1/data/` during the dev/build process.

3. Build for production:
```bash
npm run build
```

## Case Study 1: Conditional Distribution Prediction

In this challenge, you are given training data where you can observe both x and y. Your goal is to predict y for the test set where only x is given.

**Data Generation:**
- x ~ N(4, 1)
- y | x is an equal parts mixture of N(x², 1) and N(0, 1)

**Files:**
- `train.npy` - 900×2 matrix with [x, y] pairs (float16)
- `test_x.npy` - 100 vector of x values (float16)
- `test_y.npy` - 100 vector of y values (float16) - used for scoring

## Deployment

This project is configured for Netlify deployment. Simply connect your repository to Netlify and it will automatically build and deploy the frontend.

