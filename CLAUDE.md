# CLAUDE.md - Guidelines for AI Assistants

## Important Rules

### Never Show Fake Results
**CRITICAL**: Never generate synthetic or placeholder data to demonstrate expected patterns. All training results, plots, and data shown in this repository must come from actual training runs. If training data is not available, clearly indicate that and run the actual training.

## Running Training on Modal

This project uses [Modal](https://modal.com) for GPU-accelerated training.

### Authentication

Modal credentials are provided via environment variable:
- `MODAL_TOKEN_COMMAND` - Contains the command to set up Modal authentication

To authenticate, run the command stored in this environment variable:
```bash
$MODAL_TOKEN_COMMAND
```

### Case Study 3: Modular Arithmetic Training

To train the neural network for Case Study 3:

```bash
cd /path/to/jlmlcasestudies
modal run case3/scripts/modal_train_reference_jax.py --weight-decay 1.0 --output-suffix "_wd"
```

Available arguments:
- `--hidden-size` (default: 128) - Size of hidden layers
- `--n-epochs` (default: 50000) - Number of training epochs
- `--learning-rate` (default: 0.001) - Learning rate for Adam
- `--batch-size` (default: 512) - Minibatch size
- `--log-interval-epochs` (default: 10) - Log every N epochs for CSV storage
- `--weight-decay` (default: 0.0) - Weight decay for AdamW (try 1.0 for grokking)
- `--output-suffix` - Suffix for output filename (e.g., "_wd" for weight decay run)

### Case Study 2: Distribution Sampling Training

Similar pattern - see `case2/scripts/modal_train_reference.py` for details.
