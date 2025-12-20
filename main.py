import os
import random
import warnings

import numpy as np

# ---------------------------------------------------------------------
# Silence known non-actionable warnings from dependencies
# ---------------------------------------------------------------------
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"pandas_datareader(\.|$)")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"distutils(\.|$)")

# SHAP / NumPy RNG warning (comes from shap internals)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*NumPy global RNG was seeded by calling `np\.random\.seed`.*",
)

# ---------------------------------------------------------------------
# Global seed for full reproducibility
# ---------------------------------------------------------------------
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)


def main() -> None:
    print("\n=== Financial Crisis Early-Warning System ===\n", flush=True)

    # Import here to avoid any import-time weirdness (and keep main clean)
    from src.data.download_data import download_all, RAW_DATA_DIR
    from src.data.preprocess import build_and_save_ews_dataset
    from src.modeling.train_models import main as train_models_main, DATA_PATH
    from src.modeling.evaluate_models import main as evaluate_models_main
    from src.modeling.explain_models import main as explain_models_main

    try:
        from src.viz.plots import main as plots_main
    except ImportError:
        plots_main = None

    # -----------------------------------------------------------------
    # Step 1 — Download raw data
    # -----------------------------------------------------------------
    if not os.path.exists(RAW_DATA_DIR):
        print("[Step 1] Raw data not found. Downloading raw financial series...", flush=True)
        download_all()
    else:
        print("[Step 1] Raw data already available. Skipping download.", flush=True)

    # -----------------------------------------------------------------
    # Step 2 — Build the EWS dataset
    # -----------------------------------------------------------------
    if not os.path.exists(DATA_PATH):
        print("\n[Step 2] Building the EWS dataset...", flush=True)
        build_and_save_ews_dataset()
    else:
        print(f"\n[Step 2] Dataset already exists at {DATA_PATH}. Skipping.", flush=True)

    # -----------------------------------------------------------------
    # Step 3 — Train ML models
    # -----------------------------------------------------------------
    print("\n[Step 3] Training models...", flush=True)
    train_models_main(seed=SEED)

    # -----------------------------------------------------------------
    # Step 4 — Evaluate models
    # -----------------------------------------------------------------
    print("\n[Step 4] Evaluating models...", flush=True)
    evaluate_models_main(seed=SEED)

    # -----------------------------------------------------------------
    # Step 5 — Model explanations (SHAP)
    # -----------------------------------------------------------------
    print("\n[Step 5] Generating explanations (SHAP)...", flush=True)
    explain_models_main(seed=SEED)

    # -----------------------------------------------------------------
    # Step 6 — Additional plots (optional)
    # -----------------------------------------------------------------
    if plots_main is not None:
        print("\n[Step 6] Generating additional visualizations...", flush=True)
        plots_main()
    else:
        print("\n[Step 6] No optional visualization module found. Skipping.", flush=True)

    print("\n=== Pipeline completed successfully. ===", flush=True)
    print("Outputs available in: models/ and results/\n", flush=True)


if __name__ == "__main__":
    main()