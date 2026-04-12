import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))
try:
    from preprocess import get_clean_data
    from evaluate import run_benchmarking
    from ensemble import run_ensemble_benchmark
    import mne
except ImportError:
    print("Error: Required scripts (preprocess.py, evaluate.py, ensemble.py) not found in src/.")
    sys.exit(1)
def run_colab_master():
    print("====================================================")
    print("   EEG P300 BCI - FINAL SCIENTIFIC BENCHMARK        ")
    print("====================================================")
    print("\n[Step 1] Running Multi-Model Benchmark (5-Fold Grouped CV)...")
    run_benchmarking()
    print("\n[Step 2] Running Character-Level Ensemble Verification...")
    run_ensemble_benchmark()
    print("\n====================================================")
    print("   PROCESS COMPLETE. CHECK 'results/' FOR OUTPUTS.  ")
    print("====================================================")
if __name__ == "__main__":
    run_colab_master()
