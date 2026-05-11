# Data Directory

This directory is the default location for cached dataset downloads.

When running the pipeline scripts (e.g., `src/evaluate.py`), the **MOABB** library will automatically download the requested datasets and cache them here. 

**Expected Directory Structure after running:**
```
data/
└── MNE-BNCI-data/
    └── dataset_09_2014/
        └── subject_01.mat
        └── ...
```

**Note:** You do not need to manually download or unpack zip files. The pipeline handles this directly.
