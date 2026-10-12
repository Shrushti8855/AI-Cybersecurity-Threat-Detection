# Dataset Instructions

This project uses the UNSW-NB15 network intrusion detection dataset.

Official source: https://research.unsw.edu.au/projects/unsw-nb15-dataset

Download the preconfigured training and testing CSV files and place them in `data/raw/` using these filenames:

* `UNSW_NB15_training-set.csv`
* `UNSW_NB15_testing-set.csv`

The project uses the binary `label` column:

* `0`: normal traffic
* `1`: attack traffic

The `attack_cat` column is excluded from the binary model's input features.

The original dataset should not be modified. Dataset-derived results should be described accurately, and the dataset's required academic citation and usage terms should be respected.
