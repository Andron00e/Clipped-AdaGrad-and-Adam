# 1. run experiments for hyperparameter tuning
bash run_all_sgd.sh
bash run_all_adam.sh

# 2. run multiple runs to compare clipping methods
bash run_all_multirun.sh

# 3. generate histograms
bash run_all_check_tails.sh

# 4. then go to ALBERT_fine_tuning/visualization.ipynb to plot visualizations
