PLOT_FILE=/home/mehran/Documents/SAC_GCN/Plots/learning_curve.py
PLOT_LOSS=/home/mehran/Documents/SAC_GCN/Plots/loss.py

PLOT_FILE=/home/taghianj/Documents/SAC_GCN/Plots/learning_curve.py

PLOT=/home/taghianj/Documents/SAC_GCN/Plots/plot.py


python $PLOT --env-name FetchReach-v2
python $PLOT --env-name HalfCheetah-v2
python $PLOT --env-name Walker2d-v2
python $PLOT --env-name Hopper-v2

python $PLOT_FILE --env-name FetchReach-v2

