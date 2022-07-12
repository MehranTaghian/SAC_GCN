PLOT_FILE=/home/mehran/Documents/SAC_GCN/Plots/learning_curve.py
PLOT_MULTIPLE_FILE=/home/mehran/Documents/SAC_GCN/Plots/multiple_type_learning_curves.py
PLOT_LOSS=/home/mehran/Documents/SAC_GCN/Plots/loss.py

PLOT_MULTIPLE_FILE=/home/taghianj/Documents/SAC_GCN/Plots/multiple_type_learning_curves.py
PLOT_FILE=/home/taghianj/Documents/SAC_GCN/Plots/learning_curve.py


python $PLOT_MULTIPLE_FILE --env-name FetchReachDense-v1
python $PLOT_MULTIPLE_FILE --env-name FetchReachBroken-v1
python $PLOT_MULTIPLE_FILE --env-name HalfCheetah-v2
python $PLOT_MULTIPLE_FILE --env-name Walker2d-v2
python $PLOT_MULTIPLE_FILE --env-name Hopper-v2

python $PLOT_FILE --env-name FetchReachDense-v1

