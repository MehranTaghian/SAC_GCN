PLOT_FILE=/home/mehran/Documents/SAC_GCN/Plots/learning_curve.py
PLOT_MULTIPLE_FILE=/home/mehran/Documents/SAC_GCN/Plots/multiple_type_learning_curves.py

#python $PLOT_FILE --env-name FetchReachEnv-v1 --exp-type standard
#python $PLOT_FILE --env-name FetchReachEnv-v1 --exp-type wrist_flex_joint

python $PLOT_MULTIPLE_FILE --env-name FetchReachEnv-v1

