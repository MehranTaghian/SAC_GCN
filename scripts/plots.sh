PLOT_FILE=/home/mehran/Documents/SAC_GCN/Plots/learning_curve.py
PLOT_MULTIPLE_FILE=/home/mehran/Documents/SAC_GCN/Plots/multiple_type_learning_curves.py
PLOT_LOSS=/home/mehran/Documents/SAC_GCN/Plots/loss.py

python $PLOT_FILE --env-name FetchReachEnv-v0 --exp-type standard
python $PLOT_FILE --env-name FetchReachEnv-v1 --exp-type wrist_flex_joint
python $PLOT_FILE --env-name Walker2d-v2 --exp-type standard

python $PLOT_FILE --env-name AntEnv-v0 --exp-type standard

python $PLOT_FILE --env-name HalfCheetahEnv-v0 --exp-type standard

python $PLOT_MULTIPLE_FILE --env-name FetchReachDense-v1
python $PLOT_MULTIPLE_FILE --env-name HalfCheetah-v2
python $PLOT_MULTIPLE_FILE --env-name Walker2d-v2
python $PLOT_MULTIPLE_FILE --env-name Hopper-v2


python $PLOT_LOSS --env-name FetchReachEnv-v0 --exp-type standard
