PLOT_FILE=/home/mehran/Documents/SAC_GCN/Plots/learning_curve.py
PLOT_LOSS=/home/mehran/Documents/SAC_GCN/Plots/loss.py

PLOT_FILE=/home/taghianj/Documents/SAC_GCN/Plots/learning_curve.py

PLOT=/home/mehran/Documents/SAC_GCN/Plots/plot.py

PLOT=/home/taghianj/Documents/SAC_GCN/Plots/plot.py
PLOT=/home/taghianj/Documents/SAC_GCN/Plots/plot_paper.py
PLOT=/home/taghianj/Documents/SAC_GCN/Plots/plot_separately.py


python $PLOT --env-name FetchReach-v2 --epsilon 1 --window-size 500 --epsilon-entity 0.0 --epsilon-action 10000
python $PLOT --env-name Walker2d-v2 --epsilon 50 --window-size 100 --epsilon-entity 0.05 --epsilon-action 1500
python $PLOT --env-name HalfCheetah-v2
python $PLOT --env-name Hopper-v2

