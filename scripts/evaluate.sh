MAIN_FILE_SINGLE=/home/mehran/Documents/SAC_GCN/Evaluate/evaluate_single_test.py
MAIN_FILE_MULTIPLE=/home/mehran/Documents/SAC_GCN/Evaluate/evaluate_multiple_test.py
EVALUATE=/home/mehran/Documents/SAC_GCN/Evaluate/evaluate.py

EVALUATE=/home/taghianj/Documents/SAC_GCN/Evaluate/evaluate.py

python $EVALUATE --env-name FetchReach-v2 --exp-type graph --time-step 29999
python $EVALUATE --env-name HalfCheetah-v2 --exp-type graph
python $EVALUATE --env-name Walker2d-v2 --exp-type graph
python $EVALUATE --env-name Hopper-v2 --exp-type graph

python $EVALUATE --env-name HalfCheetah-v2 --exp-type graph_new
python $EVALUATE --env-name Walker2d-v2 --exp-type graph_new