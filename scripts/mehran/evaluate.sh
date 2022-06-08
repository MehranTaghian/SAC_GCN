MAIN_FILE_SINGLE=/home/mehran/Documents/SAC_GCN/Evaluate/evaluate_single_test.py
MAIN_FILE_MULTIPLE=/home/mehran/Documents/SAC_GCN/Evaluate/evaluate_multiple_test.py
EVALUATE=/home/mehran/Documents/SAC_GCN/Evaluate/evaluate.py

python $EVALUATE --env-name FetchReachEnvGraph-v0 --exp-type standard --seed 0
python $EVALUATE --env-name FetchReachEnvGraph-v0 --exp-type standard

python $EVALUATE --env-name HalfCheetah-v2 --exp-type graph
python $EVALUATE --env-name Walker2d-v2 --exp-type graph

python $MAIN_FILE_SINGLE --env-name Walker2d-v2 --exp-type graph --seed 0

python $MAIN_FILE_MULTIPLE --env-name FetchReachEnvGraph-v0 --exp-type standard
