MAIN_FILE_SINGLE=/home/mehran/Documents/SAC_GCN/Evaluate/evaluate_single.py
MAIN_FILE_MULTIPLE=/home/mehran/Documents/SAC_GCN/Evaluate/evaluate_multiple.py

python $MAIN_FILE_MULTIPLE --env-name FetchReachEnv-v0 --exp-type standard --seed 0
python $MAIN_FILE_MULTIPLE --env-name FetchReachEnv-v1 --exp-type standard --seed 0

python $MAIN_FILE_SINGLE --env-name HalfCheetahEnv-v0 --exp-type standard --seed 0
python $MAIN_FILE_SINGLE --env-name AntEnv-v0 --exp-type standard --seed 0 --aggregation sum


#python $MAIN_FILE_MULTIPLE --env-name FetchReachEnv-v1 --exp-type wrist_flex_joint
#python $MAIN_FILE_MULTIPLE --env-name FetchReachEnv-v1 --exp-type elbow_flex_joint

