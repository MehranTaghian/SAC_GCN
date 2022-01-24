MAIN_FILE_SINGLE=/home/taghianj/Documents/SAC_GCN/Evaluate/evaluate_single.py
MAIN_FILE_MULTIPLE=/home/taghianj/Documents/SAC_GCN/Evaluate/evaluate_multiple.py

#python $MAIN_FILE_SINGLE --env-name FetchReachEnv-v0 --exp-type standard


python $MAIN_FILE_MULTIPLE --env-name FetchReachEnv-v0 --exp-type standard
#python $MAIN_FILE_MULTIPLE --env-name FetchReachEnv-v1 --exp-type standard
#python $MAIN_FILE_MULTIPLE --env-name FetchReachEnv-v1 --exp-type wrist_flex_joint
#python $MAIN_FILE_MULTIPLE --env-name FetchReachEnv-v1 --exp-type elbow_flex_joint

