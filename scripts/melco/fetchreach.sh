MAIN_FILE="/local/melco3/taghianj/SAC_GCN/Controller/basic/main.py"

# Standard
tmux new-session -d -s fetchreach-0 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type standard --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 0"
tmux new-session -d -s fetchreach-1 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type standard --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 1"
tmux new-session -d -s fetchreach-2 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type standard --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 2"
tmux new-session -d -s fetchreach-3 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type standard --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 3"
tmux new-session -d -s fetchreach-4 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type standard --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 4"
tmux new-session -d -s fetchreach-5 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type standard --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 5"
tmux new-session -d -s fetchreach-6 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type standard --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 6"
tmux new-session -d -s fetchreach-7 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type standard --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 7"
tmux new-session -d -s fetchreach-8 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type standard --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 8"
tmux new-session -d -s fetchreach-9 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type standard --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 9"

# SHOULD PAN JOINT
tmux new-session -d -s shoulderpan-0 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type shoulder_pan_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 0"
tmux new-session -d -s shoulderpan-1 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type shoulder_pan_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 1"
tmux new-session -d -s shoulderpan-2 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type shoulder_pan_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 2"
tmux new-session -d -s shoulderpan-3 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type shoulder_pan_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 3"
tmux new-session -d -s shoulderpan-4 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type shoulder_pan_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 4"
tmux new-session -d -s shoulderpan-5 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type shoulder_pan_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 5"
tmux new-session -d -s shoulderpan-6 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type shoulder_pan_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 6"
tmux new-session -d -s shoulderpan-7 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type shoulder_pan_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 7"
tmux new-session -d -s shoulderpan-8 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type shoulder_pan_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 8"
tmux new-session -d -s shoulderpan-9 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type shoulder_pan_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 9"

# SHOULDER LIFT JOINT
tmux new-session -d -s shoulderlift-0 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type shoulder_lift_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 0"
tmux new-session -d -s shoulderlift-1 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type shoulder_lift_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 1"
tmux new-session -d -s shoulderlift-2 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type shoulder_lift_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 2"
tmux new-session -d -s shoulderlift-3 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type shoulder_lift_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 3"
tmux new-session -d -s shoulderlift-4 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type shoulder_lift_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 4"
tmux new-session -d -s shoulderlift-5 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type shoulder_lift_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 5"
tmux new-session -d -s shoulderlift-6 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type shoulder_lift_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 6"
tmux new-session -d -s shoulderlift-7 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type shoulder_lift_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 7"
tmux new-session -d -s shoulderlift-8 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type shoulder_lift_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 8"
tmux new-session -d -s shoulderlift-9 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type shoulder_lift_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 9"

# UPPERARM ROLL JOINT
tmux new-session -d -s upperarm-0 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type upperarm_roll_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 0"
tmux new-session -d -s upperarm-1 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type upperarm_roll_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 1"
tmux new-session -d -s upperarm-2 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type upperarm_roll_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 2"
tmux new-session -d -s upperarm-3 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type upperarm_roll_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 3"
tmux new-session -d -s upperarm-4 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type upperarm_roll_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 4"
tmux new-session -d -s upperarm-5 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type upperarm_roll_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 5"
tmux new-session -d -s upperarm-6 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type upperarm_roll_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 6"
tmux new-session -d -s upperarm-7 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type upperarm_roll_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 7"
tmux new-session -d -s upperarm-8 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type upperarm_roll_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 8"
tmux new-session -d -s upperarm-9 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type upperarm_roll_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 9"

# ELBOW FLEX JOINT
tmux new-session -d -s elbowflex-0 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type elbow_flex_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 0"
tmux new-session -d -s elbowflex-1 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type elbow_flex_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 1"
tmux new-session -d -s elbowflex-2 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type elbow_flex_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 2"
tmux new-session -d -s elbowflex-3 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type elbow_flex_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 3"
tmux new-session -d -s elbowflex-4 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type elbow_flex_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 4"
tmux new-session -d -s elbowflex-5 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type elbow_flex_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 5"
tmux new-session -d -s elbowflex-6 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type elbow_flex_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 6"
tmux new-session -d -s elbowflex-7 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type elbow_flex_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 7"
tmux new-session -d -s elbowflex-8 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type elbow_flex_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 8"
tmux new-session -d -s elbowflex-9 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type elbow_flex_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 9"

# FOREARM ROLL JOINT
tmux new-session -d -s forearm-0 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type forearm_roll_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 0"
tmux new-session -d -s forearm-1 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type forearm_roll_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 1"
tmux new-session -d -s forearm-2 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type forearm_roll_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 2"
tmux new-session -d -s forearm-3 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type forearm_roll_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 3"
tmux new-session -d -s forearm-4 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type forearm_roll_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 4"
tmux new-session -d -s forearm-5 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type forearm_roll_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 5"
tmux new-session -d -s forearm-6 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type forearm_roll_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 6"
tmux new-session -d -s forearm-7 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type forearm_roll_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 7"
tmux new-session -d -s forearm-8 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type forearm_roll_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 8"
tmux new-session -d -s forearm-9 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type forearm_roll_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 9"

# WRIST FLEX JOINT
tmux new-session -d -s wristflex-0 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type wrist_flex_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 0"
tmux new-session -d -s wristflex-1 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type wrist_flex_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 1"
tmux new-session -d -s wristflex-2 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type wrist_flex_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 2"
tmux new-session -d -s wristflex-3 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type wrist_flex_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 3"
tmux new-session -d -s wristflex-4 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type wrist_flex_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 4"
tmux new-session -d -s wristflex-5 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type wrist_flex_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 5"
tmux new-session -d -s wristflex-6 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type wrist_flex_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 6"
tmux new-session -d -s wristflex-7 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type wrist_flex_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 7"
tmux new-session -d -s wristflex-8 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type wrist_flex_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 8"
tmux new-session -d -s wristflex-9 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type wrist_flex_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 9"

# WRIST ROLL JOINT
tmux new-session -d -s wristroll-0 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type wrist_roll_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 0"
tmux new-session -d -s wristroll-1 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type wrist_roll_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 1"
tmux new-session -d -s wristroll-2 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type wrist_roll_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 2"
tmux new-session -d -s wristroll-3 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type wrist_roll_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 3"
tmux new-session -d -s wristroll-4 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type wrist_roll_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 4"
tmux new-session -d -s wristroll-5 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type wrist_roll_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 5"
tmux new-session -d -s wristroll-6 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type wrist_roll_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 6"
tmux new-session -d -s wristroll-7 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type wrist_roll_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 7"
tmux new-session -d -s wristroll-8 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type wrist_roll_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 8"
tmux new-session -d -s wristroll-9 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type wrist_roll_joint --automatic_entropy_tuning True --num_episodes 20000 --start_steps 10000 -dsf 100 --cuda --seed 9"
