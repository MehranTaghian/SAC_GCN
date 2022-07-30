MAIN_FILE="/home/taghianj/Documents/SAC_GCN/Controller/basic/main.py"

# SHOULDER PAN JOINT
#tmux new-session -d -s brokenshoulderpan-00 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type shoulder_pan_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 0"
#tmux new-session -d -s brokenshoulderpan-01 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type shoulder_pan_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 1"
#tmux new-session -d -s brokenshoulderpan-02 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type shoulder_pan_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 2"
#tmux new-session -d -s brokenshoulderpan-03 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type shoulder_pan_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 3"
#tmux new-session -d -s brokenshoulderpan-04 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type shoulder_pan_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 4"
#tmux new-session -d -s brokenshoulderpan-05 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type shoulder_pan_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 5"
#tmux new-session -d -s brokenshoulderpan-06 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type shoulder_pan_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 6"
#tmux new-session -d -s brokenshoulderpan-07 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type shoulder_pan_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 7"
#tmux new-session -d -s brokenshoulderpan-08 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type shoulder_pan_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 8"
#tmux new-session -d -s brokenshoulderpan-09 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type shoulder_pan_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 9"

# SHOULDER LIFT JOINT
#tmux new-session -d -s brokenshoulderlift-00 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type shoulder_lift_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 0"
#tmux new-session -d -s brokenshoulderlift-01 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type shoulder_lift_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 1"
#tmux new-session -d -s brokenshoulderlift-02 "CUDA_VISIBLE_DEVICES=3 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type shoulder_lift_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 2"
#tmux new-session -d -s brokenshoulderlift-03 "CUDA_VISIBLE_DEVICES=3 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type shoulder_lift_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 3"
#tmux new-session -d -s brokenshoulderlift-04 "CUDA_VISIBLE_DEVICES=3 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type shoulder_lift_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 4"
#tmux new-session -d -s brokenshoulderlift-05 "CUDA_VISIBLE_DEVICES=3 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type shoulder_lift_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 5"
#tmux new-session -d -s brokenshoulderlift-06 "CUDA_VISIBLE_DEVICES=3 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type shoulder_lift_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 6"
#tmux new-session -d -s brokenshoulderlift-07 "CUDA_VISIBLE_DEVICES=3 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type shoulder_lift_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 7"
#tmux new-session -d -s brokenshoulderlift-08 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type shoulder_lift_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 8"
#tmux new-session -d -s brokenshoulderlift-09 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type shoulder_lift_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 9"

# UPPERARM ROLL JOINT
#tmux new-session -d -s brokenupperarm-00 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type upperarm_roll_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 0"
#tmux new-session -d -s brokenupperarm-01 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type upperarm_roll_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 1"
#tmux new-session -d -s brokenupperarm-02 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type upperarm_roll_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 2"
#tmux new-session -d -s brokenupperarm-03 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type upperarm_roll_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 3"
#tmux new-session -d -s brokenupperarm-04 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type upperarm_roll_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 4"
#tmux new-session -d -s brokenupperarm-05 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type upperarm_roll_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 5"
#tmux new-session -d -s brokenupperarm-06 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type upperarm_roll_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 6"
#tmux new-session -d -s brokenupperarm-07 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type upperarm_roll_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 7"
#tmux new-session -d -s brokenupperarm-08 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type upperarm_roll_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 8"
#tmux new-session -d -s brokenupperarm-09 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type upperarm_roll_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 9"

# ELBOW FLEX JOINT
#tmux new-session -d -s brokenelbowflex-00 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type elbow_flex_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 0"
#tmux new-session -d -s brokenelbowflex-01 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type elbow_flex_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 1"
#tmux new-session -d -s brokenelbowflex-02 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type elbow_flex_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 2"
#tmux new-session -d -s brokenelbowflex-03 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type elbow_flex_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 3"
#tmux new-session -d -s brokenelbowflex-04 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type elbow_flex_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 4"
#tmux new-session -d -s brokenelbowflex-05 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type elbow_flex_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 5"
tmux new-session -d -s brokenelbowflex-06 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type elbow_flex_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 6"
tmux new-session -d -s brokenelbowflex-07 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type elbow_flex_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 7"
tmux new-session -d -s brokenelbowflex-08 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type elbow_flex_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 8"
tmux new-session -d -s brokenelbowflex-09 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type elbow_flex_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 9"

# FOREARM ROLL JOINT
tmux new-session -d -s brokenforearm-00 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type forearm_roll_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 0"
tmux new-session -d -s brokenforearm-01 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type forearm_roll_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 1"
tmux new-session -d -s brokenforearm-02 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type forearm_roll_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 2"
tmux new-session -d -s brokenforearm-03 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type forearm_roll_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 3"
tmux new-session -d -s brokenforearm-04 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type forearm_roll_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 4"
tmux new-session -d -s brokenforearm-05 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type forearm_roll_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 5"
tmux new-session -d -s brokenforearm-06 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type forearm_roll_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 6"
tmux new-session -d -s brokenforearm-07 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type forearm_roll_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 7"
tmux new-session -d -s brokenforearm-08 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type forearm_roll_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 8"
tmux new-session -d -s brokenforearm-09 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type forearm_roll_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 9"

# WRIST FLEX JOINT
tmux new-session -d -s brokenwristflex-00 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type wrist_flex_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 0"
tmux new-session -d -s brokenwristflex-01 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type wrist_flex_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 1"
tmux new-session -d -s brokenwristflex-02 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type wrist_flex_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 2"
tmux new-session -d -s brokenwristflex-03 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type wrist_flex_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 3"
#tmux new-session -d -s brokenwristflex-04 "CUDA_VISIBLE_DEVICES=3 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type wrist_flex_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 4"
#tmux new-session -d -s brokenwristflex-05 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type wrist_flex_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 5"
#tmux new-session -d -s brokenwristflex-06 "CUDA_VISIBLE_DEVICES=3 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type wrist_flex_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 6"
#tmux new-session -d -s brokenwristflex-07 "CUDA_VISIBLE_DEVICES=3 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type wrist_flex_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 7"
#tmux new-session -d -s brokenwristflex-08 "CUDA_VISIBLE_DEVICES=3 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type wrist_flex_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 8"
#tmux new-session -d -s brokenwristflex-09 "CUDA_VISIBLE_DEVICES=3 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type wrist_flex_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 9"

# WRIST ROLL JOINT
#tmux new-session -d -s brokenwristroll-00 "CUDA_VISIBLE_DEVICES=3 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type wrist_roll_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 0"
#tmux new-session -d -s brokenwristroll-01 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type wrist_roll_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 1"
#tmux new-session -d -s brokenwristroll-02 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type wrist_roll_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 2"
#tmux new-session -d -s brokenwristroll-03 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type wrist_roll_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 3"
#tmux new-session -d -s brokenwristroll-04 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type wrist_roll_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 4"
#tmux new-session -d -s brokenwristroll-05 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type wrist_roll_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 5"
#tmux new-session -d -s brokenwristroll-06 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type wrist_roll_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 6"
#tmux new-session -d -s brokenwristroll-07 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type wrist_roll_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 7"
#tmux new-session -d -s brokenwristroll-08 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type wrist_roll_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 8"
#tmux new-session -d -s brokenwristroll-09 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachBroken-v2 --exp-type wrist_roll_joint --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 9"
