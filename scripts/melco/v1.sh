MAIN_FILE="/local/melco3/taghianj/SAC_GCN/main.py"


# FetchReachEnv-v1
#tmux new-session -d -s v1-0 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name FetchReachEnv-v1 --exp-type standard --automatic_entropy_tuning True --num_steps 250000 --start_steps 10000 -dsf 100 --cuda --seed 0"
#tmux new-session -d -s v1-1 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name FetchReachEnv-v1 --exp-type standard --automatic_entropy_tuning True --num_steps 250000 --start_steps 10000 -dsf 100 --cuda --seed 1"
#tmux new-session -d -s v1-2 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name FetchReachEnv-v1 --exp-type standard --automatic_entropy_tuning True --num_steps 250000 --start_steps 10000 -dsf 100 --cuda --seed 2"
#tmux new-session -d -s v1-3 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name FetchReachEnv-v1 --exp-type standard --automatic_entropy_tuning True --num_steps 250000 --start_steps 10000 -dsf 100 --cuda --seed 3"
#tmux new-session -d -s v1-4 "CUDA_VISIBLE_DEVICES=3 python $MAIN_FILE --env-name FetchReachEnv-v1 --exp-type standard --automatic_entropy_tuning True --num_steps 250000 --start_steps 10000 -dsf 100 --cuda --seed 4"
#tmux new-session -d -s v1-5 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name FetchReachEnv-v1 --exp-type standard --automatic_entropy_tuning True --num_steps 250000 --start_steps 10000 -dsf 100 --cuda --seed 5"
#tmux new-session -d -s v1-6 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name FetchReachEnv-v1 --exp-type standard --automatic_entropy_tuning True --num_steps 250000 --start_steps 10000 -dsf 100 --cuda --seed 6"
#tmux new-session -d -s v1-7 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachEnv-v1 --exp-type standard --automatic_entropy_tuning True --num_steps 250000 --start_steps 10000 -dsf 100 --cuda --seed 7"
#tmux new-session -d -s v1-8 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachEnv-v1 --exp-type standard --automatic_entropy_tuning True --num_steps 250000 --start_steps 10000 -dsf 100 --cuda --seed 8"
#tmux new-session -d -s v1-9 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name FetchReachEnv-v1 --exp-type standard --automatic_entropy_tuning True --num_steps 250000 --start_steps 10000 -dsf 100 --cuda --seed 9"


#tmux new-session -d -s v1-0 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name FetchReachEnv-v1 --exp-type wrist_flex_joint --automatic_entropy_tuning True --num_steps 250000 --start_steps 10000 -dsf 100 --cuda --seed 0"
#tmux new-session -d -s v1-1 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name FetchReachEnv-v1 --exp-type wrist_flex_joint --automatic_entropy_tuning True --num_steps 250000 --start_steps 10000 -dsf 100 --cuda --seed 1"
#tmux new-session -d -s v1-2 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name FetchReachEnv-v1 --exp-type wrist_flex_joint --automatic_entropy_tuning True --num_steps 250000 --start_steps 10000 -dsf 100 --cuda --seed 2"
#tmux new-session -d -s v1-3 "CUDA_VISIBLE_DEVICES=3 python $MAIN_FILE --env-name FetchReachEnv-v1 --exp-type wrist_flex_joint --automatic_entropy_tuning True --num_steps 250000 --start_steps 10000 -dsf 100 --cuda --seed 3"
#tmux new-session -d -s v1-4 "CUDA_VISIBLE_DEVICES=3 python $MAIN_FILE --env-name FetchReachEnv-v1 --exp-type wrist_flex_joint --automatic_entropy_tuning True --num_steps 250000 --start_steps 10000 -dsf 100 --cuda --seed 4"
#tmux new-session -d -s v1-5 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name FetchReachEnv-v1 --exp-type wrist_flex_joint --automatic_entropy_tuning True --num_steps 250000 --start_steps 10000 -dsf 100 --cuda --seed 5"
#tmux new-session -d -s v1-6 "CUDA_VISIBLE_DEVICES=5 python $MAIN_FILE --env-name FetchReachEnv-v1 --exp-type wrist_flex_joint --automatic_entropy_tuning True --num_steps 250000 --start_steps 10000 -dsf 100 --cuda --seed 6"
#tmux new-session -d -s v1-7 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachEnv-v1 --exp-type wrist_flex_joint --automatic_entropy_tuning True --num_steps 250000 --start_steps 10000 -dsf 100 --cuda --seed 7"
#tmux new-session -d -s v1-8 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name FetchReachEnv-v1 --exp-type wrist_flex_joint --automatic_entropy_tuning True --num_steps 250000 --start_steps 10000 -dsf 100 --cuda --seed 8"
#tmux new-session -d -s v1-9 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name FetchReachEnv-v1 --exp-type wrist_flex_joint --automatic_entropy_tuning True --num_steps 250000 --start_steps 10000 -dsf 100 --cuda --seed 9"


tmux new-session -d -s v1-0 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name FetchReachEnv-v1 --exp-type elbow_flex_joint --automatic_entropy_tuning True --num_steps 250000 --start_steps 10000 -dsf 100 --cuda --seed 0"
tmux new-session -d -s v1-1 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name FetchReachEnv-v1 --exp-type elbow_flex_joint --automatic_entropy_tuning True --num_steps 250000 --start_steps 10000 -dsf 100 --cuda --seed 1"
#tmux new-session -d -s v1-2 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name FetchReachEnv-v1 --exp-type elbow_flex_joint --automatic_entropy_tuning True --num_steps 250000 --start_steps 10000 -dsf 100 --cuda --seed 2"
#tmux new-session -d -s v1-3 "CUDA_VISIBLE_DEVICES=3 python $MAIN_FILE --env-name FetchReachEnv-v1 --exp-type elbow_flex_joint --automatic_entropy_tuning True --num_steps 250000 --start_steps 10000 -dsf 100 --cuda --seed 3"
#tmux new-session -d -s v1-4 "CUDA_VISIBLE_DEVICES=3 python $MAIN_FILE --env-name FetchReachEnv-v1 --exp-type elbow_flex_joint --automatic_entropy_tuning True --num_steps 250000 --start_steps 10000 -dsf 100 --cuda --seed 4"
#tmux new-session -d -s v1-5 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name FetchReachEnv-v1 --exp-type elbow_flex_joint --automatic_entropy_tuning True --num_steps 250000 --start_steps 10000 -dsf 100 --cuda --seed 5"
#tmux new-session -d -s v1-6 "CUDA_VISIBLE_DEVICES=5 python $MAIN_FILE --env-name FetchReachEnv-v1 --exp-type elbow_flex_joint --automatic_entropy_tuning True --num_steps 250000 --start_steps 10000 -dsf 100 --cuda --seed 6"
#tmux new-session -d -s v1-7 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachEnv-v1 --exp-type elbow_flex_joint --automatic_entropy_tuning True --num_steps 250000 --start_steps 10000 -dsf 100 --cuda --seed 7"
#tmux new-session -d -s v1-8 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name FetchReachEnv-v1 --exp-type elbow_flex_joint --automatic_entropy_tuning True --num_steps 250000 --start_steps 10000 -dsf 100 --cuda --seed 8"
#tmux new-session -d -s v1-9 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name FetchReachEnv-v1 --exp-type elbow_flex_joint --automatic_entropy_tuning True --num_steps 250000 --start_steps 10000 -dsf 100 --cuda --seed 9"
