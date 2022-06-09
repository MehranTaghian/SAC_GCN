MAIN_FILE="/local/melco3/taghianj/SAC_GCN/Controller/basic/main.py"

# FetchReachEnv-v0
#tmux new-session -d -s v0-0 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type important-joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 20000 -dsf 100 --cuda --seed 0"
#tmux new-session -d -s v0-1 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type important-joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 20000 -dsf 100 --cuda --seed 1"
#tmux new-session -d -s v0-2 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type important-joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 20000 -dsf 100 --cuda --seed 2"
#tmux new-session -d -s v0-3 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type important-joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 20000 -dsf 100 --cuda --seed 3"
#tmux new-session -d -s v0-4 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type important-joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 20000 -dsf 100 --cuda --seed 4"
#tmux new-session -d -s v0-5 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type important-joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 20000 -dsf 100 --cuda --seed 5"
#tmux new-session -d -s v0-6 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type important-joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 20000 -dsf 100 --cuda --seed 6"
#tmux new-session -d -s v0-7 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type important-joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 20000 -dsf 100 --cuda --seed 7"
#tmux new-session -d -s v0-8 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type important-joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 20000 -dsf 100 --cuda --seed 8"
#tmux new-session -d -s v0-9 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type important-joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 20000 -dsf 100 --cuda --seed 9"


tmux new-session -d -s occluded-v0-0 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type non-important-joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 20000 -dsf 100 --cuda --seed 0"
tmux new-session -d -s occluded-v0-1 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type non-important-joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 20000 -dsf 100 --cuda --seed 1"
tmux new-session -d -s occluded-v0-2 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type non-important-joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 20000 -dsf 100 --cuda --seed 2"
tmux new-session -d -s occluded-v0-3 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type non-important-joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 20000 -dsf 100 --cuda --seed 3"
tmux new-session -d -s occluded-v0-4 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type non-important-joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 20000 -dsf 100 --cuda --seed 4"
tmux new-session -d -s occluded-v0-5 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type non-important-joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 20000 -dsf 100 --cuda --seed 5"
tmux new-session -d -s occluded-v0-6 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type non-important-joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 20000 -dsf 100 --cuda --seed 6"
tmux new-session -d -s occluded-v0-7 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type non-important-joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 20000 -dsf 100 --cuda --seed 7"
tmux new-session -d -s occluded-v0-8 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type non-important-joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 20000 -dsf 100 --cuda --seed 8"
tmux new-session -d -s occluded-v0-9 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachDense-v1 --exp-type non-important-joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 20000 -dsf 100 --cuda --seed 9"

