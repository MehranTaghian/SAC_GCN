MAIN_FILE="/local/melco3/taghianj/SAC_GCN/main.py"

# FetchReachEnv-v0
tmux new-session -d -s v0-0 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name FetchReachEnv-v0 --exp-type standard --automatic_entropy_tuning True --num_steps 1000000 --start_steps 10000 -dsf 100 --cuda --seed 0"
tmux new-session -d -s v0-1 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name FetchReachEnv-v0 --exp-type standard --automatic_entropy_tuning True --num_steps 1000000 --start_steps 10000 -dsf 100 --cuda --seed 1"
tmux new-session -d -s v0-2 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name FetchReachEnv-v0 --exp-type standard --automatic_entropy_tuning True --num_steps 1000000 --start_steps 10000 -dsf 100 --cuda --seed 2"
tmux new-session -d -s v0-3 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name FetchReachEnv-v0 --exp-type standard --automatic_entropy_tuning True --num_steps 1000000 --start_steps 10000 -dsf 100 --cuda --seed 3"
tmux new-session -d -s v0-4 "CUDA_VISIBLE_DEVICES=3 python $MAIN_FILE --env-name FetchReachEnv-v0 --exp-type standard --automatic_entropy_tuning True --num_steps 1000000 --start_steps 10000 -dsf 100 --cuda --seed 4"
tmux new-session -d -s v0-5 "CUDA_VISIBLE_DEVICES=3 python $MAIN_FILE --env-name FetchReachEnv-v0 --exp-type standard --automatic_entropy_tuning True --num_steps 1000000 --start_steps 10000 -dsf 100 --cuda --seed 5"
tmux new-session -d -s v0-6 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name FetchReachEnv-v0 --exp-type standard --automatic_entropy_tuning True --num_steps 1000000 --start_steps 10000 -dsf 100 --cuda --seed 6"
tmux new-session -d -s v0-7 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name FetchReachEnv-v0 --exp-type standard --automatic_entropy_tuning True --num_steps 1000000 --start_steps 10000 -dsf 100 --cuda --seed 7"
tmux new-session -d -s v0-8 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachEnv-v0 --exp-type standard --automatic_entropy_tuning True --num_steps 1000000 --start_steps 10000 -dsf 100 --cuda --seed 8"
tmux new-session -d -s v0-9 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachEnv-v0 --exp-type standard --automatic_entropy_tuning True --num_steps 1000000 --start_steps 10000 -dsf 100 --cuda --seed 9"

tmux new-session -d -s v3-0 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name FetchReachEnv-v3 --exp-type standard --automatic_entropy_tuning True --num_steps 1000000 --start_steps 10000 -dsf 100 --cuda --seed 0"
tmux new-session -d -s v3-1 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name FetchReachEnv-v3 --exp-type standard --automatic_entropy_tuning True --num_steps 1000000 --start_steps 10000 -dsf 100 --cuda --seed 1"
tmux new-session -d -s v3-2 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name FetchReachEnv-v3 --exp-type standard --automatic_entropy_tuning True --num_steps 1000000 --start_steps 10000 -dsf 100 --cuda --seed 2"
tmux new-session -d -s v3-3 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name FetchReachEnv-v3 --exp-type standard --automatic_entropy_tuning True --num_steps 1000000 --start_steps 10000 -dsf 100 --cuda --seed 3"
tmux new-session -d -s v3-4 "CUDA_VISIBLE_DEVICES=3 python $MAIN_FILE --env-name FetchReachEnv-v3 --exp-type standard --automatic_entropy_tuning True --num_steps 1000000 --start_steps 10000 -dsf 100 --cuda --seed 4"
tmux new-session -d -s v3-5 "CUDA_VISIBLE_DEVICES=3 python $MAIN_FILE --env-name FetchReachEnv-v3 --exp-type standard --automatic_entropy_tuning True --num_steps 1000000 --start_steps 10000 -dsf 100 --cuda --seed 5"
tmux new-session -d -s v3-6 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name FetchReachEnv-v3 --exp-type standard --automatic_entropy_tuning True --num_steps 1000000 --start_steps 10000 -dsf 100 --cuda --seed 6"
tmux new-session -d -s v3-7 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name FetchReachEnv-v3 --exp-type standard --automatic_entropy_tuning True --num_steps 1000000 --start_steps 10000 -dsf 100 --cuda --seed 7"
tmux new-session -d -s v3-8 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachEnv-v3 --exp-type standard --automatic_entropy_tuning True --num_steps 1000000 --start_steps 10000 -dsf 100 --cuda --seed 8"
tmux new-session -d -s v3-9 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachEnv-v3 --exp-type standard --automatic_entropy_tuning True --num_steps 1000000 --start_steps 10000 -dsf 100 --cuda --seed 9"

#tmux new-session -d -s v4-0 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name FetchReachEnv-v4 --exp-type standard --automatic_entropy_tuning True --num_steps 1000000 --start_steps 10000 -dsf 100 --cuda --seed 0"
#tmux new-session -d -s v4-1 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name FetchReachEnv-v4 --exp-type standard --automatic_entropy_tuning True --num_steps 1000000 --start_steps 10000 -dsf 100 --cuda --seed 1"
#tmux new-session -d -s v4-2 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name FetchReachEnv-v4 --exp-type standard --automatic_entropy_tuning True --num_steps 1000000 --start_steps 10000 -dsf 100 --cuda --seed 2"
#tmux new-session -d -s v4-3 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name FetchReachEnv-v4 --exp-type standard --automatic_entropy_tuning True --num_steps 1000000 --start_steps 10000 -dsf 100 --cuda --seed 3"
#tmux new-session -d -s v4-4 "CUDA_VISIBLE_DEVICES=3 python $MAIN_FILE --env-name FetchReachEnv-v4 --exp-type standard --automatic_entropy_tuning True --num_steps 1000000 --start_steps 10000 -dsf 100 --cuda --seed 4"
#tmux new-session -d -s v4-5 "CUDA_VISIBLE_DEVICES=3 python $MAIN_FILE --env-name FetchReachEnv-v4 --exp-type standard --automatic_entropy_tuning True --num_steps 1000000 --start_steps 10000 -dsf 100 --cuda --seed 5"
#tmux new-session -d -s v4-6 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name FetchReachEnv-v4 --exp-type standard --automatic_entropy_tuning True --num_steps 1000000 --start_steps 10000 -dsf 100 --cuda --seed 6"
#tmux new-session -d -s v4-7 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name FetchReachEnv-v4 --exp-type standard --automatic_entropy_tuning True --num_steps 1000000 --start_steps 10000 -dsf 100 --cuda --seed 7"
#tmux new-session -d -s v4-8 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachEnv-v4 --exp-type standard --automatic_entropy_tuning True --num_steps 1000000 --start_steps 10000 -dsf 100 --cuda --seed 8"
#tmux new-session -d -s v4-9 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReachEnv-v4 --exp-type standard --automatic_entropy_tuning True --num_steps 1000000 --start_steps 10000 -dsf 100 --cuda --seed 9"



#tmux new-session -d -s v0-0 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name HalfCheetahEnv-v0 --exp-type standard --automatic_entropy_tuning True --num_steps 1000000 --start_steps 10000 -dsf 100 --cuda --seed 0"
#tmux new-session -d -s v0-1 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name HalfCheetahEnv-v0 --exp-type standard --automatic_entropy_tuning True --num_steps 1000000 --start_steps 10000 -dsf 100 --cuda --seed 1"
#tmux new-session -d -s v0-2 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name HalfCheetahEnv-v0 --exp-type standard --automatic_entropy_tuning True --num_steps 1000000 --start_steps 10000 -dsf 100 --cuda --seed 2"
#tmux new-session -d -s v0-3 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name HalfCheetahEnv-v0 --exp-type standard --automatic_entropy_tuning True --num_steps 1000000 --start_steps 10000 -dsf 100 --cuda --seed 3"
#tmux new-session -d -s v0-4 "CUDA_VISIBLE_DEVICES=3 python $MAIN_FILE --env-name HalfCheetahEnv-v0 --exp-type standard --automatic_entropy_tuning True --num_steps 1000000 --start_steps 10000 -dsf 100 --cuda --seed 4"
#tmux new-session -d -s v0-5 "CUDA_VISIBLE_DEVICES=3 python $MAIN_FILE --env-name HalfCheetahEnv-v0 --exp-type standard --automatic_entropy_tuning True --num_steps 1000000 --start_steps 10000 -dsf 100 --cuda --seed 5"
#tmux new-session -d -s v0-6 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name HalfCheetahEnv-v0 --exp-type standard --automatic_entropy_tuning True --num_steps 1000000 --start_steps 10000 -dsf 100 --cuda --seed 6"
#tmux new-session -d -s v0-7 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name HalfCheetahEnv-v0 --exp-type standard --automatic_entropy_tuning True --num_steps 1000000 --start_steps 10000 -dsf 100 --cuda --seed 7"
#tmux new-session -d -s v0-8 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name HalfCheetahEnv-v0 --exp-type standard --automatic_entropy_tuning True --num_steps 1000000 --start_steps 10000 -dsf 100 --cuda --seed 8"
#tmux new-session -d -s v0-9 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name HalfCheetahEnv-v0 --exp-type standard --automatic_entropy_tuning True --num_steps 1000000 --start_steps 10000 -dsf 100 --cuda --seed 9"



