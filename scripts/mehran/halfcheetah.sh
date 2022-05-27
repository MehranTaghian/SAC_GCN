MAIN_FILE="/home/taghianj/Documents/SAC_GCN/Controller/basic/main.py"

# standard
#tmux new-session -d -s halfcheetahv0-0 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name HalfCheetahEnv-v0 --exp-type standard --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 0"
#tmux new-session -d -s halfcheetahv0-1 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name HalfCheetahEnv-v0 --exp-type standard --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 1"
#tmux new-session -d -s halfcheetahv0-2 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name HalfCheetahEnv-v0 --exp-type standard --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 2"
#tmux new-session -d -s halfcheetahv0-3 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name HalfCheetahEnv-v0 --exp-type standard --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 3"
#tmux new-session -d -s halfcheetahv0-4 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name HalfCheetahEnv-v0 --exp-type standard --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 4"
#tmux new-session -d -s halfcheetahv0-5 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name HalfCheetahEnv-v0 --exp-type standard --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 5"
#tmux new-session -d -s halfcheetahv0-6 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name HalfCheetahEnv-v0 --exp-type standard --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 6"
#tmux new-session -d -s halfcheetahv0-7 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name HalfCheetahEnv-v0 --exp-type standard --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 7"
#tmux new-session -d -s halfcheetahv0-8 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name HalfCheetahEnv-v0 --exp-type standard --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 8"
#tmux new-session -d -s halfcheetahv0-9 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name HalfCheetahEnv-v0 --exp-type standard --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 9"

# bfoot
#tmux new-session -d -s bfoot-v0-0 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name HalfCheetahEnv-v0 --exp-type bfoot --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 0"
#tmux new-session -d -s bfoot-v0-1 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name HalfCheetahEnv-v0 --exp-type bfoot --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 1"
#tmux new-session -d -s bfoot-v0-2 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name HalfCheetahEnv-v0 --exp-type bfoot --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 2"
#tmux new-session -d -s bfoot-v0-3 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name HalfCheetahEnv-v0 --exp-type bfoot --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 3"
#tmux new-session -d -s bfoot-v0-4 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name HalfCheetahEnv-v0 --exp-type bfoot --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 4"
#tmux new-session -d -s bfoot-v0-5 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name HalfCheetahEnv-v0 --exp-type bfoot --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 5"
#tmux new-session -d -s bfoot-v0-6 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name HalfCheetahEnv-v0 --exp-type bfoot --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 6"
#tmux new-session -d -s bfoot-v0-7 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name HalfCheetahEnv-v0 --exp-type bfoot --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 7"
#tmux new-session -d -s bfoot-v0-8 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name HalfCheetahEnv-v0 --exp-type bfoot --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 8"
#tmux new-session -d -s bfoot-v0-9 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name HalfCheetahEnv-v0 --exp-type bfoot --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 9"

# fthigh
#tmux new-session -d -s fthigh-v0-0 "CUDA_VISIBLE_DEVICES=3 python $MAIN_FILE --env-name HalfCheetahEnv-v0 --exp-type fthigh --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 0"
#tmux new-session -d -s fthigh-v0-1 "CUDA_VISIBLE_DEVICES=3 python $MAIN_FILE --env-name HalfCheetahEnv-v0 --exp-type fthigh --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 1"
#tmux new-session -d -s fthigh-v0-2 "CUDA_VISIBLE_DEVICES=3 python $MAIN_FILE --env-name HalfCheetahEnv-v0 --exp-type fthigh --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 2"
#tmux new-session -d -s fthigh-v0-3 "CUDA_VISIBLE_DEVICES=3 python $MAIN_FILE --env-name HalfCheetahEnv-v0 --exp-type fthigh --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 3"
#tmux new-session -d -s fthigh-v0-4 "CUDA_VISIBLE_DEVICES=3 python $MAIN_FILE --env-name HalfCheetahEnv-v0 --exp-type fthigh --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 4"
tmux new-session -d -s fthigh-v0-5 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name HalfCheetahEnv-v0 --exp-type fthigh --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 5"
tmux new-session -d -s fthigh-v0-6 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name HalfCheetahEnv-v0 --exp-type fthigh --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 6"
tmux new-session -d -s fthigh-v0-7 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name HalfCheetahEnv-v0 --exp-type fthigh --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 7"
tmux new-session -d -s fthigh-v0-8 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name HalfCheetahEnv-v0 --exp-type fthigh --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 8"
tmux new-session -d -s fthigh-v0-9 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name HalfCheetahEnv-v0 --exp-type fthigh --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 9"
