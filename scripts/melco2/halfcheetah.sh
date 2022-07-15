MAIN_FILE="/home/taghianj/Documents/SAC_GCN/Controller/basic/main.py"

# standard
#tmux new-session -d -s halfcheetah-0 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type standard --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 0"
#tmux new-session -d -s halfcheetah-1 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type standard --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 1"
#tmux new-session -d -s halfcheetah-2 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type standard --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 2"
#tmux new-session -d -s halfcheetah-3 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type standard --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 3"
#tmux new-session -d -s halfcheetah-4 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type standard --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 4"
#tmux new-session -d -s halfcheetah-5 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type standard --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 5"
#tmux new-session -d -s halfcheetah-6 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type standard --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 6"
#tmux new-session -d -s halfcheetah-7 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type standard --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 7"
#tmux new-session -d -s halfcheetah-8 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type standard --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 8"
#tmux new-session -d -s halfcheetah-9 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type standard --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 9"

# root
#tmux new-session -d -s halfcheetah-root-0 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type root --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 0"
#tmux new-session -d -s halfcheetah-root-1 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type root --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 1"
#tmux new-session -d -s halfcheetah-root-2 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type root --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 2"
#tmux new-session -d -s halfcheetah-root-3 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type root --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 3"
#tmux new-session -d -s halfcheetah-root-4 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type root --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 4"
#tmux new-session -d -s halfcheetah-root-5 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type root --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 5"
#tmux new-session -d -s halfcheetah-root-6 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type root --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 6"
#tmux new-session -d -s halfcheetah-root-7 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type root --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 7"
#tmux new-session -d -s halfcheetah-root-8 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type root --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 8"
#tmux new-session -d -s halfcheetah-root-9 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type root --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 9"

# bthigh
#tmux new-session -d -s bthigh-0 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type bthigh --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 0"
#tmux new-session -d -s bthigh-1 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type bthigh --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 1"
#tmux new-session -d -s bthigh-2 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type bthigh --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 2"
#tmux new-session -d -s bthigh-3 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type bthigh --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 3"
#tmux new-session -d -s bthigh-4 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type bthigh --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 4"
#tmux new-session -d -s bthigh-5 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type bthigh --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 5"
#tmux new-session -d -s bthigh-6 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type bthigh --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 6"
#tmux new-session -d -s bthigh-7 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type bthigh --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 7"
#tmux new-session -d -s bthigh-8 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type bthigh --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 8"
#tmux new-session -d -s bthigh-9 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type bthigh --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 9"

# bshin
#tmux new-session -d -s bshin-0 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type bshin --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 0"
#tmux new-session -d -s bshin-1 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type bshin --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 1"
#tmux new-session -d -s bshin-2 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type bshin --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 2"
#tmux new-session -d -s bshin-3 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type bshin --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 3"
#tmux new-session -d -s bshin-4 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type bshin --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 4"
#tmux new-session -d -s bshin-5 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type bshin --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 5"
#tmux new-session -d -s bshin-6 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type bshin --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 6"
#tmux new-session -d -s bshin-7 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type bshin --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 7"
#tmux new-session -d -s bshin-8 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type bshin --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 8"
#tmux new-session -d -s bshin-9 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type bshin --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 9"

# bfoot
#tmux new-session -d -s bfoot-0 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type bfoot --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 0"
#tmux new-session -d -s bfoot-1 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type bfoot --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 1"
#tmux new-session -d -s bfoot-2 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type bfoot --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 2"
#tmux new-session -d -s bfoot-3 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type bfoot --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 3"
#tmux new-session -d -s bfoot-4 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type bfoot --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 4"
#tmux new-session -d -s bfoot-5 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type bfoot --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 5"
#tmux new-session -d -s bfoot-6 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type bfoot --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 6"
#tmux new-session -d -s bfoot-7 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type bfoot --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 7"
#tmux new-session -d -s bfoot-8 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type bfoot --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 8"
#tmux new-session -d -s bfoot-9 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type bfoot --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 9"


# fthigh
#tmux new-session -d -s fthigh-0 "CUDA_VISIBLE_DEVICES=3 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type fthigh --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 0"
#tmux new-session -d -s fthigh-1 "CUDA_VISIBLE_DEVICES=3 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type fthigh --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 1"
#tmux new-session -d -s fthigh-2 "CUDA_VISIBLE_DEVICES=3 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type fthigh --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 2"
#tmux new-session -d -s fthigh-3 "CUDA_VISIBLE_DEVICES=3 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type fthigh --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 3"
#tmux new-session -d -s fthigh-4 "CUDA_VISIBLE_DEVICES=3 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type fthigh --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 4"
#tmux new-session -d -s fthigh-5 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type fthigh --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 5"
#tmux new-session -d -s fthigh-6 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type fthigh --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 6"
#tmux new-session -d -s fthigh-7 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type fthigh --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 7"
#tmux new-session -d -s fthigh-8 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type fthigh --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 8"
#tmux new-session -d -s fthigh-9 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type fthigh --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 9"

# fshin
#tmux new-session -d -s fshin-0 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type fshin --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 0"
#tmux new-session -d -s fshin-1 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type fshin --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 1"
#tmux new-session -d -s fshin-2 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type fshin --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 2"
#tmux new-session -d -s fshin-3 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type fshin --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 3"
#tmux new-session -d -s fshin-4 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type fshin --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 4"
#tmux new-session -d -s fshin-5 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type fshin --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 5"
#tmux new-session -d -s fshin-6 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type fshin --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 6"
#tmux new-session -d -s fshin-7 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type fshin --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 7"
#tmux new-session -d -s fshin-8 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type fshin --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 8"
#tmux new-session -d -s fshin-9 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type fshin --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 9"

# ffoot
#tmux new-session -d -s ffoot-0 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type ffoot --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 0"
#tmux new-session -d -s ffoot-1 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type ffoot --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 1"
#tmux new-session -d -s ffoot-2 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type ffoot --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 2"
#tmux new-session -d -s ffoot-3 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type ffoot --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 3"
#tmux new-session -d -s ffoot-4 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type ffoot --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 4"
#tmux new-session -d -s ffoot-5 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type bfoot --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 5"
#tmux new-session -d -s ffoot-6 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type bfoot --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 6"
#tmux new-session -d -s ffoot-7 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type bfoot --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 7"
#tmux new-session -d -s ffoot-8 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type bfoot --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 8"
#tmux new-session -d -s ffoot-9 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type bfoot --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 9"
