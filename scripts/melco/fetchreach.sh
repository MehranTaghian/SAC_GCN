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

