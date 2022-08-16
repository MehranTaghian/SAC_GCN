MAIN_FILE="/local/melco3/taghianj/SAC_GCN/Controller/graph/main.py"

# FetchReach-v2
#tmux new-session -d -s fetchreach-0 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name FetchReach-v2 --exp-type graph --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 -ef 100 --cuda --seed 0"
#tmux new-session -d -s fetchreach-1 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name FetchReach-v2 --exp-type graph --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 -ef 100 --cuda --seed 1"
#tmux new-session -d -s fetchreach-2 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name FetchReach-v2 --exp-type graph --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 -ef 100 --cuda --seed 2"
tmux new-session -d -s fetchreach-3 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name FetchReach-v2 --exp-type graph --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 -ef 100 --cuda --seed 3"
tmux new-session -d -s fetchreach-4 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name FetchReach-v2 --exp-type graph --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 -ef 100 --cuda --seed 4"
#tmux new-session -d -s fetchreach-5 "CUDA_VISIBLE_DEVICES=3 python $MAIN_FILE --env-name FetchReach-v2 --exp-type graph --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 -ef 100 --cuda --seed 5"
#tmux new-session -d -s fetchreach-6 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name FetchReach-v2 --exp-type graph --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 -ef 100 --cuda --seed 6"
tmux new-session -d -s fetchreach-7 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name FetchReach-v2 --exp-type graph --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 -ef 100 --cuda --seed 7"
tmux new-session -d -s fetchreach-8 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name FetchReach-v2 --exp-type graph --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 -ef 100 --cuda --seed 8"
#tmux new-session -d -s fetchreach-9 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name FetchReach-v2 --exp-type graph --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 -ef 100 --cuda --seed 9"
