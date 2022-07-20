MAIN_FILE="/home/mehran/Documents/SAC_GCN/Controller/graph/main.py"
MAIN_FILE="/home/mehran/Documents/SAC_GCN/Controller/basic/main.py"

MAIN_FILE="/local/melco3/taghianj/SAC_GCN/Controller/graph/main.py"

MAIN_FILE="/home/taghianj/Documents/SAC_GCN/Controller/graph/main.py"
MAIN_FILE="/home/taghianj/Documents/SAC_GCN/Controller/basic/main.py"


tmux new-session -d -s walker2d-test "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name Walker2d-v2 --exp-type test --automatic_entropy_tuning True --num_episodes 30000 --start_steps 0 -dsf 5 --cuda --seed 0"
tmux new-session -d -s halfcheetah-test "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name HalfCheetah-v2 --exp-type test --automatic_entropy_tuning True --num_episodes 30000 --start_steps 0 -dsf 5 --cuda --seed 0"
tmux new-session -d -s fetchreach-test "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name FetchReach-v2 --exp-type standard --automatic_entropy_tuning True --num_episodes 30000 --start_steps 10000 -dsf 100 --cuda --seed 0"
